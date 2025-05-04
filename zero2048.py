from __future__ import annotations

import time
import math
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict, Optional, Union
from torch import nn
import numba
from tqdm import tqdm
import os
from game_alt import GameRules, GameState, BoardType
from model import ZeroNetwork
import torch
import multiprocessing

__all__ = [
    "ZeroModel",
    "ZeroMCTSNode",
    "ZeroPlayer",
]
device = 'cpu'
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.backends.cuda.is_available():
    device = torch.device("cuda")

ZeroProbs = Tuple[List[float], float]
FILTERS, BLOCKS=8, 1

# BoardType = np.ndarray[Any, np.dtype[np.int64]]

class ZeroModel:
    """Policy‑Value network interface.

    `infer(board)` must return (priors, value):
      • priors: length‑4 list of probabilities for U,R,D,L.
      • value : scalar in [-1,1] estimating goodness for the *player*.
    """
    ACTIONS = 4  # up, right, down, left

    def __init__(self, n=4, m=4, k=20, filters=FILTERS, blocks=BLOCKS, model = None):
        if model is None:
            self.model = ZeroNetwork(n, m, k, filters, blocks)

    def infer(self, board: BoardType) -> ZeroProbs:
        # *** Replace this with real NN inference ***
        raise NotImplementedError
        # priors = [1.0 / self.ACTIONS] * self.ACTIONS  # uniform policy
        # # simple heuristic value: log2(max_tile)/11 ∈ [0,1]; scale to [-1,1]
        # max_tile = board.max(initial=0)
        # value = math.log2(max_tile + 1) / 11.0 * 2 - 1
        # return priors, float(value)


@numba.njit(cache=True)
def to_one_hot_inplace(board: BoardType, result: np.ndarray, idx, n, m, k: int) -> None:
    for i in range(n):
        for j in range(m):
            tile_value = board[i, j]
            if tile_value > 0:
                channel_idx = int(math.log2(tile_value) - 1 + 0.001)
                if 1 <= channel_idx < k:
                    result[channel_idx, i, j] = 1.0


def to_one_hot(board: Union[BoardType, List[BoardType]], k: int) -> np.ndarray:
    if not isinstance(board, list):
        board = [board]

    n, m = board[0].shape  # n = height (rows), m = width (cols)
    # Standard shape: (batch, channels, height, width)
    one_hot = np.zeros((len(board), k, n, m), dtype=np.float32)

    for i, b in enumerate(board):
        to_one_hot_inplace(b, one_hot[i], 0, n, m, k)

    return one_hot

class ZeroModelNN(ZeroModel):
    """Policy‑Value network interface.

    `infer(board)` must return (priors, value):
      • priors: length‑4 list of probabilities for U,R,D,L.
      • value : scalar in [-1,1] estimating goodness for the *player*.
    """
    ACTIONS = 4  # up, right, down, left

    def __init__(self, n=4, m=4, k=20, filters=FILTERS, blocks=BLOCKS, model: nn.Module = None):
        super(ZeroModelNN, self).__init__()
        self.model = model if model else ZeroNetwork(n, m, k, filters, blocks)

    def eval(self):
        self.model.eval()

    def set_model(self, model: nn.Module):
        self.model = model

    def save(self, path = None):
        if path is None:
            path = "model.pth"
        self.model.save(path)

    def infer(self, board: BoardType) -> ZeroProbs:
        # convert board into one-hot encoding, where board is currently nxm with value in each square
        # should be 1xnxmxk with k=20 (max tile) with a 1 in the square of the value, else 0
        model_input = to_one_hot(board, 20)
        # convert to tensor
        model_input = torch.from_numpy(model_input).float().to('mps')
        # run through model
        with torch.no_grad():
            priors, value = self.model(model_input)
        # convert to numpy
        priors = priors.cpu().numpy()[0]
        value = value.cpu().numpy()[0]
        # convert to list
        priors = priors.tolist()
        return priors, float(value)


C_PUCT = math.sqrt(2)  # exploration constant (tune later)

class ZeroMCTSNode:
    PLAYER = 0
    CHANCE = 1

    def __init__(
        self,
        node_type: int,
        model: ZeroModel,
        rules: GameRules,
        state: GameState,
        parent: Optional["ZeroMCTSNode"] = None,
    ) -> None:
        self.parent: Optional[ZeroMCTSNode] = parent
        self.type: int = node_type
        self.model: ZeroModel = model
        self.rules: GameRules = rules
        self.state: GameState = state

        # children: key -> child node
        #   key == int (direction)  when PLAYER node
        #   key == (r,c,val)        when CHANCE node
        self.children: Dict[Any, "ZeroMCTSNode"] = {}

        # statistics
        self.visits: int = 0  # N(s)
        self.value_sum: float = 0.0  # W(s) – sum of leaf evaluations

        # bookkeeping for PLAYER nodes
        if self.type == self.PLAYER:
            self.valid_moves: List[int] = rules.get_valid_moves(state.board)
            # priors fixed at expansion time
            self.priors, _ = model.infer(state.board)
            # mask illegal moves to zero prob, renormalise
            total_p = 0.0
            for a in range(4):
                if a not in self.valid_moves:
                    self.priors[a] = 0.0
                total_p += self.priors[a]
            if total_p == 0:
                # all moves illegal (should be terminal) – avoid div/0
                self.priors = [1e-8] * 4
                total_p = 4e-8
            self.priors = [p / total_p for p in self.priors]
        else:
            self.valid_moves = []  # unused
            self.priors = []      # unused

    def get_probs(self)-> ZeroProbs:
        p = [child.visits for child in self.children.values()]
        # debug:
        assert sum(p) == self.visits, "visits don't match"
        p = [x/self.visits for x in p]
        value = self.value_sum / self.visits
        return p, value

    @property
    def is_terminal(self) -> bool:
        return self.rules.is_terminal(self.state.board)

    @property
    def q_value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

    def expand_player_child(self) -> Tuple[int, "ZeroMCTSNode"]:
        assert self.type == self.PLAYER
        assert self.valid_moves, "no moves left to expand"
        action = self.valid_moves.pop()
        changed, gain, new_board = self.rules.simulate_move(self.state.board, action)
        # should always change because we filtered valid moves
        child_state = GameState(new_board, self.state.score + gain)
        child = ZeroMCTSNode(self.CHANCE, self.model, self.rules, child_state, parent=self)
        self.children[action] = child
        return action, child

    def expand_chance_child(self) -> Tuple[Any, "ZeroMCTSNode"]:
        assert self.type == self.CHANCE
        new_board, action_tuple = self.rules.add_random_tiles(self.state.board, return_action=True)
        child_state = GameState(new_board, self.state.score)
        child = ZeroMCTSNode(self.PLAYER, self.model, self.rules, child_state, parent=self)

        self.children[new_board.tobytes()] = child
        return action_tuple, child

    def select_puct_child(self) -> Tuple[int, "ZeroMCTSNode"]:
        assert self.type == self.PLAYER and self.children, "PUCT selection invalid"
        best_key, best_child = None, None
        best_score = float("-inf")
        sqrt_parent = math.sqrt(self.visits)
        for a, child in self.children.items():
            p = self.priors[a]
            u = C_PUCT * p * sqrt_parent / (1 + child.visits)
            score = child.q_value + u
            if score > best_score:
                best_score = score
                best_key, best_child = a, child
        return best_key, best_child

class ZeroPlayer:
    """Single‑player MCTS agent.

    Usage:
        rules  = GameRules()
        model  = ZeroModel()
        player = ZeroPlayer(model, rules)
        action = player.play(start_state, time_limit=0.1)
    """

    def __init__(self, model: ZeroModel, rules: GameRules, dirichlet_eps: float = 0.25, dirichlet_alpha: float = 0.3):
        self.model = model
        self.rules = rules
        self.dir_eps = dirichlet_eps
        self.dir_alpha = dirichlet_alpha

    def play(self, state: GameState, time_limit: float = 0.5, simulations: Optional[int] = None) -> Tuple[int, ZeroProbs]:
        root = ZeroMCTSNode(ZeroMCTSNode.PLAYER, self.model, self.rules, state)
        self._add_dirichlet_noise(root)

        deadline = time.time() + time_limit+2
        sims_done = 0
        x =time.time() < deadline
        while (simulations is None and time.time() < deadline) or (simulations is not None and sims_done < simulations):
            leaf, path = self._tree_search(root)
            leaf_value = self._evaluate_leaf(leaf)
            self._backprop(path, leaf_value)
            sims_done += 1

        # pick action with highest visit count
        best_action = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
        # retrieve probabilities and value
        return best_action, root.get_probs()

    def _add_dirichlet_noise(self, root: ZeroMCTSNode):
        if self.dir_eps <= 0:
            return
        noise = np.random.dirichlet([self.dir_alpha] * 4)
        root.priors = [ (1 - self.dir_eps) * p + self.dir_eps * n for p, n in zip(root.priors, noise) ]

    def _tree_search(self, root: ZeroMCTSNode) -> Tuple[ZeroMCTSNode, List[ZeroMCTSNode]]:
        node = root
        path = [node]
        while not node.is_terminal:
            if node.type == ZeroMCTSNode.PLAYER:
                # expansion phase
                if node.valid_moves:
                    _, node = node.expand_player_child()
                    path.append(node)
                    break
                else:
                    _, node = node.select_puct_child()
                    path.append(node)
            else:  # chance node, always expand exactly one child
                _, node = node.expand_chance_child()
                path.append(node)
        return node, path

    def _evaluate_leaf(self, node: ZeroMCTSNode) -> float:
        if node.is_terminal:
            max_tile = self.rules.get_max_tile(node.state.board)
            return math.log2(node.state.score) # same scaling as model
        _, value = self.model.infer(node.state.board)
        return float(value)

    def _backprop(self, path: List[ZeroMCTSNode], leaf_value: float):
        for n in reversed(path):
            n.visits += 1
            n.value_sum += leaf_value
            # no sign flip – single player

class ZeroTrainer:
    def __init__(self, model: ZeroModelNN, rules: GameRules, player: Optional[ZeroPlayer] = None):
        self.model = model
        self.rules = rules
        self.player = ZeroPlayer(model, rules) if player is None else player

    def train(self, epochs: int, batch_size: int, **kwargs):
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        model = self.model.model
        model.train()
        model = model.to('mps')
        optimizer = torch.optim.SGD(self.model.model.parameters(), lr=0.01, momentum=0.9)

        for epoch in range(epochs):
            samples = []
            epoch_stats = {
                'max_score': 0,
                'max_tile': 0,
                'max_turns': 0,
                'total_score': 0,
                'total_tile': 0,
                'total_turns': 0,
                'games': 0
            }

            # CPU HEAVY

            # Use tqdm for the games in each epoch
            game_pbar = tqdm(range(batch_size), desc=f'Epoch {epoch + 1}/{epochs}')


            for game_idx in game_pbar:
                state = self.rules.get_initial_state()
                trajectory = []
                turn_count = 0

                # play until terminal
                while not self.rules.is_terminal(state.board):
                    turn_count += 1
                    action, (pi_probs, value_est) = self.player.play(state, simulations=1000)
                    trajectory.append((state, pi_probs))

                    # Update progress bar with current game stats
                    max_tile = self.rules.get_max_tile(state.board)
                    game_pbar.set_postfix(
                        game=f"{game_idx + 1}/{batch_size}",
                        turn=turn_count,
                        score=state.score,
                        max_tile=max_tile,
                        val_est=f"{value_est:.2f}"
                    )

                    new_board, gain = self.rules.apply_move(state.board, action)
                    new_board = self.rules.add_random_tiles(new_board)
                    state = GameState(new_board, state.score + gain)

                # Game completed - record final stats
                max_tile = self.rules.get_max_tile(state.board)
                z = math.log2(int(max_tile) + 1) / 11.0 * 2 - 1  # scaled into [-1,1]

                # Update epoch statistics
                epoch_stats['games'] += 1
                epoch_stats['max_score'] = max(epoch_stats['max_score'], state.score)
                epoch_stats['max_tile'] = max(epoch_stats['max_tile'], max_tile)
                epoch_stats['max_turns'] = max(epoch_stats['max_turns'], turn_count)
                epoch_stats['total_score'] += state.score
                epoch_stats['total_tile'] += max_tile
                epoch_stats['total_turns'] += turn_count

                # Add to training samples
                for (s_t, pi_t) in trajectory:
                    samples.append((s_t, pi_t, z))

            # Calculate averages
            avg_score = epoch_stats['total_score'] / epoch_stats['games'] if epoch_stats['games'] > 0 else 0
            avg_tile = epoch_stats['total_tile'] / epoch_stats['games'] if epoch_stats['games'] > 0 else 0
            avg_turns = epoch_stats['total_turns'] / epoch_stats['games'] if epoch_stats['games'] > 0 else 0

            ## END CPU HEAVY

            ## GPU HEAVY
            # Training step
            if samples:
                states, pis, zs = zip(*samples)
                boards = [to_one_hot(state.board, 20)[0] for state in states]
                state_tensor = torch.tensor(np.stack(boards), dtype=torch.float32).to('mps')
                pi_tensor = torch.tensor(pis, dtype=torch.float32).to('mps')  # N x 4
                z_tensor = torch.tensor(zs, dtype=torch.float32).unsqueeze(1).to('mps')  # N x 1

                optimizer.zero_grad()
                p_pred, v_pred = model(state_tensor)
                policy_loss = -(pi_tensor * torch.log(p_pred + 1e-8)).sum(dim=1).mean()
                value_loss = torch.nn.functional.mse_loss(v_pred, z_tensor)
                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()

                # Create a summary tqdm bar for the epoch results
                epoch_summary = tqdm(total=0, bar_format='{desc}',
                                     desc=f"Epoch {epoch + 1}: loss={loss.item():.4f} (π:{policy_loss:.4f}, v:{value_loss:.4f}) | "
                                          f"MaxTile: {epoch_stats['max_tile']} | MaxScore: {epoch_stats['max_score']} | "
                                          f"AvgTile: {avg_tile:.1f} | AvgScore: {avg_score:.1f} | AvgTurns: {avg_turns:.1f}")
                epoch_summary.close()
            ## END GPU HEAVY
            # Save checkpoints
            if epoch % 10 == 0:
                self.model.save(f"checkpoints/model_epoch_{epoch}.pth")



if __name__ == "__main__":
    rules = GameRules()
    model = ZeroModel()
    agent = ZeroPlayer(model, rules)

    runner = game_runner = __import__("game_alt").GameRunner(rules)
    turn = 0
    while not runner.is_game_over():
        turn += 1
        action, _ = agent.play(GameState(runner.get_board(), runner.get_score()), time_limit=0.2)
        gain, _ = runner.move(action)
        runner.generate_tiles()
        print(f"Turn {turn} – action {action}, gain {gain}\n{runner.render_ascii()}")
