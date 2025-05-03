from __future__ import annotations

import time
import math
import random
from typing import List, Tuple, Any, Dict, Optional
import numpy as np
from dataclasses import dataclass
from torch import nn
import numba

from game_alt import GameRules, GameState, BoardType
from model import ZeroNetwork

__all__ = [
    "ZeroModel",
    "ZeroMCTSNode",
    "ZeroPlayer",
]

ZeroProbs = Tuple[List[float], float]
FILTERS, BLOCKS=64, 8

# BoardType = np.ndarray[Any, np.dtype[np.int64]]

class ZeroModel:
    """Policy‑Value network interface.

    `infer(board)` must return (priors, value):
      • priors: length‑4 list of probabilities for U,R,D,L.
      • value : scalar in [-1,1] estimating goodness for the *player*.
    """
    ACTIONS = 4  # up, right, down, left

    def __init__(self, n=4, m=4, k=20, filters=FILTERS, blocks=BLOCKS, model = None):
        pass

    def infer(self, board: BoardType) -> ZeroProbs:
        # *** Replace this with real NN inference ***
        priors = [1.0 / self.ACTIONS] * self.ACTIONS  # uniform policy
        import ipdb; ipdb.set_trace()
        # simple heuristic value: log2(max_tile)/11 ∈ [0,1]; scale to [-1,1]
        max_tile = board.max(initial=0)
        value = math.log2(max_tile + 1) / 11.0 * 2 - 1
        return priors, float(value)

@numba.njit(cache=True)
def to_one_hot(board: BoardType, k: int) -> np.ndarray:
    n, m = board.shape
    one_hot = np.zeros((1, k, m, n), dtype=np.float16)
    for i in range(n):
        for j in range(m):
            if board[i,j] > 0:
                one_hot[0,board[i,j], i,j] = 1.0
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

    def infer(self, board: BoardType) -> ZeroProbs:
        # convert board into one-hot encoding, where board is currently nxm with value in each square
        # should be 1xnxmxk with k=20 (max tile) with a 1 in the square of the value, else 0
        model_input = to_one_hot(board, 20)





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
        new_board, [action_tuple] = self.rules.add_random_tiles(self.state.board, return_action=True)
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

        deadline = time.time() + time_limit
        sims_done = 0
        while (simulations is None and time.time() < deadline) or (simulations is not None and sims_done < simulations):
            leaf, path = self._tree_search(root)
            leaf_value = self._evaluate_leaf(leaf)
            self._backprop(path, leaf_value)
            sims_done += 1

        print("sims done:", sims_done)
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

import torch


class ZeroTrainer:
    def __init__(self, model: ZeroModel, rules: GameRules, player: Optional[ZeroPlayer] = None):
        self.model = model
        self.rules = rules
        self.player = ZeroPlayer(model, rules) if player is None else player

    def train(self, epochs: int, batch_size: int, **kwargs):
        for i in range(epochs):
            samples = []
            for batch in range(batch_size):
                state = self.rules.get_initial_state()
                while not self.rules.is_terminal(state.board):
                    action, data = self.player.play(state, **kwargs)
                    samples.append((state, action, data))
                    new_board, score_gain = self.rules.apply_move(state.board, action)
                    new_board = self.rules.add_random_tiles(new_board)[0]
                    state = GameState(new_board, state.score + score_gain)
            # Update model with samples
            self.model.update(samples)





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
