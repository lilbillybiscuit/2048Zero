from __future__ import annotations

import math
import os
import random
import time
from datetime import datetime
from typing import List, Tuple, Any, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from game_alt import GameRules, GameState, BitBoard, GameRunner
from zeromodel import ZeroNetwork
from zeromonitoring import ZeroMonitor, EpochStats, print_epoch_summary

__all__ = [
    "ZeroMCTSNode",
    "ZeroPlayer",
]
device = 'cpu'
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")

ZeroProbs = Tuple[List[float], float]
FILTERS, BLOCKS = 8, 1
C_PUCT = math.sqrt(2)

class ZeroMCTSNode:
    PLAYER = 0
    CHANCE = 1

    def __init__(
        self,
        node_type: int,
        model: ZeroNetwork,
        rules: GameRules,
        state: GameState,
        parent: Optional["ZeroMCTSNode"] = None,
    ) -> None:
        self.parent: Optional[ZeroMCTSNode] = parent
        self.type: int = node_type
        self.model: ZeroNetwork = model
        self.rules: GameRules = rules
        self.score: np.int64 = state.score
        self.height, self.width = state.board.shape
        
        # Use the bitboard from GameState if available, otherwise create it
        if state.bitboard is not None:
            self.bitboard = state.bitboard
        else:
            self.bitboard = BitBoard.from_numpy(state.board)

        # children: key -> child node
        #   key == int (direction)  when PLAYER node
        #   key == int              when CHANCE node (using bitboard as key)
        self.children: Dict[Any, "ZeroMCTSNode"] = {}

        # statistics
        self.visits: int = 0  # N(s)
        self.value_sum: float = 0.0  # W(s) – sum of leaf evaluations

        if self.type == self.PLAYER:
            # We can use the actual board from GameState for this
            self.valid_moves: List[int] = rules.get_valid_moves(state.board)
            self.priors, _ = model.infer(np.expand_dims(state.board, axis=0))
            self.priors = self.priors[0]  # remove batch dim
            
            # mask illegal moves to zero prob, renormalise
            total_p = 0.0
            for a in range(4):
                if a not in self.valid_moves:
                    self.priors[a] = 0.0
                total_p += self.priors[a]
            if total_p == 0:
                # all moves illegal (should be terminal) – avoid div/0
                self.priors = np.array([1e-8] * 4)
                total_p = 4e-8
            self.priors = self.priors / total_p
        else:
            self.valid_moves = []  # unused
            self.priors = np.array([])      # unused

    def get_probs(self)-> ZeroProbs:
        p = []
        for i in range(4):
            p.append(self.children[i].visits if i in self.children else 0)
        # debug:
        assert sum(p) == self.visits, "visits don't match"
        p = [x/self.visits for x in p]
        value = self.value_sum / self.visits
        return p, value

    @property
    def is_terminal(self) -> bool:
        # Use thread-safe bitboard operation to check if terminal
        return not BitBoard.has_valid_moves(self.bitboard, self.height, self.width)

    @property
    def q_value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

    def expand_player_child(self) -> Tuple[int, "ZeroMCTSNode"]:
        assert self.type == self.PLAYER
        assert self.valid_moves, "no moves left to expand"
        action = self.valid_moves.pop()
        
        # Convert bitboard to numpy for rules operations
        np_board = BitBoard.to_numpy(self.bitboard, self.height, self.width)
        changed, gain, new_board = self.rules.simulate_move(np_board, action)
        # should always change because we filtered valid moves
        
        # Create new bitboard and GameState for child
        child_bitboard = BitBoard.from_numpy(new_board)
        child_state = GameState(new_board, self.score + gain, child_bitboard)
        
        # Create child node with the new state
        child = ZeroMCTSNode(self.CHANCE, self.model, self.rules, child_state, parent=self)
        self.children[action] = child
        return action, child

    def expand_chance_child(self) -> Tuple[Any, "ZeroMCTSNode"]:
        assert self.type == self.CHANCE
        
        # Convert bitboard to numpy for rules operations
        np_board = BitBoard.to_numpy(self.bitboard, self.height, self.width)
        new_board, action_tuple = self.rules.add_random_tiles(np_board, return_action=True)
        
        # Create new bitboard and GameState for child
        child_bitboard = BitBoard.from_numpy(new_board)
        child_state = GameState(new_board, self.score, child_bitboard)
        
        # Create child node with the new state
        child = ZeroMCTSNode(self.PLAYER, self.model, self.rules, child_state, parent=self)
        
        # Use the bitboard as key for chance nodes
        self.children[child_bitboard] = child
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

    def __init__(self, model: ZeroNetwork, rules: GameRules, dirichlet_eps: float = 0.25, dirichlet_alpha: float = 0.3):
        self.model = model
        self.rules = rules
        self.dir_eps = dirichlet_eps
        self.dir_alpha = dirichlet_alpha
        # Cache for board evaluation to avoid redundant neural network inference
        self._value_cache = {}

    def play(self, state: GameState, time_limit: float = 0.5, simulations: Optional[int] = None) -> Tuple[int, ZeroProbs]:
        # Create the root node with the GameState
        root = ZeroMCTSNode(ZeroMCTSNode.PLAYER, self.model, self.rules, state)
        self._add_dirichlet_noise(root)

        # Clear the value cache at the start of each play decision
        self._value_cache.clear()

        deadline = time.time() + time_limit+2
        sims_done = 0
        while (simulations is None and time.time() < deadline) or (simulations is not None and sims_done < simulations):
            leaf, path = self._tree_search(root)
            leaf_value = self._evaluate_leaf(leaf)
            self._backprop(path, leaf_value)
            sims_done += 1

        best_action = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
        return best_action, root.get_probs()

    def _add_dirichlet_noise(self, root: ZeroMCTSNode):
        if self.dir_eps <= 0:
            return
        noise = np.random.dirichlet([self.dir_alpha] * 4)
        root.priors = [(1 - self.dir_eps) * p + self.dir_eps * n for p, n in zip(root.priors, noise)]

    @staticmethod
    def _puct_score(q_value: float, prior: float, sqrt_parent_visits: float, child_visits: int) -> float:
        u = C_PUCT * prior * sqrt_parent_visits / (1 + child_visits)
        return q_value + u

    def _tree_search(self, root: ZeroMCTSNode) -> Tuple[ZeroMCTSNode, List[ZeroMCTSNode]]:
        node = root
        path = [node]

        while not node.is_terminal:
            if node.type == ZeroMCTSNode.PLAYER:
                if node.valid_moves:
                    _, node = node.expand_player_child()
                    path.append(node)
                    break
                else:
                    sqrt_parent = math.sqrt(node.visits)
                    best_score = float("-inf")
                    best_child = None

                    for a, child in node.children.items():
                        p = node.priors[a]
                        score = self._puct_score(child.q_value, p, sqrt_parent, child.visits)
                        if score > best_score:
                            best_score = score
                            best_child = child

                    node = best_child
                    path.append(node)
            else:
                _, node = node.expand_chance_child()
                path.append(node)

        return node, path

    def _evaluate_leaf(self, node: ZeroMCTSNode) -> float:
        if node.is_terminal:
            # TODO: convert bitboard to numpy only if needed for max_tile calculation
            np_board = BitBoard.to_numpy(node.bitboard, node.height, node.width)
            max_tile = self.rules.get_max_tile(np_board)
            return math.log2(node.score) if node.score > 0 else 0.0

        if node.bitboard in self._value_cache:
            return self._value_cache[node.bitboard]

        np_board = BitBoard.to_numpy(node.bitboard, node.height, node.width)
        board_np = np.expand_dims(np_board, axis=0)
        _, value = self.model.infer(board_np)
        result = float(value.item())

        self._value_cache[node.bitboard] = result
        return result

    def _backprop(self, path: List[ZeroMCTSNode], leaf_value: float):
        for n in reversed(path):
            n.visits += 1
            n.value_sum += leaf_value

class ZeroTrainer:
    def __init__(self, 
                 model: ZeroNetwork, 
                 rules: GameRules, 
                 player: Optional[ZeroPlayer] = None,
                 use_wandb: bool = True,
                 project_name: str = "2048-zero",
                 experiment_name: Optional[str] = None):
        """Initialize the Zero trainer with model, rules and wandb tracking
        
        Args:
            model: Neural network model
            rules: Game rules
            player: Optional pre-configured player (will create one if None)
            use_wandb: Whether to use wandb for tracking (default: True)
            project_name: Project name for wandb
            experiment_name: Experiment name for wandb (auto-generated if None)
        """
        self.model = model
        self.rules = rules
        self.player = ZeroPlayer(model, rules) if player is None else player
        
        # Generate experiment name if none provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            board_size = f"{rules.height}x{rules.width}"
            model_size = f"f{model.conv1.out_channels}_b{len(model.blocks)}"
            experiment_name = f"zero_{board_size}_{model_size}_{timestamp}"
            
        # Initialize monitor
        config = {
            "model_filters": model.conv1.out_channels,
            "model_blocks": len(model.blocks),
            "model_k": model.k,
            "board_height": rules.height,
            "board_width": rules.width,
            "dirichlet_eps": self.player.dir_eps,
            "dirichlet_alpha": self.player.dir_alpha,
        }
        
        self.monitor = ZeroMonitor(
            use_wandb=use_wandb,
            project_name=project_name,
            experiment_name=experiment_name,
            config=config
        )

    def _collect_selfplay_data(self, games_per_epoch: int, simulations: int = 50, num_cpus: int = os.cpu_count()) -> tuple[list[Any], dict[str, Any]]:
        """Collect self-play data for training
        
        Args:
            games_per_epoch: Number of self-play games to generate
            simulations: Number of MCTS simulations per move
            
        Returns:
            Tuple of (samples, statistics) where samples is a list of (board, policy, value) tuples
            and statistics is a dictionary of game statistics
        """
        samples = []
        
        # Initialize statistics tracking with direction names
        stats = EpochStats(
            height=self.rules.height, 
            width=self.rules.width,
            direction_names=self.rules.DIRECTION_NAMES
        )
        
        # Create progress bar for games
        game_pbar = tqdm(range(games_per_epoch), desc='Self-play')
        
        for game_idx in game_pbar:
            # Get initial state from rules (already includes bitboard)
            state = self.rules.get_initial_state()
            trajectory = []
            turn_count = 0
            action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Track action distribution

            while not self.rules.is_terminal(state.board):
                turn_count += 1
                action, (pi_probs, value_est) = self.player.play(state, simulations=simulations)
                action_counts[action] += 1
                trajectory.append((state.board.copy(), pi_probs))

                # Calculate statistics
                max_tile = self.rules.get_max_tile(state.board)
                game_pbar.set_postfix(
                    game=f"{game_idx + 1}/{games_per_epoch}", 
                    turn=turn_count,
                    score=state.score, 
                    max_tile=max_tile, 
                    val_est=f"{value_est:.2f}"
                )

                new_board, gain = self.rules.apply_move(state.board, action)
                new_board = self.rules.add_random_tiles(new_board)
                new_bitboard = BitBoard.from_numpy(new_board)
                state = GameState(new_board, state.score + gain, new_bitboard)

            max_tile = self.rules.get_max_tile(state.board)
            z = math.log2(int(max_tile) + 1) / 11.0 * 2 - 1  # scaled into [-1,1]

            stats.update_game_stats(
                score=state.score,
                max_tile=max_tile,
                turns=turn_count,
                action_counts=action_counts,
                board=state.board
            )
            
            # Log individual game to wandb if enabled
            if self.monitor.use_wandb:
                # Calculate action distribution as percentages
                total_actions = sum(action_counts.values())
                action_dist = {
                    f"action_{self.rules.get_direction_name(a)}": count / total_actions 
                    for a, count in action_counts.items() if total_actions > 0
                }
                
                self.monitor.log_game_stats(
                    score=state.score,
                    max_tile=max_tile,
                    turns=turn_count,
                    value_target=z,
                    trajectory_length=len(trajectory),
                    action_dist=action_dist
                )

            for (board_t, pi_t) in trajectory:
                samples.append((board_t, pi_t, z))

        stats.update_samples(len(samples))

        return samples, stats.get_stats()
    
    def _update_network(self, samples: List[Tuple], optimizer: torch.optim.Optimizer, 
                     batch_size: int, epoch: int) -> Tuple[float, float, float]:
        """Update neural network weights using collected samples in batches
        
        Args:
            samples: List of (board, policy, value) tuples
            optimizer: PyTorch optimizer
            batch_size: Training batch size
            epoch: Current epoch number (for logging)
            
        Returns:
            Tuple of (average_loss, average_policy_loss, average_value_loss)
        """
        self.model.train()

        random.shuffle(samples)

        num_samples = len(samples)
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        # Create progress bar for batch processing
        batch_pbar = tqdm(range(num_batches), desc='Training batches')
        
        # Process samples in batches
        for i in batch_pbar:
            # Get batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch = samples[start_idx:end_idx]
            
            # Unpack batch
            boards, pis, zs = zip(*batch)
            
            # Stack boards and convert to numpy array
            boards_array = np.stack(boards, axis=0)  # Shape: (batch_size, height, width)

            state_tensor = self.model.to_onehot(boards_array).to(device)
            pi_tensor = torch.tensor(pis, dtype=torch.float32).to(device)
            z_tensor = torch.tensor(zs, dtype=torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad()

            p_pred, v_pred = self.model(state_tensor)

            policy_loss = -(pi_tensor * torch.log(p_pred + 1e-8)).sum(dim=1).mean()
            value_loss = torch.nn.functional.mse_loss(v_pred, z_tensor)
            loss = policy_loss + value_loss
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Accumulate losses
            batch_size_actual = end_idx - start_idx
            total_loss += loss.item() * batch_size_actual
            total_policy_loss += policy_loss.item() * batch_size_actual
            total_value_loss += value_loss.item() * batch_size_actual
            
            # Update progress bar
            current_loss = loss.item()
            current_policy_loss = policy_loss.item()
            current_value_loss = value_loss.item()
            batch_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'π_loss': f'{current_policy_loss:.4f}',
                'v_loss': f'{current_value_loss:.4f}'
            })
            
            # Log batch metrics using the monitor's internal global step
            self.monitor.log_batch_stats(
                batch_loss=current_loss,
                batch_policy_loss=current_policy_loss,
                batch_value_loss=current_value_loss,
                batch_size=batch_size_actual,
                epoch=epoch
            )
        
        # Calculate average losses
        avg_loss = total_loss / num_samples
        avg_policy_loss = total_policy_loss / num_samples
        avg_value_loss = total_value_loss / num_samples
        
        return avg_loss, avg_policy_loss, avg_value_loss

    
    def train(self, epochs: int = 100, games_per_epoch: int = 32, batch_size: int = 64, 
             lr: float = 0.001, log_interval: int = 1, checkpoint_interval: int = 10, 
             resume: bool = False, **kwargs):
        """Train the model using self-play for the specified number of epochs
        
        Args:
            epochs: Number of training epochs
            games_per_epoch: Number of self-play games to generate per epoch
            batch_size: Batch size for network updates
            lr: Learning rate for optimizer
            log_interval: How often to log detailed metrics (every N epochs)
            checkpoint_interval: How often to save model checkpoints (every N epochs)
            resume: Whether to resume training from previous state
            **kwargs: Additional arguments including wandb config overrides
        """
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
            
        # Update monitor config with training parameters
        config_update = {
            "epochs": epochs,
            "games_per_epoch": games_per_epoch,
            "batch_size": batch_size,
            "learning_rate": lr,
            "scheduler_step_size": 20,
            "scheduler_gamma": 0.5,
            "weight_decay": 1e-4,
            # Add any additional kwargs
            **{k: v for k, v in kwargs.items() if k != 'use_wandb'}
        }
        self.monitor.config.update(config_update)
        
        # Initialize wandb with the model
        if self.monitor.use_wandb and not self.monitor.wandb_initialized:
            self.monitor.init_wandb(watch_model=self.model)
        
        # Initialize optimizer with specified learning rate
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        
        # Optional: Add learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # Create a run tracker file to allow resume
        self.monitor.run_data.update({
            "model_config": {
                "filters": self.model.conv1.out_channels,
                "blocks": len(self.model.blocks),
                "k": self.model.k
            },
            "board_config": {
                "height": self.rules.height,
                "width": self.rules.width
            },
            "training_config": {
                "epochs": epochs,
                "games_per_epoch": games_per_epoch,
                "batch_size": batch_size,
                "lr": lr
            }
        })
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Set model to evaluation mode during self-play
            self.model.eval()
            
            # 1. Collect training data with self-play
            print(f"Epoch {epoch+1}/{epochs}: Collecting self-play data...")
            samples, epoch_stats = self._collect_selfplay_data(
                games_per_epoch=games_per_epoch,
                simulations=kwargs.get('simulations', 50)
            )
            
            # Log self-play statistics to wandb
            self.monitor.log_selfplay_stats(epoch_stats, epoch)
            
            # Calculate statistics for display
            avg_score = epoch_stats['total_score'] / epoch_stats['games'] if epoch_stats['games'] > 0 else 0
            avg_tile = epoch_stats['total_tile'] / epoch_stats['games'] if epoch_stats['games'] > 0 else 0
            avg_turns = epoch_stats['total_turns'] / epoch_stats['games'] if epoch_stats['games'] > 0 else 0
            
            # 2. Update neural network if samples were collected
            loss, pi_loss, v_loss = None, None, None
            if samples:
                print(f"Training on {len(samples)} samples with batch size {batch_size}...")
                # Set model to training mode
                self.model.train()
                
                # Update network with batched training
                # Don't use step_offset anymore since we use global_step in the function
                loss, pi_loss, v_loss = self._update_network(
                    samples, optimizer, batch_size, epoch=epoch
                )
                
                # Step the learning rate scheduler
                scheduler.step()
                
                # Log training metrics
                self.monitor.log_training_stats(
                    loss=loss,
                    policy_loss=pi_loss,
                    value_loss=v_loss,
                    learning_rate=scheduler.get_last_lr()[0],
                    samples=len(samples),
                    epoch=epoch
                )
            
            # Report results
            print_epoch_summary(
                EpochStats(self.rules.height, self.rules.width),  # Temporary stats object just for formatting
                epoch, epochs, loss, pi_loss, v_loss, 
                scheduler.get_last_lr()[0] if scheduler else None
            )
                
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Prepare epoch data for run tracker
            epoch_data = {
                "epoch": epoch,
                "samples": len(samples),
                "max_tile": epoch_stats['max_tile'],
                "max_score": epoch_stats['max_score'],
                "avg_tile": avg_tile,
                "avg_score": avg_score,
                "avg_turns": avg_turns,
                "loss": loss if samples else None,
                "policy_loss": pi_loss if samples else None,
                "value_loss": v_loss if samples else None,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch_time_seconds": epoch_time
            }
            
            # Update run tracker
            self.monitor.update_run_data(epoch_data)
            
            # Save run tracker
            run_path = f"checkpoints/{self.monitor.experiment_name}_run.json"
            self.monitor.save_run_data(run_path)
                
            # 3. Save checkpoint periodically
            if epoch % checkpoint_interval == 0 or epoch == epochs - 1:
                checkpoint_path = f"checkpoints/{self.monitor.experiment_name}_epoch_{epoch}.pth"
                self.model.save(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                
                # Log model to wandb
                self.monitor.log_checkpoint(checkpoint_path)
            
            # 4. Log example boards to wandb periodically
            if self.monitor.use_wandb and epoch % log_interval == 0 or epoch == epochs - 1:
                # Extract some example boards from training samples
                if samples:
                    sample_indices = list(range(0, len(samples), max(1, len(samples)//8)))[:8]
                    board_examples = [samples[i][0] for i in sample_indices]
                    self.monitor.log_board_examples(board_examples, f"epoch_{epoch}_boards")
                    
        # Finish wandb run when training is complete
        self.monitor.finish()
            
        return self.monitor.run_data



if __name__ == "__main__":
    # Run a simple demo game
    rules = GameRules(num_spawn_tiles_per_move=1)
    model = ZeroNetwork(4, 4, 16).to(device)
    agent = ZeroPlayer(model, rules)
    runner = GameRunner(rules)
    
    print("\n====== 2048 Zero Demo ======")
    print("Playing a quick demo game with a randomly initialized model")
    print(runner.render_ascii())
    
    # Play until game over or max 10 turns for the demo
    turn = 0
    while not runner.is_game_over() and turn < 10:
        turn += 1
        
        # Get current board and score
        current_board = runner.get_board()
        current_score = runner.get_score()
        max_tile = rules.get_max_tile(current_board)
        
        # Create GameState with bitboard for MCTS
        bitboard = BitBoard.from_numpy(current_board)
        state = GameState(current_board, current_score, bitboard)
        
        # Get action from agent with short time limit for the demo
        print(f"\nThinking about move {turn}...")
        action, (probs, value) = agent.play(state, time_limit=0.2)
        action_name = rules.get_direction_name(action)
        
        # Execute the action
        gain, _ = runner.move(action)
        runner.generate_tiles()
        
        # Display the result
        print(f"Move: {action_name} (value est: {value:.2f})")
        print(f"Score: {current_score} (+{gain}) | Max Tile: {max_tile}")
        print(runner.render_ascii())
    
    print("\n====== Demo Finished ======")
    print(f"Final score: {runner.get_score()}")
    print(f"Max tile: {runner.get_max_tile()}")
    print(f"Turns played: {turn}")
    print("\nTo run training: python train.py")
    print("To play a full game: python play.py --simulations 100")
