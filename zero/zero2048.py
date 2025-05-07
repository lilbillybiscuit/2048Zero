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

from zero import *

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

    def __init__(self, model: ZeroNetwork, rules: GameRules, dirichlet_eps: float = 0.25, dirichlet_alpha: float = 0.5, temperature: float = 1.0):
        """
        Initialize the ZeroPlayer with model, rules and search parameters
        
        Args:
            model: Neural network model
            rules: Game rules
            dirichlet_eps: Weight of Dirichlet noise added to root prior (0.0-1.0)
            dirichlet_alpha: Concentration parameter for Dirichlet distribution (lower=peakier)
            temperature: Temperature for action selection (1.0=normal, <1.0=more deterministic)
        """
        self.model = model
        self.rules = rules
        self.dir_eps = dirichlet_eps
        self.dir_alpha = dirichlet_alpha
        self.temperature = temperature
        # Cache for board evaluation to avoid redundant neural network inference
        self._value_cache = {}

    def play(self, state: GameState, time_limit: float = 0.5, simulations: Optional[int] = None, temperature: Optional[float] = None) -> Tuple[int, ZeroProbs]:
        """
        Run MCTS search and select an action
        
        Args:
            state: Current game state
            time_limit: Time limit in seconds (ignored if simulations is provided)
            simulations: Number of MCTS simulations to run
            temperature: Override the default temperature for this move
        
        Returns:
            Tuple of (selected_action, (action_probabilities, value_estimate))
        """
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

        # Temperature-based action selection
        temp = temperature if temperature is not None else self.temperature
        # For turn count, we'll use a simple heuristic based on score if history not available
        # In 2048, score roughly corresponds to number of moves made
        turn_count = int(state.score / 10) if not hasattr(state, 'get_history') else len(state.get_history())
        
        # Use temperature-based selection for exploration vs. exploitation
        # Lower temperature for later game stages (more deterministic)
        if turn_count >= 15:
            temp = min(temp, 0.5)  # More focused play in mid-to-late game

        if temp < 0.05 or turn_count > 30:
            # Deterministic selection for very low temperature or endgame
            best_action = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
        else:
            # Sample based on visit count distribution and temperature
            visits = np.array([child.visits for _, child in sorted(root.children.items())])
            actions = np.array(sorted(root.children.keys()))
            
            # Apply temperature scaling
            if temp != 1.0:
                visits = visits ** (1.0 / temp)
                
            # Normalize to probabilities
            probs = visits / visits.sum()
            
            # Random sampling based on the distribution
            best_action = np.random.choice(actions, p=probs)
            
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
            # Use the same reward function as training for consistency
            from zero.reward_functions import default_reward_func
            # Convert bitboard to numpy for max_tile calculation
            np_board = BitBoard.to_numpy(node.bitboard, node.height, node.width)
            # Create game stats structure matching what's used in training
            game_stats = {
                'score': node.score,
                'max_tile': node.rules.get_max_tile(np_board)
            }
            # Calculate reward using same function as training
            z, _ = default_reward_func(node, game_stats)
            return z

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
                 experiment_name: Optional[str] = None,
                 reward_function = None):
        """Initialize the Zero trainer with model, rules and wandb tracking
        
        Args:
            model: Neural network model
            rules: Game rules
            player: Optional pre-configured player (will create one if None)
            use_wandb: Whether to use wandb for tracking (default: True)
            project_name: Project name for wandb
            experiment_name: Experiment name for wandb (auto-generated if None)
            reward_function: Custom function to calculate reward value with signature:
                           reward_function(state, game_stats) -> (reward_value, reward_name)
        """
        self.model = model
        self.rules = rules
        self.player = ZeroPlayer(model, rules) if player is None else player
        self.reward_function = reward_function
        
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
            "reward_function": "custom" if self.reward_function else "default_score"
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
            
            # Calculate game statistics for reward function
            game_stats = {
                'score': state.score,
                'max_tile': max_tile,
                'turns': turn_count,
                'action_counts': action_counts,
                'final_board': state.board
            }
            
            # Define default reward functions if none provided
            def default_score_reward(state, stats):
                """Default score-based reward"""
                score = stats['score']
                # Use raw log score without normalization to [-1,1]
                z = math.log(score + 100)
                return z, "unbounded_score"
            
            # Use provided reward function or default to score reward
            if self.reward_function:
                z, reward_type = self.reward_function(state, game_stats)
            else:
                # Default to score reward (not max tile)
                z, reward_type = default_score_reward(state, game_stats)
                
            # Store reward info in statistics
            game_stats['reward'] = z
            game_stats['reward_type'] = reward_type
            
            # Record the reward type in the epoch stats for logging
            stats.update_reward_type(reward_type)

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

            state_tensor = self.model.to_onehot(boards_array, device=device)
            pi_tensor = torch.tensor(pis, dtype=torch.float32, device=device)
            z_tensor = torch.tensor(zs, dtype=torch.float32, device=device).unsqueeze(1)
            
            # Periodically check policy target distribution (do this rarely to avoid slowing training)
            if epoch > 0 and epoch % 10 == 0 and i == 0:
                # Calculate average policy entropy to check if targets are too deterministic
                target_entropy = -(pi_tensor * torch.log(pi_tensor + 1e-8)).sum(dim=1).mean().item()
                print(f"\nEpoch {epoch} policy target statistics:")
                print(f"  Average policy entropy: {target_entropy:.4f}")
                
                # Count how many samples have a highly dominant action (>0.9 probability)
                dominant = torch.max(pi_tensor, dim=1)[0] > 0.9
                dominant_pct = dominant.float().mean().item() * 100
                print(f"  Samples with dominant action (>0.9): {dominant_pct:.1f}%")
                
                # Calculate action distribution across batch
                action_counts = torch.argmax(pi_tensor, dim=1).unique(return_counts=True)[1]
                total = action_counts.sum().item()
                action_freq = action_counts.float() / total
                print(f"  Action distribution: {action_freq.cpu().numpy().round(2)}")
                
                # Calculate how often the most common move is selected
                most_common_pct = action_counts.max().item() / total * 100
                print(f"  Most common move: {most_common_pct:.1f}% of samples")

            optimizer.zero_grad()

            p_pred, v_pred = self.model(state_tensor)
            
            # Diagnostic: Save policy targets and predictions for first batch of first epoch
            if i == 0 and epoch == 0:
                # Print policy distributions for the first few samples
                print("\nDiagnostic - Policy target/prediction distributions:")
                for idx in range(min(5, batch_size_actual)):
                    print(f"Sample {idx}:")
                    print(f"  Target: {pi_tensor[idx].cpu().numpy().round(3)}")
                    print(f"  Predicted: {p_pred[idx].detach().cpu().numpy().round(3)}")
                    print(f"  Entropy (target): {-(pi_tensor[idx] * torch.log(pi_tensor[idx] + 1e-8)).sum().item():.4f}")
                    print(f"  Entropy (pred): {-(p_pred[idx] * torch.log(p_pred[idx] + 1e-8)).sum().item():.4f}")
                
                # Calculate theoretical minimum loss for these targets
                uniform_pred = torch.ones_like(p_pred) / 4.0  # Uniform distribution across 4 actions
                theoretical_min = -(pi_tensor * torch.log(uniform_pred + 1e-8)).sum(dim=1).mean().item()
                print(f"Theoretical minimum cross-entropy with uniform prediction: {theoretical_min:.4f}")
                
                # Check if policy targets are all similar (might indicate MCTS exploration issue)
                target_sim = torch.mean(torch.std(pi_tensor, dim=0)).item()
                print(f"Policy target diversity (stdev across batch): {target_sim:.4f}")
                
                # Print action statistics
                action_probs = torch.mean(pi_tensor, dim=0).cpu().numpy()
                print(f"Average action probabilities in batch: {action_probs.round(3)}")
            
            # Calculate individual sample losses for debugging
            sample_losses = -(pi_tensor * torch.log(p_pred + 1e-8)).sum(dim=1)
            
            # Regular cross-entropy loss
            policy_loss = sample_losses.mean()
            
            # Use Huber/Smooth L1 loss for value prediction with unbounded rewards
            value_loss = torch.nn.functional.smooth_l1_loss(v_pred, z_tensor)
            # Scale value loss down to balance with policy loss
            value_loss_weight = 0.05
            loss = policy_loss + value_loss_weight * value_loss
            
            # Diagnostic: If policy loss is suspiciously constant
            if i == 0 and epoch > 0 and epoch % 5 == 0:
                # Check gradient flow
                params_with_grad = sum(p.requires_grad for p in self.model.parameters())
                policy_params = sum(1 for name, p in self.model.named_parameters() if 'policy' in name and p.requires_grad)
                print(f"\nGradient flow check at epoch {epoch}:")
                print(f"  Parameters requiring gradients: {params_with_grad}")
                print(f"  Policy head parameters: {policy_params}")
                print(f"  Sample policy losses: {sample_losses[:5].detach().cpu().numpy().round(3)}")
                print(f"  Value target range: [{z_tensor.min().item():.2f}, {z_tensor.max().item():.2f}]")
            
            # Backward pass
            loss.backward()
            
            # Check for NaN gradients
            has_nan = False
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan = True
                    print(f"Warning: NaN gradient in {name}")
            
            # Check policy gradient flow every 5 epochs in the first batch
            if i == 0 and epoch % 5 == 0:
                gradients = []
                policy_gradients = []
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        gradients.append(grad_norm)
                        if 'policy' in name:
                            policy_gradients.append((name, grad_norm))
                
                if gradients:
                    avg_grad = sum(gradients) / len(gradients)
                    max_grad = max(gradients)
                    print(f"Gradient stats - Avg: {avg_grad:.6f}, Max: {max_grad:.6f}")
                    
                    if policy_gradients:
                        print("Policy gradients:")
                        for name, norm in policy_gradients:
                            print(f"  {name}: {norm:.6f}")
            
            # Gradient clipping to prevent exploding gradients with unbounded values
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Skip updates if we have NaN gradients
            if not has_nan:
                # Update weights
                optimizer.step()
            else:
                print("Skipping parameter update due to NaN gradients")
            
            # Accumulate losses
            batch_size_actual = end_idx - start_idx
            total_loss += loss.item() * batch_size_actual
            total_policy_loss += policy_loss.item() * batch_size_actual
            # Use the actual weighted value loss for reporting consistency
            total_value_loss += (value_loss_weight * value_loss.item()) * batch_size_actual
            
            # Update progress bar
            current_loss = loss.item()
            current_policy_loss = policy_loss.item()
            current_value_loss = value_loss_weight * value_loss.item()
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
             resume: bool = False, resume_from: str = None, **kwargs):
        """Train the model using self-play for the specified number of epochs
        
        Args:
            epochs: Number of training epochs
            games_per_epoch: Number of self-play games to generate per epoch
            batch_size: Batch size for network updates
            lr: Learning rate for optimizer
            log_interval: How often to log detailed metrics (every N epochs)
            checkpoint_interval: How often to save model checkpoints (every N epochs)
            resume: Whether to resume training from previous state
            resume_from: Path to checkpoint file to resume from (can be local file or URL)
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
        
        # Handle resuming from checkpoint
        start_epoch = 0
        if resume and resume_from:
            print(f"Resuming training from checkpoint: {resume_from}")
            
            # Handle remote URLs (use helper script if needed)
            if resume_from.startswith(("http://", "https://", "r2://")):
                print(f"Remote checkpoint specified: {resume_from}")
                try:
                    # If it's an R2 URL, use the r2_checkpoint.py script to download it
                    if resume_from.startswith("r2://"):
                        import subprocess
                        cmd = [sys.executable, "r2_checkpoint.py", resume_from]
                        print(f"Running: {' '.join(cmd)}")
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode != 0:
                            print(f"Error downloading from R2: {result.stderr}")
                        else:
                            # Extract the local path from the output
                            output_lines = result.stdout.strip().split("\n")
                            for line in output_lines:
                                if line.startswith("Checkpoint downloaded to:"):
                                    resume_from = line.split(":", 1)[1].strip()
                                    break
                    
                    # If it's a direct HTTP URL, download it manually
                    elif resume_from.startswith(("http://", "https://")):
                        import requests
                        
                        # Make sure checkpoints directory exists
                        os.makedirs("checkpoints", exist_ok=True)
                        
                        # Get filename from URL
                        local_path = os.path.join("checkpoints", resume_from.split("/")[-1])
                        
                        # Download the file
                        print(f"Downloading checkpoint from {resume_from} to {local_path}")
                        response = requests.get(resume_from, timeout=60)
                        response.raise_for_status()
                        
                        with open(local_path, "wb") as f:
                            f.write(response.content)
                            
                        print(f"Successfully downloaded checkpoint to {local_path}")
                        resume_from = local_path
                        
                        # Try to download the run file too
                        try:
                            run_url = resume_from.replace(".pth", "_run.json")
                            run_path = os.path.join("checkpoints", run_url.split("/")[-1])
                            
                            run_response = requests.get(run_url, timeout=10)
                            if run_response.status_code == 200:
                                with open(run_path, "wb") as f:
                                    f.write(run_response.content)
                                print(f"Also downloaded run data to {run_path}")
                        except Exception as e:
                            print(f"Note: Could not download run data: {e}")
                            
                except Exception as e:
                    print(f"Error downloading checkpoint: {e}")
                    print("Starting with a fresh model instead")
            
            # Now load the checkpoint if available
            if os.path.exists(resume_from):
                # Try to determine the epoch from filename
                checkpoint_file = os.path.basename(resume_from)
                if "_epoch_" in checkpoint_file:
                    try:
                        start_epoch = int(checkpoint_file.split("_epoch_")[1].split(".")[0]) + 1
                    except:
                        print("Could not determine epoch from filename, starting from 0")
                
                # Load model weights
                print(f"Loading model weights from {resume_from}")
                state_dict = torch.load(resume_from, map_location='cpu')
                self.model.load_state_dict(state_dict)
                print(f"Successfully loaded model from checkpoint. Resuming from epoch {start_epoch}")
                
                # Try to load run data too
                run_file = resume_from.replace(".pth", "_run.json")
                if os.path.exists(run_file):
                    try:
                        with open(run_file, 'r') as f:
                            run_data = json.load(f)
                            self.monitor.run_data = run_data
                        print(f"Successfully loaded run data from {run_file}")
                    except Exception as e:
                        print(f"Error loading run data: {e}")
            else:
                print(f"Warning: Checkpoint file {resume_from} not found. Starting fresh.")
                # Create checkpoints directory if it doesn't exist
                os.makedirs("checkpoints", exist_ok=True)

                # Download the checkpoint
                response = requests.get(resume_from, timeout=60)
                response.raise_for_status()

                # Extract filename from URL or use a default
                if "/" in resume_from:
                    local_path = os.path.join("checkpoints", resume_from.split("/")[-1])
                else:
                    local_path = os.path.join("checkpoints", "downloaded_checkpoint.pth")

                # Save the checkpoint
                with open(local_path, 'wb') as f:
                    f.write(response.content)

                print(f"Successfully downloaded checkpoint to {local_path}")
                resume_from = local_path

                # Also try to download the run file
                run_url = resume_from.replace(".pth", "_run.json").replace("_epoch_", "_run")
                try:
                    run_response = requests.get(run_url, timeout=30)
                    run_response.raise_for_status()

                    local_run_path = local_path.replace(".pth", "_run.json").replace("_epoch_", "_run")
                    with open(local_run_path, 'wb') as f:
                        f.write(run_response.content)

                    print(f"Successfully downloaded run data to {local_run_path}")
                except Exception as e:
                    print(f"Note: Could not download run data: {e}")

            # Now try to load the local checkpoint
            if os.path.exists(resume_from):
                # Extract epoch number from filename
                checkpoint_file = os.path.basename(resume_from)
                if "_epoch_" in checkpoint_file:
                    try:
                        start_epoch = int(checkpoint_file.split("_epoch_")[1].split(".")[0]) + 1
                    except:
                        print("Could not determine epoch from filename, starting from 0")
                
                # Load model weights
                state_dict = torch.load(resume_from, map_location='cpu')
                self.model.load_state_dict(state_dict)
                print(f"Successfully loaded model from checkpoint. Resuming from epoch {start_epoch}")
                
                # Also try to load run data
                run_file = resume_from.replace(".pth", "_run.json").replace("_epoch_", "_run")
                if os.path.exists(run_file):
                    import json
                    try:
                        with open(run_file, 'r') as f:
                            self.monitor.run_data = json.load(f)
                        print(f"Successfully loaded run data from {run_file}")
                    except Exception as e:
                        print(f"Error loading run data: {e}")
            else:
                print(f"Warning: Checkpoint file {resume_from} not found. Starting fresh.")
        
        # Initialize optimizer with specified learning rate
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        
        # Optional: Add learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # If resuming, adjust scheduler
        if resume and start_epoch > 0:
            for _ in range(start_epoch):
                scheduler.step()
            print(f"Adjusted learning rate for resumed training: {scheduler.get_last_lr()[0]}")
        
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
                "lr": lr,
                "resume": resume,
                "resume_from": resume_from,
                "start_epoch": start_epoch
            }
        })
        
        # Training loop
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            
            # Set model to evaluation mode during self-play
            self.model.eval()
            
            # 1. Collect training data with self-play
            print(f"Epoch {epoch+1}/{epochs}: Collecting self-play data...")
            
            # Configure exploration parameters for self-play
            # Use higher exploration in early epochs, then decrease gradually
            exploration_factor = max(0.5, 1.0 - epoch / (epochs * 0.5))  # Linearly decrease to 0.5 by mid-training
            dirichlet_eps = kwargs.get('dirichlet_eps', 0.25) * exploration_factor
            dirichlet_alpha = kwargs.get('dirichlet_alpha', 0.5)
            temperature = max(0.8, kwargs.get('temperature', 1.0) * exploration_factor)
            
            # Print exploration settings for this epoch
            print(f"Self-play exploration: dir_eps={dirichlet_eps:.2f}, dir_alpha={dirichlet_alpha:.2f}, temp={temperature:.2f}")
            
            # Update player parameters
            self.player.dir_eps = dirichlet_eps
            self.player.dir_alpha = dirichlet_alpha
            self.player.temperature = temperature
            
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
                "epoch_time_seconds": epoch_time,
                "reward_type": epoch_stats.get('reward_type', 'default_score')
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
        action, (probs, value) = agent.play(state, simulations=100)
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
