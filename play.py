#!/usr/bin/env python3
"""
2048 Zero interactive play script
"""
import os
import torch
import time
import argparse
from zeromodel import ZeroNetwork
from zero2048 import ZeroPlayer
from game_alt import GameRules, BitBoard, GameState, GameRunner

def main():
    parser = argparse.ArgumentParser(description="2048 Zero - Play with a trained model")
    parser.add_argument('--checkpoint', type=str, help='Load model from checkpoint')
    parser.add_argument('--simulations', type=int, default=100, help='MCTS simulations per move')
    parser.add_argument('--time-limit', type=float, default=0.5, help='Time limit for each move (seconds)')
    
    args = parser.parse_args()
    
    # Initialize game rules and model
    rules = GameRules(num_spawn_tiles_per_move=1)
    model = ZeroNetwork(4, 4, 16)
    
    # Device selection
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device for inference")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device for inference")
    else:
        print(f"Using CPU device for inference")
    
    model.to(device)
    
    # Load checkpoint if specified
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load(args.checkpoint)
        print(f"Loaded model from {args.checkpoint}")
    else:
        print("Using untrained model (random play)")
    
    # Create agent and game runner
    agent = ZeroPlayer(model, rules)
    runner = GameRunner(rules)
    turn = 0
    
    print("\n====== 2048 Zero AI Game ======")
    print(f"Starting a new game with {args.simulations} MCTS simulations per move")
    print(runner.render_ascii())
    
    # Play the game until terminal state
    while not runner.is_game_over():
        turn += 1
        
        # Get current board and score
        current_board = runner.get_board()
        current_score = runner.get_score()
        max_tile = rules.get_max_tile(current_board)
        
        # Create GameState with bitboard for MCTS
        bitboard = BitBoard.from_numpy(current_board)
        state = GameState(current_board, current_score, bitboard)
        
        # Get action from agent
        print(f"\nTurn {turn} - Thinking... (Score: {current_score}, Max Tile: {max_tile})")
        action, (probs, value) = agent.play(state, time_limit=args.time_limit, simulations=args.simulations)
        action_name = rules.get_direction_name(action)
        
        # Show action probabilities
        prob_str = " ".join([f"{rules.get_direction_name(i)}: {p:.2f}" for i, p in enumerate(probs)])
        
        # Execute the action
        gain, changed = runner.move(action)
        runner.generate_tiles()
        
        # Display the result
        print(f"Move: {action_name} (value est: {value:.2f}, probs: {prob_str})")
        print(f"Gained: {gain}" + (" [GAME OVER]" if runner.is_game_over() else ""))
        print(runner.render_ascii())
    
    print(f"\n====== Game Over! ======")
    print(f"Final score: {runner.get_score()}")
    print(f"Max tile: {runner.get_max_tile()}")
    print(f"Turns played: {turn}")

if __name__ == "__main__":
    main()