#!/usr/bin/env python3
"""
Play multiple 2048 games using the AlphaZero implementation and generate statistics.
This script plays a specified number of games using a trained model and produces
statistics about max tiles achieved, win rates, average scores, and more.

Example usage:
    python play_stats.py --weights weights/r0.pt --games 100 --simulations 50 --save-stats
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from datetime import datetime
from collections import Counter
from tabulate import tabulate
from tqdm import tqdm

from zero.game import GameRules, GameState, BitBoard
from zero.zeromodel import ZeroNetwork
from zero.zero2048 import ZeroPlayer

def download_weights(url):
    """Download weights from URL if needed"""
    import requests
    import os
    
    # Extract filename from URL
    filename = os.path.basename(url)
    local_path = os.path.join("weights", filename)
    
    # Create weights directory if needed
    os.makedirs("weights", exist_ok=True)
    
    # Check if weights already exist
    if os.path.exists(local_path):
        print(f"Using existing weights: {local_path}")
        return local_path
    
    # Download weights
    print(f"Downloading weights from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get file size for progress tracking
    total_size = int(response.headers.get('content-length', 0))
    
    # Save weights file
    with open(local_path, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Print progress
                    progress = downloaded / total_size * 100
                    sys.stdout.write(f"\rProgress: {progress:.1f}%")
                    sys.stdout.flush()
            print()  # Newline after progress
    
    print(f"Downloaded weights to {local_path}")
    return local_path

def load_model(weights_path, device=None):
    """Load model from weights file"""
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Create model
    rules = GameRules()
    model = ZeroNetwork(
        rules.height,  # 4
        rules.width,   # 4
        16,            # k channels
        filters=128,   # filters
        blocks=10      # blocks
    )
    
    # Load weights
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, rules

def play_game(model, rules, num_simulations=100, render=False):
    """Play a single game and return statistics"""
    # Create player with MCTS
    player = ZeroPlayer(model, rules)
    
    # Initial state
    state = rules.get_initial_state()
    
    # Game statistics
    turn_count = 0
    max_tile_achieved = 0
    
    # Play until terminal state
    while not rules.is_terminal(state.board):
        if render:
            print(f"\nTurn {turn_count+1}")
            print(rules.render_ascii(state.board))
        
        # Get action from agent with MCTS
        action, (probs, value) = player.play(state, simulations=num_simulations)
        
        # Apply move
        new_board, gain = rules.apply_move(state.board, action)
        new_board = rules.add_random_tiles(new_board)
        new_bitboard = BitBoard.from_numpy(new_board)
        state = GameState(new_board, state.score + gain, new_bitboard)
        
        # Update statistics
        turn_count += 1
        current_max_tile = rules.get_max_tile(state.board)
        max_tile_achieved = max(max_tile_achieved, current_max_tile)
        
        if render:
            print(f"Action: {rules.get_direction_name(action)}")
            print(f"Score: {state.score}, Max Tile: {max_tile_achieved}")
    
    # Game finished
    if render:
        print("\nGame over!")
        print(rules.render_ascii(state.board))
        print(f"Final score: {state.score}")
        print(f"Max tile: {max_tile_achieved}")
        print(f"Turns: {turn_count}")
    
    return {
        'max_tile': max_tile_achieved,
        'score': state.score,
        'turns': turn_count
    }

def play_games(model, rules, num_games=100, num_simulations=100):
    """Play multiple games and collect statistics"""
    results = []
    
    print(f"Playing {num_games} games with {num_simulations} simulations per move...")
    for i in tqdm(range(num_games)):
        result = play_game(model, rules, num_simulations=num_simulations)
        results.append(result)
    
    return results

def analyze_results(results, win_threshold=2048):
    """Analyze game results and create statistics"""
    # Count games reaching each tile threshold
    max_tiles = [result['max_tile'] for result in results]
    
    # Create statistics for power-of-2 tiles (4, 8, 16, etc.)
    tile_stats = {}
    
    # Start from 2^2 (4) up to 2^11 (2048) or higher if needed
    max_power = max(11, int(np.ceil(np.log2(max(max_tiles + [2048])))))  # Ensure we go at least to 2048
    for power in range(2, max_power + 1):
        tile_value = 2 ** power
        count = sum(1 for tile in max_tiles if tile >= tile_value)
        percentage = count / len(results) * 100
        tile_stats[tile_value] = (count, percentage)
    
    # Calculate average score and turns
    avg_score = sum(result['score'] for result in results) / len(results)
    avg_turns = sum(result['turns'] for result in results) / len(results)
    
    # Calculate win rate (reaching the win_threshold tile)
    wins = sum(1 for tile in max_tiles if tile >= win_threshold)
    win_rate = wins / len(results) * 100
    
    return {
        'tile_stats': tile_stats,
        'avg_score': avg_score,
        'avg_turns': avg_turns,
        'num_games': len(results),
        'win_rate': win_rate,
        'win_threshold': win_threshold
    }

def print_statistics(stats):
    """Print statistics in a nice format"""
    print(f"\nStatistics for {stats['num_games']} games:")
    print(f"Average score: {stats['avg_score']:.1f}")
    print(f"Average turns: {stats['avg_turns']:.1f}")
    
    # Create table for tile statistics
    table_data = []
    for tile_value, (count, percentage) in sorted(stats['tile_stats'].items()):
        table_data.append([
            f"{tile_value}",
            f"{count}/{stats['num_games']}",
            f"{percentage:.1f}%"
        ])
    
    print("\nMax Tile Achievement Rates:")
    try:
        print(tabulate(table_data, headers=["Tile", "Count", "Percentage"], tablefmt="grid"))
    except Exception:
        # Fallback to ASCII table if tabulate fails for any reason
        print_ascii_table(stats)

def print_ascii_table(stats):
    """Print a simple ASCII table for tile statistics"""
    print("\nTile Achievement Rates:")
    print("┌───────┬────────┬─────────┐")
    print("│ Tile  │ Count  │ Percent │")
    print("├───────┼────────┼─────────┤")
    
    for tile_value, (count, percentage) in sorted(stats['tile_stats'].items()):
        print(f"│ {tile_value:5} │ {count:4}/{stats['num_games']:<3} │ {percentage:6.1f}% │")
    
    print("└───────┴────────┴─────────┘")
    
    # Print win rate
    if 'win_rate' in stats:
        print(f"\nWin Rate (≥{stats['win_threshold']} tile): {stats['win_rate']:.1f}%")

def save_statistics(stats, args):
    """Save statistics to a JSON file"""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(args.weights) if args.weights else "url_model"
    filename = f"stats_{model_name}_{args.games}games_{timestamp}.json"
    filepath = os.path.join(args.output_dir, filename)
    
    # Convert stats to a JSON-serializable format
    json_stats = {
        'num_games': stats['num_games'],
        'avg_score': stats['avg_score'],
        'avg_turns': stats['avg_turns'],
        'win_rate': stats['win_rate'],
        'win_threshold': stats['win_threshold'],
        'tile_stats': {}
    }
    
    # Convert tile stats
    for tile_value, (count, percentage) in stats['tile_stats'].items():
        json_stats['tile_stats'][str(tile_value)] = {
            'count': count,
            'percentage': percentage
        }
    
    # Add run configuration
    json_stats['config'] = {
        'simulations': args.simulations,
        'device': args.device if args.device else 'auto',
        'weights': args.weights if args.weights else 'url_model',
        'date': timestamp
    }
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(json_stats, f, indent=2)
    
    print(f"\nStatistics saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Play multiple 2048 games and analyze max tile statistics')
    parser.add_argument('--url', type=str, default=None, help='URL to weights file')
    parser.add_argument('--weights', type=str, default=None, help='Path to weights file')
    parser.add_argument('--games', type=int, default=100, help='Number of games to play')
    parser.add_argument('--win-threshold', type=int, default=2048, help='Tile value considered a win')
    parser.add_argument('--save-stats', action='store_true', help='Save statistics to a JSON file')
    parser.add_argument('--output-dir', type=str, default='stats', help='Directory to save statistics')
    parser.add_argument('--simulations', type=int, default=50, help='MCTS simulations per move')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda, mps, auto)')
    parser.add_argument('--render', action='store_true', help='Render the first game')
    args = parser.parse_args()
    
    # Handle weights loading
    if args.weights is None and args.url is None:
        # Use local weights by default if available
        if os.path.exists("weights/r0.pt"):
            args.weights = "weights/r0.pt"
        else:
            # Replace with your actual default URL if needed
            args.url = "https://example.com/weights/latest.pt"
    
    weights_path = None
    if args.weights:
        weights_path = args.weights
    elif args.url:
        weights_path = download_weights(args.url)
    
    # Load model
    model, rules = load_model(weights_path, device=args.device)
    
    # Play one game with rendering if requested
    if args.render:
        print("Playing a sample game with rendering...")
        play_game(model, rules, num_simulations=args.simulations, render=True)
    
    # Play games and collect statistics
    results = play_games(model, rules, num_games=args.games, num_simulations=args.simulations)
    
    # Analyze results
    stats = analyze_results(results, win_threshold=args.win_threshold)
    
    # Print statistics
    try:
        print_statistics(stats)
    except ImportError:
        # If tabulate is not available, fall back to ASCII table
        print_ascii_table(stats)
        
    # Save statistics if requested
    if args.save_stats:
        save_statistics(stats, args)

if __name__ == "__main__":
    main()