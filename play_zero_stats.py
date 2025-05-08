#!/usr/bin/env python3
import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from tabulate import tabulate
from zero.game import GameRules, GameState, BitBoard, GameRunner
from zero.zeromodel import ZeroNetwork
from zero.zero2048 import ZeroPlayer

def run_zero_games(weights_path="weights/r76.pt", num_games=100, num_simulations=50, render_first=False):
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    rules = GameRules()
    model = ZeroNetwork(
        rules.height,
        rules.width,
        16,
        filters=128,
        blocks=10
    )
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    player = ZeroPlayer(model, rules)
    results = defaultdict(int)
    total_score = 0
    total_turns = 0

    print(f"Playing {num_games} games with {num_simulations} simulations per move...")
    for game_idx in tqdm(range(num_games)):
        runner = GameRunner(rules)
        turn_count = 0

        while not runner.is_game_over():
            board = runner.get_board()
            score = runner.get_score()
            max_tile = runner.get_max_tile()
            bitboard = BitBoard.from_numpy(board)
            state = GameState(board, score, bitboard)
            action, (probs, value) = player.play(state, simulations=num_simulations)
            gain, _ = runner.move(action)
            runner.generate_tiles()
            turn_count += 1

        final_score = runner.get_score()
        max_tile = runner.get_max_tile()
        results[int(max_tile)] += 1
        total_score += final_score
        total_turns += turn_count

        print(f"Game {game_idx+1}/{num_games}: Score = {final_score}, Max Tile = {max_tile}, Turns = {turn_count}")

    avg_score = total_score / num_games
    avg_turns = total_turns / num_games

    print(f"\nResults for {num_games} games played with {num_simulations} MCTS simulations per move:")
    print(f"Average score: {avg_score:.1f}")
    print(f"Average turns: {avg_turns:.1f}")

    tile_stats = []
    for i in range(1, 20):
        tile_value = 1 << i
        if tile_value in results or i <= 12:
            count = results[tile_value]
            percentage = count / num_games * 100
            at_least_count = sum(results[1 << j] for j in range(i, 20) if (1 << j) in results)
            at_least_percentage = at_least_count / num_games * 100
            tile_stats.append([
                f"{tile_value}",
                f"{count}/{num_games}",
                f"{percentage:.1f}%",
                f"{at_least_count}/{num_games}",
                f"{at_least_percentage:.1f}%"
            ])

    ret = []
    for i in range(1, 20):
        tile_value = 1 << i
        if tile_value in results:
            ret.append((tile_value, results[tile_value]))
        else:
            ret.append((tile_value, 0))
    return total_score, ret

if __name__ == "__main__":
    run_zero_games(weights_path="weights/r0.pt", num_games=100, num_simulations=50, render_first=False)