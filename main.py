import argparse, random
from minimax import ai2048_bitboard as ai

from zero import GameRunner, GameRules
from typing import List, Tuple
from collections import defaultdict

def run_random_game(rules: GameRules, rounds=100) -> Tuple[int, List[int]]:
    results = defaultdict(int)
    score = 0.0
    for i in range(rounds):
        runner = GameRunner(rules)
        while not runner.is_game_over():
            direction = random.choice(runner.get_valid_moves())
            runner.move(direction)
            runner.generate_tiles()

        results[int(runner.get_max_tile())] += 1
        score += runner.get_score()
        print(f"Game {i+1}: Score = {runner.get_score()}, Max Tile = {runner.get_max_tile()}")
    print(f"Average Score: {score / rounds}")
    ret = []
    for i in range(1, 20):
        if (1<<i) in results:
            ret.append(((1<<i), results[(1<<i)]))
        else:
            ret.append(((1<<i), 0))
    return score, ret


def run_max_merge_heuristic_game(rules: GameRules) -> Tuple[int, int]:
    runner = GameRunner(rules)
    turn = 0
    while not runner.is_game_over():
        valid_moves = runner.get_valid_moves()
        if not valid_moves:
            break

        best_move = None
        most_merges = -1

        move_stats = []
        for direction in valid_moves:
            board_copy = runner.get_board()

            _, score_gain, new_board = rules.simulate_move(board_copy, direction)

            move_stats.append((direction, score_gain))

            if score_gain > most_merges:
                most_merges = score_gain
                best_move = direction

        chosen_move = best_move if best_move is not None else valid_moves[0]

        runner.move(chosen_move)
        runner.generate_tiles()

        turn += 1


    return int(runner.get_score()), int(runner.get_max_tile())

def run_max_heurisic_games(rules: GameRules, rounds = 100) -> Tuple[int, List[int]]:
    results = defaultdict(int)
    score = 0.0
    for i in range(rounds):
        score1, max_tile = run_max_merge_heuristic_game(rules)
        results[max_tile] += 1
        print(f"Game {i + 1}: Score = {score1}, Max Tile = {max_tile}")
        score += score1
    print(f"Average Score: {score / rounds}")

    ret = []
    for i in range(1, 20):
        if (1<<i) in results:
            ret.append(((1<<i), results[(1<<i)]))
        else:
            ret.append(((1<<i), 0))
    return score, ret

from collections import defaultdict
from tqdm import tqdm
from minimax.ai2048_bitboard import expectimax
from minimax.bitwrapper import BitBoardEnv

def run_bitboard_games(rounds=10, depth=3):
    results = defaultdict(int)
    total_score = 0.0
    for _ in tqdm(range(rounds)):
        env = BitBoardEnv()
        board = env.b
        score = 0
        while True:
            direction, _ = expectimax(board, depth)
            if direction == -1:
                break
            new_board, gain = env.move(direction)
            if new_board == board:
                break
            board = new_board
            score += gain
            if env.count_empty() == 0:
                break
            env.add_random_tile()
            board = env.b
        max_tile = max_val_in_bitboard(board)
        results[max_tile] += 1
        total_score += score
    avg_score = total_score / rounds
    ret = []
    for i in range(1, 20):
        tile_value = 1 << i
        ret.append((tile_value, results.get(tile_value, 0)))
    return avg_score, ret

def max_val_in_bitboard(board):
    max_exp = 0
    for i in range(16):
        exp = (board >> (4 * i)) & 0xF
        max_exp = max(max_exp, exp)
    return 0 if max_exp == 0 else (1 << max_exp)

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

import torch
from tqdm import tqdm
from collections import defaultdict
from zero.game import GameRules, GameState, BitBoard, GameRunner
from zero.zeromodel import ZeroNetwork
from zero.zero2048 import ZeroPlayer

def run_zero_games(weights_path="weights/r76.pt", num_games=100, num_simulations=50):
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

    for _ in tqdm(range(num_games)):
        runner = GameRunner(rules)
        while not runner.is_game_over():
            board = runner.get_board()
            score = runner.get_score()
            bitboard = BitBoard.from_numpy(board)
            state = GameState(board, score, bitboard)
            action, _ = player.play(state, simulations=num_simulations)
            runner.move(action)
            runner.generate_tiles()
        final_score = runner.get_score()
        max_tile = runner.get_max_tile()
        results[int(max_tile)] += 1
        total_score += final_score

    avg_score = total_score / num_games
    freq_list = []
    for i in range(1, 20):
        tile_value = 1 << i
        freq_list.append((tile_value, results.get(tile_value, 0)))
    return avg_score, freq_list

if __name__ == "__main__":
    rules = GameRules(
        height=4, width=4, spawn_rates={1: 0.9, 2: 0.1}, num_spawn_tiles_per_move=2, num_initial_tiles=2
    )
    avg_score1, results1 = run_random_game(rules)
    avg_score2, results2 = run_max_heurisic_games(rules)
    avg_score3, results3 = run_bitboard_games(rounds=100, depth=3)
    avg_score4, results4 = run_zero_games(weights_path="weights/r0.pt", num_games=100, num_simulations=50)

    # print results
    print("\nRandom Game Results:")
    print(f"Average Score: {avg_score1}")
    print("Tile Achievement Rates:", results1)
    print("\nMax Merge Heuristic Game Results:")
    print(f"Average Score: {avg_score2}")
    print("Tile Achievement Rates:", results2)
    print("\nBitboard Game Results:")
    print(f"Average Score: {avg_score3}")
    print("Tile Achievement Rates:", results3)
    print("\nZero Game Results:")
    print(f"Average Score: {avg_score4}")
    print("Tile Achievement Rates:", results4)








