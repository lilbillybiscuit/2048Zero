#!/usr/bin/env python3

import argparse, random, statistics, importlib
from typing import Callable, Tuple
import simplified_2048 as env

ai = importlib.import_module("ai2048_bitboard")   # minimax AI module
GameFun = Callable[[int, bool, int], int]  # (depth, verbose, seed) -> score


def play_one_random(seed: int = None) -> int:
    if seed is not None:
        random.seed(seed)
    g = env.Simplified2048()
    while not g.is_game_over():
        move = random.choice(g.get_valid_moves())
        g.move(move)
    return int(g.get_score())


def play_one_merges(seed: int = None) -> int:
    if seed is not None:
        random.seed(seed)
    g = env.Simplified2048()
    while not g.is_game_over():
        best = None
        best_merges = -1
        best_gain = -1
        for direction in env.Simplified2048.DIRECTIONS:
            g_tmp = g.clone()
            gain, _ = g_tmp.move(direction)
            if gain == 0 and g_tmp.is_game_over():
                continue  # illegal or no‑op
            merges = int(g_tmp.get_score() - g.get_score() - gain)  # not used, always 0
            # count merges directly
            merges = gain // 2 if gain else 0
            if (merges, gain) > (best_merges, best_gain):
                best_merges, best_gain, best = merges, gain, direction
        g.move(best)
    return int(g.get_score())


def play_one_minimax(depth: int, seed: int) -> int:
    return ai.play(depth=depth, verbose=False, seed=seed)


def run_batch(
    label: str,
    fn: GameFun,
    games: int,
    depth: int | None = None,
) -> Tuple[list[int], int]:
    scores = []
    best_seed, best_score = None, -1
    for i in range(games):
        seed = random.randrange(2**32)
        if label == "minimax":
            score = fn(depth, False, seed)  # depth param for minimax
        else:
            score = fn(seed)
        scores.append(score)
        if score > best_score:
            best_seed, best_score = seed, score
    print(
        f"{label:<7}  mean={statistics.mean(scores):7.1f}  "
        f"min={min(scores):5d}  max={max(scores):5d}  best_seed={best_seed}"
    )
    return scores, best_seed


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=100, help="games per strategy")
    p.add_argument("--depth", type=int, default=3, help="expectimax depth")
    args = p.parse_args()

    print(f"\n== Benchmark: {args.games} games per strategy ==\n")

    run_batch("naive",   play_one_random,  args.games)
    run_batch("merges",  play_one_merges,  args.games)
    run_batch("minimax", play_one_minimax, args.games, depth=args.depth)
