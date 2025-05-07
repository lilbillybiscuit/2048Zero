#!/usr/bin/env python3
"""
2048 single run          →   python main.py 2048 [--depth 3]
2048 best‑of‑N benchmark →   python main.py 2048-best  --games 100 --depth 3
"""

import argparse, random
from minimax import ai2048_bitboard as ai
# --------------------------------------------------------------------------- #
# 2048 helpers
# --------------------------------------------------------------------------- #
def play_once(depth: int, verbose: bool):
    """Play one game and print / return the score."""
    score = ai.play(depth=depth, verbose=verbose, seed=random.randrange(2**32))
    if not verbose:
        print(f"score = {score}")
    return score


def best_of_n(games: int, depth: int):
    """Run `games` silent games, track the best, then replay it verbosely."""
    best_seed  = None
    best_score = -1
    for i in range(1, games + 1):
        seed = random.randrange(2**32)
        score = ai.play(depth=depth, verbose=False, seed=seed)
        if score > best_score:
            best_score, best_seed = score, seed
        print(f"[{i:3d}/{games}] score = {score}")
    print(f"\nBEST score={best_score}  seed={best_seed}\n")
    # ai.play(depth=depth, verbose=True, seed=best_seed)


def other_task():
    print("Other task placeholder.  Add your code here.")


def build_parser():
    p = argparse.ArgumentParser(description="Utility launcher")
    sub = p.add_subparsers(dest="cmd", required=True)


    p1 = sub.add_parser("2048", help="Play one AI game")
    p1.add_argument("--depth", type=int, default=3, help="expectimax depth")
    p1.add_argument("--quiet", action="store_true", help="suppress move log")

    p2 = sub.add_parser("2048-best", help="Run many games and replay the best")
    p2.add_argument("--games", type=int, default=100, help="how many runs")
    p2.add_argument("--depth", type=int, default=3, help="expectimax depth")


    # sub.add_parser("naive", help="Run some other experiment")
    return p


def main():
    args = build_parser().parse_args()

    if args.cmd == "2048":
        play_once(depth=args.depth, verbose=not args.quiet)

    elif args.cmd == "2048-best":
        best_of_n(games=args.games, depth=args.depth)

    elif args.cmd == "other-task":
        other_task()


if __name__ == "__main__":
    main()
