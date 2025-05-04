#!/usr/bin/env python3
"""
play.py  –  glue layer between Game2048Animation (game.py)
           and the bit‑board AI (2048_bitboard.py).

• Runs a head‑less CLI game OR (if Pygame installed) a simple window.
• Uses expectimax(depth=3) from 2048_bitboard.py to pick each move.
--------------------------------------------------------------------
Directory layout this assumes:

project/
│
├── game.py                  # defines Game2048Animation (your rules / GUI hooks)
├── 2048_bitboard.py         # AI + bit‑board utilities
└── play.py                  # <-- this file
"""

import math
import sys
import time

try:
    import numpy as np
except ImportError:
    np = None

# ------- import your own modules ---------------------------------
from visual.game import Game2048Animation          # <-- change to the right name
from test_minmax import expectimax        # (depth defaults to 3)


# -----------------------------------------------------------------
# helpers to convert between NumPy/list boards and 64‑bit bit‑board
# -----------------------------------------------------------------
def board_to_bitboard(board):
    """
    Take a 4×4 *value* board (0,2,4,8,…)  → 64‑bit int (4‑bit exponents).
    Accepts: np.ndarray, list[list[int]], list[list[np.int_]].
    """
    if np is not None and isinstance(board, np.ndarray):
        flat = board.ravel()
    else:
        flat = [cell for row in board for cell in row]

    bb = 0
    for k, val in enumerate(flat):
        exp = 0 if val == 0 else int(math.log2(val))
        bb |= exp << (4 * k)
    return bb


# map AI dir‑index 0‑3 to Game2048Animation.move() direction strings
IDX2DIR = ['up', 'right', 'down', 'left']


# -----------------------------------------------------------------
# choose one of two front‑ends: CLI or Pygame (if available)
# -----------------------------------------------------------------
def cli_loop():
    """Plain‑text play in the terminal."""
    game = Game2048Animation()
    step = 0

    while not game.game_over:
        bb = board_to_bitboard(game.board)
        dir_idx, _ = expectimax(bb, depth=3)
        if dir_idx == -1:          # AI sees no legal moves
            break

        moved, gained, _ = game.move(IDX2DIR[dir_idx])
        if not moved:
            break

        # print board
        print(f"\nStep {step} – move {IDX2DIR[dir_idx]}, +{gained}, score={game.score}")
        for row in game.board:
            print(" ".join(f"{v:5d}" for v in row))
        step += 1
        time.sleep(0.05)

    print(f"\nGame over. Final score: {game.score}")


def pygame_loop():
    """Uses a bare‑bones Pygame viewer if pygame is available."""
    try:
        import pygame
    except ModuleNotFoundError:
        print("Pygame not installed – falling back to CLI.")
        cli_loop()
        return

    # --- minimal on‑screen renderer --------------------------------
    TILE_COLORS = {
        0:  (204, 192, 179), 2:  (238, 228, 218),   4:  (237, 224, 200),
        8:  (242, 177, 121), 16: (245, 149, 99),   32: (246, 124, 95),
        64: (246, 94, 59),   128:(237, 207, 114), 256:(237, 204, 97),
        512:(237, 200, 80), 1024:(237, 197, 63), 2048:(237, 194, 46),
    }
    S, M = 110, 15              # tile size, margin
    WIDTH = 4*S + 5*M
    HEIGHT = WIDTH + 60

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2048 – bit‑board AI")
    font_num = pygame.font.SysFont("Helvetica", 36, bold=True)
    font_hud = pygame.font.SysFont("Helvetica", 24)

    game = Game2048Animation()
    clock = pygame.time.Clock()
    last_ai = 0

    def draw():
        screen.fill((187, 173, 160))
        # grid bg
        pygame.draw.rect(screen, (170,157,143), (0,0,WIDTH,WIDTH), border_radius=8)
        for r in range(4):
            for c in range(4):
                val = game.board[r, c]
                color = TILE_COLORS.get(val, (60,58,50))
                rect = pygame.Rect(M + c*(S+M), M + r*(S+M), S, S)
                pygame.draw.rect(screen, color, rect, border_radius=5)
                if val:
                    txt = font_num.render(str(val), True,
                         (119,110,101) if val<8 else (249,246,242))
                    screen.blit(txt, txt.get_rect(center=rect.center))
        # score bar
        pygame.draw.rect(screen, (170,157,143), (0, WIDTH+5, WIDTH, 55), border_radius=5)
        s_txt = font_hud.render(f"Score: {game.score}", True, (255,255,255))
        screen.blit(s_txt, (M, WIDTH+20))
        pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

        now = time.time()
        if not game.game_over and now - last_ai > 0.12:
            bb = board_to_bitboard(game.board)
            dir_idx, _ = expectimax(bb, depth=3)
            if dir_idx == -1:
                game.game_over = True
            else:
                game.move(IDX2DIR[dir_idx])
            last_ai = now

        draw()
        clock.tick(60)


# -----------------------------------------------------------------
if __name__ == "__main__":
    # choose GUI if pygame installed; otherwise CLI.
    try:
        import pygame          # noqa: F401
        pygame_loop()
    except ModuleNotFoundError:
        cli_loop()
