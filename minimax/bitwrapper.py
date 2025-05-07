"""
bitwrapper.py
═════════════
Bridges the NumPy‑based Simplified2048 engine with the original
64‑bit‑nibble “bit‑board” representation used by the expectimax AI.
"""

from simplified_2048 import Simplified2048
import numpy as np
from typing import Tuple

class BitBoardEnv:
    DIRS = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT'}

    def __init__(self):
        self.game = Simplified2048()          # default: 4×4, 1 spawn / move
        self.b    = self._np_to_bits(self.game.get_board())

    # ───────────────────────────────── helpers ─────────────────────────────────
    @staticmethod
    def _np_to_bits(arr: np.ndarray) -> int:
        """4×4 NumPy grid → 64‑bit board (4‑bit exponents)."""
        v = 0
        k = 0
        for r in range(4):
            for c in range(4):
                cell = int(arr[r, c])
                exp  = int(np.log2(cell)) if cell else 0
                v   |= (exp & 0xF) << (4 * k)
                k   += 1
        return v

    def _sync_bits_to_np(self) -> None:
        """Copies current bit‑board back into the NumPy grid."""
        grid = np.zeros((4, 4), dtype=np.int64)
        for k in range(16):
            e = (self.b >> (4 * k)) & 0xF
            grid[k // 4, k % 4] = 0 if e == 0 else (1 << e)
        self.game._board[:] = grid          # internal update (speed)

    # ────────────────────────────── public facade ──────────────────────────────
    def move(self, direction: int) -> Tuple[int, int]:
        """
        Executes a move in the wrapped game and refreshes the bit‑board.
        Returns (new_bit_board, score_gain_for_this_move).
        """
        gain, _ = self.game.move(direction)
        self.b  = self._np_to_bits(self.game.get_board())
        return self.b, int(gain)

    def add_random_tile(self) -> None:
        """Adds a spawn tile via engine rules and refreshes bit‑board."""
        self.game._add_random_tile()
        self.b = self._np_to_bits(self.game.get_board())

    def count_empty(self) -> int:
        return int(np.sum(self.game._board == 0))

    # cloning helper (handy if you later want multiple envs)
    def clone(self) -> "BitBoardEnv":
        twin       = BitBoardEnv.__new__(BitBoardEnv)
        twin.game  = self.game.clone()
        twin.b     = self._np_to_bits(twin.game.get_board())
        return twin
