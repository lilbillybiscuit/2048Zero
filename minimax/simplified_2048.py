#!/usr/bin/env python3
"""
simplified_2048.py

A fully self‑contained, programmatic‑only version of the 2048 game
designed for AI experiments (e.g., MCTS or expectimax).
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any

# Type alias for clarity
BoardType = np.ndarray[Any, np.dtype[np.int64]]


class Simplified2048:
    """
    A simplified 2048 implementation with configurable board size,
    spawn distribution, and number of tiles spawned after each move.
    Directions are encoded as integers:
        0 = Up, 1 = Right, 2 = Down, 3 = Left
    """

    # Direction constants
    UP: int = 0
    RIGHT: int = 1
    DOWN: int = 2
    LEFT: int = 3
    DIRECTIONS: List[int] = [UP, RIGHT, DOWN, LEFT]

    # Default probability of spawning a 2‑tile or 4‑tile
    DEFAULT_SPAWN_RATES: Dict[int, float] = {2: 0.9, 4: 0.1}

    # --------------------------------------------------------------------- #
    #                               INIT                                    #
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        height: int = 4,
        width: int = 4,
        spawn_rates: Dict[int, float] = None,
        num_spawn_tiles_per_move: int = 1,
        num_initial_tiles: int = 2,
    ):
        if spawn_rates is None:
            spawn_rates = self.DEFAULT_SPAWN_RATES

        # Sanity checks
        assert height > 0 and width > 0, "Board dimensions must be positive."
        assert num_spawn_tiles_per_move > 0, "num_spawn_tiles_per_move must be positive."
        assert num_initial_tiles > 0, "num_initial_tiles must be positive."
        assert isinstance(spawn_rates, dict), "`spawn_rates` must be a dictionary."
        assert np.isclose(sum(spawn_rates.values()), 1.0), "Spawn probabilities must sum to 1."

        self.height: int = height
        self.width: int = width
        self.num_spawn_tiles_per_move: int = num_spawn_tiles_per_move
        self.num_initial_tiles: int = num_initial_tiles

        # Copy spawn distribution into convenient parallel lists
        self._spawn_rates: Dict[int, float] = spawn_rates.copy()
        self._spawn_values: List[int] = list(self._spawn_rates.keys())
        self._spawn_weights: List[float] = list(self._spawn_rates.values())

        # Game state
        self._board: BoardType = np.zeros((self.height, self.width), dtype=np.int64)
        self._score: np.int64 = np.int64(0)
        self._game_over: bool = False

        # Seed initial tiles
        for _ in range(self.num_initial_tiles):
            self._add_random_tile()

    # ------------------------------------------------------------------ #
    #                         PUBLIC API                                  #
    # ------------------------------------------------------------------ #
    def get_board(self) -> BoardType:
        """Return a *copy* of the current board."""
        return self._board.copy()

    def get_score(self) -> np.int64:
        """Current accumulated score."""
        return self._score

    def is_game_over(self) -> bool:
        """True iff no moves remain."""
        if self._game_over:
            return True
        if not self._check_if_any_moves_possible():
            self._game_over = True
        return self._game_over

    def move(self, direction: int) -> Tuple[np.int64, np.int64]:
        """
        Attempt to move/merge tiles in the given direction.
        Returns (score_gain, total_score). If the move has no effect,
        returns (0, current_score). Marks game over if no moves remain.
        """
        assert 0 <= direction <= 3, "Direction must be 0, 1, 2, or 3."
        if self._game_over:
            return np.int64(0), self._score

        original = self._board.copy()
        working = self._board.copy()
        move_gain: np.int64 = np.int64(0)

        # Horizontal moves
        if direction in (self.LEFT, self.RIGHT):
            reverse = direction == self.RIGHT
            for r in range(self.height):
                processed, gain = self._process_line(working[r, :], reverse)
                working[r, :] = processed
                move_gain += gain
        # Vertical moves
        else:
            reverse = direction == self.DOWN
            for c in range(self.width):
                processed, gain = self._process_line(working[:, c], reverse)
                working[:, c] = processed
                move_gain += gain

        board_changed = not np.array_equal(original, working)

        if board_changed:
            self._board = working
            self._score += move_gain

            # Spawn new tiles
            for _ in range(self.num_spawn_tiles_per_move):
                if not self._add_random_tile():
                    break  # board full

            if not self._check_if_any_moves_possible():
                self._game_over = True

            return move_gain, self._score
        else:
            if not self._check_if_any_moves_possible():
                self._game_over = True
            return np.int64(0), self._score

    def clone(self) -> "Simplified2048":
        """Return a deep copy of the game object (fast, avoids __init__)."""
        new_game = Simplified2048.__new__(Simplified2048)
        new_game.height = self.height
        new_game.width = self.width
        new_game.num_spawn_tiles_per_move = self.num_spawn_tiles_per_move
        new_game.num_initial_tiles = self.num_initial_tiles
        new_game._spawn_rates = self._spawn_rates.copy()
        new_game._spawn_values = self._spawn_values[:]
        new_game._spawn_weights = self._spawn_weights[:]
        new_game._board = self._board.copy()
        new_game._score = self._score
        new_game._game_over = self._game_over
        return new_game

    def render_ascii(self, cell_width: int = 6) -> str:
        """Return an ASCII art string visualizing the board."""
        sep = "+" + ("-" * cell_width + "+") * self.width
        out_lines: List[str] = [sep]
        for r in range(self.height):
            row_parts = ["|"]
            for c in range(self.width):
                val = self._board[r, c]
                cell = str(val) if val != 0 else "."
                row_parts.append(cell.center(cell_width))
                row_parts.append("|")
            out_lines.append("".join(row_parts))
            out_lines.append(sep)
        return "\n".join(out_lines)

    def get_valid_moves(self) -> List[int]:
        """Return list of directions that would change the board."""
        valid: List[int] = []
        original = self._board.copy()
        for d in self.DIRECTIONS:
            temp = original.copy()
            self._execute_move_on_board(d, temp)
            if not np.array_equal(original, temp):
                valid.append(d)
        return valid

    def get_max_tile(self) -> np.int64:
        """Largest value on the current board."""
        return np.max(self._board) if self._board.size else np.int64(0)

    # ------------------------------------------------------------------ #
    #                    INTERNAL / HELPER FUNCTIONS                     #
    # ------------------------------------------------------------------ #
    def _add_random_tile(self) -> bool:
        """Place a random tile (according to spawn rates)."""
        empties = np.argwhere(self._board == 0)
        if len(empties) == 0:
            return False
        r, c = random.choice(empties)
        val = random.choices(self._spawn_values, weights=self._spawn_weights, k=1)[0]
        self._board[r, c] = val
        return True

    def _process_line(
        self, line: np.ndarray, reverse: bool
    ) -> Tuple[np.ndarray, np.int64]:
        """
        Slide/merge a 1‑D row or column.
        Returns (new_line, gained_score_for_line).
        """
        if reverse:
            line = line[::-1]

        compressed = line[line != 0]            # eliminate zeros
        result = np.zeros_like(line)
        gain = np.int64(0)
        write_idx = 0
        i = 0
        while i < len(compressed):
            v = compressed[i]
            if i + 1 < len(compressed) and v == compressed[i + 1]:
                merged = v * 2
                result[write_idx] = merged
                gain += merged
                write_idx += 1
                i += 2
            else:
                result[write_idx] = v
                write_idx += 1
                i += 1

        if reverse:
            result = result[::-1]
        return result, gain

    def _execute_move_on_board(self, direction: int, board: BoardType) -> None:
        """Apply move logic on `board` in‑place (helper for validity checks)."""
        if direction in (self.LEFT, self.RIGHT):
            reverse = direction == self.RIGHT
            for r in range(self.height):
                board[r, :], _ = self._process_line(board[r, :], reverse)
        else:
            reverse = direction == self.DOWN
            for c in range(self.width):
                board[:, c], _ = self._process_line(board[:, c], reverse)

    def _check_if_any_moves_possible(self) -> bool:
        """Fast check for empties or adjacent equal tiles."""
        if np.any(self._board == 0):
            return True
        # Horizontal adjacency
        if np.any(self._board[:, :-1] == self._board[:, 1:]):
            return True
        # Vertical adjacency
        if np.any(self._board[:-1, :] == self._board[1:, :]):
            return True
        return False


# --------------------------------------------------------------------------- #
#                           EXAMPLE (local test)                              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    game = Simplified2048()
    print("Initial board:")
    print(game.render_ascii())
    mv_gain, tot = game.move(game.DOWN)
    print(f"\nAfter DOWN (gain={mv_gain}, total={tot}):")
    print(game.render_ascii())
