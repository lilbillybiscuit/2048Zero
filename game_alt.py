import numpy as np
import random
import copy
from typing import Dict, List, Tuple, Optional, Any

# Define a type alias for the board for clarity
BoardType = np.ndarray[Any, np.dtype[np.int64]]

class Simplified2048:
    """
    A simplified, programmatic-only version of the 2048 game with
    customizable dimensions, spawn rates, and number of tiles spawned per move.
    Designed for AI experiments like MCTS. No GUI.
    Uses integer directions: 0: Up, 1: Right, 2: Down, 3: Left.
    """
    UP: int = 0
    RIGHT: int = 1
    DOWN: int = 2
    LEFT: int = 3
    DIRECTIONS: List[int] = [UP, RIGHT, DOWN, LEFT]
    DEFAULT_SPAWN_RATES: Dict[int, float] = {2: 0.9, 4: 0.1}
    height: int
    width: int
    num_spawn_tiles_per_move: int
    num_initial_tiles: int
    _spawn_rates: Dict[int, float]
    _spawn_values: List[int]
    _spawn_weights: List[float]
    _board: BoardType
    _score: np.int64
    _game_over: bool

    def __init__(self,
                 height: int = 4,
                 width: int = 4,
                 spawn_rates: Dict[int, float] = DEFAULT_SPAWN_RATES,
                 num_spawn_tiles_per_move: int= 2,
                 num_initial_tiles:int= 2):
        self.height = height
        self.width = width
        assert num_spawn_tiles_per_move > 0, "num_spawn_tiles_per_move must be positive."
        assert num_initial_tiles > 0, "num_initial_tiles must be positive."
        assert num_spawn_tiles_per_move <= self.height * self.width, "Cannot spawn more tiles than available spaces."
        assert height > 0 and width > 0, "Board dimensions must be positive."
        assert isinstance(spawn_rates, dict), "spawn_rates must be a dictionary."
        assert np.isclose(sum(spawn_rates.values()), 1.0), "Spawn rate probabilities must sum to 1.0."

        self.num_spawn_tiles_per_move = num_spawn_tiles_per_move
        self.num_initial_tiles = num_initial_tiles

        self._spawn_rates = spawn_rates.copy()
        self._spawn_values = list(self._spawn_rates.keys())
        self._spawn_weights = list(self._spawn_rates.values())

        self._board = np.zeros((self.height, self.width), dtype=np.int64)
        self._score = np.int64(0)
        self._game_over = False

        for _ in range(self.num_initial_tiles):
            self._add_random_tile()

    # --- Core API ---

    def get_board(self) -> BoardType:
        """Returns a copy of the current board state (numpy array)."""
        return self._board.copy()

    def get_score(self) -> np.int64:
        """Returns the current score."""
        return self._score

    def is_game_over(self) -> bool:
        """Returns True if the game is over, False otherwise."""
        if self._game_over:
            return True
        # Check lazily - only calculate if not already marked as over
        if not self._check_if_any_moves_possible():
            self._game_over = True
            return True
        return False

    def move(self, direction: int) -> Tuple[np.int64, np.int64]:
        """
        Attempts to move tiles in the given direction. Adds
        `num_spawn_tiles_per_move` new tiles if the board changed.

        Args:
            direction (int): 0: Up, 1: Right, 2: Down, 3: Left.

        Returns:
            tuple: (score_gain, total_score)
                   - score_gain (int): Score added by this move (0 if no move).
                   - total_score (int): The total score after the move.
                   Returns (0, current_score) if the move is invalid or game is over.
        """
        if self._game_over:
            return np.int64(0), self._score

        assert 0 <=direction and direction <= 3, "Invalid direction. Use 0 (Up), 1 (Right), 2 (Down), or 3 (Left)."

        original_board: BoardType = self._board.copy()
        new_board: BoardType = self._board.copy() # Start with current state
        total_score_gain: np.int64 = np.int64(0)

        # --- Execute Move Logic (No Rotation) ---
        reverse: bool
        processed_line: np.ndarray[Any, np.dtype[np.int64]]
        gain: np.int64
        if direction == self.LEFT or direction == self.RIGHT:
            reverse = (direction == self.RIGHT)
            for r in range(self.height):
                processed_line, gain = self._process_line(new_board[r, :], reverse)
                new_board[r, :] = processed_line
                total_score_gain += gain
        elif direction == self.UP or direction == self.DOWN:
            reverse = (direction == self.DOWN)
            for c in range(self.width):
                processed_line, gain = self._process_line(new_board[:, c], reverse)
                new_board[:, c] = processed_line
                total_score_gain += gain
        # --- End Move Logic ---

        board_changed: bool = not np.array_equal(original_board, new_board)

        if board_changed:
            self._board = new_board # Commit the changes
            self._score += total_score_gain

            # Add the configured number of tiles
            # self.generate_tiles()

            # Check for game over *after* attempting to add all tiles
            if not self._check_if_any_moves_possible():
                self._game_over = True

            return total_score_gain, self._score
        else:
            # Board didn't change, but check if game is over anyway
            if not self._check_if_any_moves_possible():
                self._game_over = True
            return np.int64(0), self._score

    def generate_tiles(self) -> bool:
        """
        Generates a new tile on the board based on spawn rates.
        Returns True if successful, False if no empty cells are available.
        """
        for _ in range(self.num_spawn_tiles_per_move):
            if not self._add_random_tile():
                return False
        return True

    def clone(self) -> 'Simplified2048':
        """
        Creates an efficient copy of the game state.
        """
        new_game = Simplified2048.__new__(Simplified2048) # Create instance without calling __init__
        new_game.height = self.height
        new_game.width = self.width
        # Copy spawn config
        new_game._spawn_rates = self._spawn_rates.copy()
        new_game._spawn_values = self._spawn_values[:]
        new_game._spawn_weights = self._spawn_weights[:]
        new_game.num_spawn_tiles_per_move = self.num_spawn_tiles_per_move
        new_game.num_initial_tiles = self.num_initial_tiles
        # Copy game state
        new_game._board = self._board.copy()
        new_game._score = self._score
        new_game._game_over = self._game_over
        return new_game

    def render_ascii(self, cell_width: int = 6) -> str:
        """Returns an ASCII string representation of the board."""
        separator: str = "+" + ("-" * cell_width + "+") * self.width
        output: List[str] = [separator]
        for r in range(self.height):
            row_str: List[str] = ["|"]
            for c in range(self.width):
                val: np.int64 = self._board[r, c]
                cell_str: str = str(val) if val != 0 else "."
                row_str.append(cell_str.center(cell_width))
                row_str.append("|")
            output.append("".join(row_str))
            output.append(separator)
        return "\n".join(output)

    # --- Additional Helper Methods ---

    def get_valid_moves(self) -> List[int]:
        """
        Returns a list of directions (int) that would result in a board change.
        """
        valid: List[int] = []
        original_board: BoardType = self._board.copy() # Keep original safe

        for direction in self.DIRECTIONS:
            temp_board: BoardType = original_board.copy() # Work on a copy for checking
            self._execute_move_on_board(direction, temp_board) # Simulate move on copy
            if not np.array_equal(original_board, temp_board):
                valid.append(direction)
        return valid

    def get_max_tile(self) -> np.int64:
        """Returns the value of the highest tile on the board."""
        return np.max(self._board) if self._board.size > 0 else np.int64(0)

    # --- Internal Logic ---

    def _add_random_tile(self) -> bool:
        """Adds a random tile based on spawn_rates to an empty cell. Returns True if successful."""
        empty_cells: np.ndarray = np.argwhere(self._board == 0)
        if len(empty_cells) == 0:
            return False # No space left

        idx: int = np.random.randint(len(empty_cells))
        r: int
        c: int
        r, c = empty_cells[idx]
        # Use pre-calculated lists for efficiency
        val: int = random.choices(self._spawn_values, weights=self._spawn_weights, k=1)[0]
        self._board[r, c] = val
        return True

    def _execute_move_on_board(self, direction: int, board: BoardType) -> None:
        """
        Helper for get_valid_moves. Performs move logic on a given board *without*
        updating score or internal state. Modifies the passed board directly.
        """
        reverse: bool
        processed_line: np.ndarray[Any, np.dtype[np.int64]]
        if direction == self.LEFT or direction == self.RIGHT:
            reverse = (direction == self.RIGHT)
            for r in range(self.height):
                processed_line, _ = self._process_line(board[r, :], reverse)
                board[r, :] = processed_line
        elif direction == self.UP or direction == self.DOWN:
            reverse = (direction == self.DOWN)
            for c in range(self.width):
                processed_line, _ = self._process_line(board[:, c], reverse)
                board[:, c] = processed_line

    def _process_line(self, line: np.ndarray[Any, np.dtype[np.int64]], reverse: bool) -> Tuple[np.ndarray[Any, np.dtype[np.int64]], np.int64]:
        """
        Processes a single line (row/column) for sliding and merging.

        Args:
            line (1D NumPy array): The line to process.
            reverse (bool): If True, process movement towards the end of the line.

        Returns:
            tuple: (new_line, merge_score)
        """
        line_len: int = len(line)
        temp_line: np.ndarray[Any, np.dtype[np.int64]] = line.copy()

        if reverse:
            temp_line = temp_line[::-1] # Reverse for processing

        # 1. Compress: Move non-zero elements to the left (index 0)
        compressed: np.ndarray[Any, np.dtype[np.int64]] = temp_line[temp_line != 0]
        processed_line: np.ndarray[Any, np.dtype[np.int64]] = np.zeros_like(temp_line)
        merge_score: np.int64 = np.int64(0)
        write_idx: int = 0

        # 2. Merge
        i: int = 0
        while i < len(compressed):
            current_val: np.int64 = compressed[i]

            if i + 1 < len(compressed) and current_val == compressed[i+1]:
                merged_val: np.int64 = current_val * 2
                processed_line[write_idx] = merged_val
                merge_score += merged_val
                write_idx += 1
                i += 2 # Skip both merged tiles
            else:
                processed_line[write_idx] = current_val
                write_idx += 1
                i += 1 # Move to next tile

        if reverse:
            processed_line = processed_line[::-1] # Reverse back

        return processed_line, merge_score

    def _check_if_any_moves_possible(self) -> bool:
        """Checks if any move would change the board state OR if empty cells exist."""
        # 1. Check for empty cells first (fastest check)
        if np.any(self._board == 0):
            return True

        # 2. Check for adjacent identical cells (possible merges)
        # Horizontal checks
        for r in range(self.height):
            for c in range(self.width - 1):
                if self._board[r, c] == self._board[r, c + 1]:
                    return True
        # Vertical checks
        for c in range(self.width):
            for r in range(self.height - 1):
                if self._board[r, c] == self._board[r + 1, c]:
                    return True

        # No empty cells and no possible merges
        return False

# --- Example Usage ---
if __name__ == "__main__":

    # set seeds
    # random.seed(42)
    # np.random.seed(42)

    # Example 1: Standard 4x4 (default: 1 tile per move)
    print("--- Standard 4x4 Game (1 tile/move) ---")
    game: Simplified2048 = Simplified2048(height=4, width=4)
    print("Initial State:")
    print(game.render_ascii())
    print(f"Score: {game.get_score()}")
    game.move(game.DOWN)
    print("\nAfter DOWN move:")
    print(game.render_ascii())
    print(f"Score: {game.get_score()}")
    print("-" * 40)

    # Example 2: 4x4 with 2 tiles per move
    print("--- Standard 4x4 Game (2 tiles/move) ---")
    game_2_tiles: Simplified2048 = Simplified2048(height=4, width=4, num_spawn_tiles_per_move=2)
    print("Initial State:")
    print(game_2_tiles.render_ascii())
    print(f"Score: {game_2_tiles.get_score()}")
    game_2_tiles.move(game_2_tiles.RIGHT)
    print("\nAfter RIGHT move:")
    print(game_2_tiles.render_ascii()) # Should have 2 new tiles if space allows
    print(f"Score: {game_2_tiles.get_score()}")
    game_2_tiles.move(game_2_tiles.UP)
    print("\nAfter UP move:")
    print(game_2_tiles.render_ascii()) # Should have 2 more new tiles
    print(f"Score: {game_2_tiles.get_score()}")
    print("-" * 40)


    # Example 3: Custom Size and Spawn Rates (More 4s)
    print("--- Custom 3x5 Game (More 4s, 1 tile/move) ---")
    custom_rates: Dict[int, float] = {2: 0.5, 4: 0.5}
    game_custom: Simplified2048 = Simplified2048(height=3, width=5, spawn_rates=custom_rates)
    print("Initial State:")
    print(game_custom.render_ascii(cell_width=5))
    print(f"Score: {game_custom.get_score()}")

    move_names: Dict[int, str] = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
    turn: int = 0
    gain: np.int64
    total_score: np.int64
    while not game_custom.is_game_over():
        turn += 1
        valid_moves: List[int] = game_custom.get_valid_moves()
        if not valid_moves: break
        chosen_move: int = random.choice(valid_moves)
        print(f"\nTurn {turn}: Action: {move_names[chosen_move]}")
        gain, total_score = game_custom.move(chosen_move)
        game_custom.generate_tiles()
        print(game_custom.render_ascii(cell_width=5))
        print(f"Score Gain: {gain}, Total Score: {total_score}")

    print("\n--- CUSTOM GAME OVER ---")
    print("Final Board:")
    print(game_custom.render_ascii(cell_width=5))
    print(f"Final Score: {game_custom.get_score()}")
    print(f"Max Tile: {game_custom.get_max_tile()}")

    # Example 4: Cloning with new config
    print("\n--- Cloning Demo (2 tiles/move config) ---")
    game_to_clone: Simplified2048 = Simplified2048(4, 4, num_spawn_tiles_per_move=2)
    print("Original Game (Turn 0):")
    print(game_to_clone.render_ascii())
    clone: Simplified2048 = game_to_clone.clone()
    print(f"Clone num_spawn_tiles_per_move: {clone.num_spawn_tiles_per_move}")
    # Move original
    game_to_clone.move(game_to_clone.RIGHT)
    print("\nOriginal Game After RIGHT:")
    print(game_to_clone.render_ascii())
    print("Clone State (Should be unchanged):")
    print(clone.render_ascii())
    # Move clone
    clone.move(clone.UP)
    print("\nClone State After UP:")
    print(clone.render_ascii()) # Should have 2 new tiles