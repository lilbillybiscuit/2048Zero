import numpy as np
import random
import copy
import numba
from typing import Dict, List, Tuple, Optional, Any, NamedTuple, Union

BoardType = np.ndarray[Any, np.dtype[np.int64]]

class GameState(NamedTuple):
    board: BoardType
    score: np.int64

@numba.njit(cache=True)
def _process_line_numba(line: BoardType) -> Tuple[BoardType, np.int64]:
    line_len = len(line)
    new_line = np.zeros_like(line)
    merge_score: np.int64 = np.int64(0)
    write_idx: int = 0
    last_merged: bool = False

    read_idx = 0
    processed_indices = -1
    target_idx = 0

    while read_idx < line_len:
        val = line[read_idx]
        if val == 0:
            read_idx += 1
            continue

        if target_idx > 0 and new_line[target_idx - 1] == val and not last_merged:
            merged_val = val * 2
            new_line[target_idx - 1] = merged_val
            merge_score += merged_val
            last_merged = True
        else:
            new_line[target_idx] = val
            last_merged = False
            target_idx += 1

        read_idx += 1

    return new_line, merge_score


@numba.njit(cache=True)
def _perform_move_logic_numba(board: BoardType, direction: int) -> Tuple[BoardType, np.int64]:
    height, width = board.shape
    new_board = board.copy()
    total_score_gain: np.int64 = np.int64(0)
    line_result: BoardType
    gain: np.int64

    if direction == 3: # Left
        for r in range(height):
            line = new_board[r, :]
            processed_line, gain = _process_line_numba(line)
            new_board[r, :] = processed_line
            total_score_gain += gain
    elif direction == 1: # Right
        for r in range(height):
            line = new_board[r, ::-1]
            processed_line, gain = _process_line_numba(line)
            new_board[r, :] = processed_line[::-1]
            total_score_gain += gain
    elif direction == 0: # Up
        for c in range(width):
            col = new_board[:, c].copy() # Need copy for Numba safety
            processed_col, gain = _process_line_numba(col)
            new_board[:, c] = processed_col
            total_score_gain += gain
    elif direction == 2: # Down
        for c in range(width):
            col = new_board[::-1, c].copy() # Need copy for Numba safety
            processed_col, gain = _process_line_numba(col)
            new_board[:, c] = processed_col[::-1]
            total_score_gain += gain

    return new_board, total_score_gain

@numba.njit(cache=True)
def _check_if_any_moves_possible_numba(board: BoardType) -> bool:
    height, width = board.shape

    if np.any(board == 0):
       return True

    for r in range(height):
        for c in range(width - 1):
            if board[r, c] != 0 and board[r, c] == board[r, c + 1]:
                return True
    for c in range(width):
        for r in range(height - 1):
             if board[r, c] != 0 and board[r, c] == board[r + 1, c]:
                return True

    return False

@numba.njit(cache=True)
def _add_random_tile_numba(board: BoardType, spawn_values: np.ndarray, spawn_weights: np.ndarray) -> Optional[Tuple[int, int, int]]:
    empty_rows, empty_cols = np.where(board == 0)
    num_empty = len(empty_rows)

    if num_empty == 0:
        return None

    idx: int = random.randrange(num_empty)
    r: int = empty_rows[idx]
    c: int = empty_cols[idx]

    # Numba doesn't support random.choices directly with weights
    # Workaround using cumulative weights and random.random()
    choice_idx = np.searchsorted(np.cumsum(spawn_weights), random.random(), side="right")
    val = spawn_values[choice_idx]

    return r, c, val


class GameRules:
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
    _spawn_values_np: np.ndarray
    _spawn_weights_np: np.ndarray

    def __init__(self,
                 height: int = 4,
                 width: int = 4,
                 spawn_rates: Dict[int, float] = DEFAULT_SPAWN_RATES,
                 num_spawn_tiles_per_move: int = 1,
                 num_initial_tiles: int = 2):
        assert height > 0 and width > 0
        assert num_spawn_tiles_per_move > 0
        assert num_initial_tiles >= 0
        max_tiles = height * width
        assert num_initial_tiles <= max_tiles
        assert isinstance(spawn_rates, dict)
        assert all(isinstance(k, int) and k > 0 for k in spawn_rates.keys())
        assert np.isclose(sum(spawn_rates.values()), 1.0)

        self.height = height
        self.width = width
        self.num_spawn_tiles_per_move = num_spawn_tiles_per_move
        self.num_initial_tiles = num_initial_tiles

        self._spawn_rates = spawn_rates.copy()
        # Store as numpy arrays for Numba compatibility
        self._spawn_values_np = np.array(list(self._spawn_rates.keys()), dtype=np.int64)
        self._spawn_weights_np = np.array(list(self._spawn_rates.values()), dtype=np.float64)


    def get_initial_state(self) -> GameState:
        board = np.zeros((self.height, self.width), dtype=np.int64)
        score = np.int64(0)
        max_tiles = self.height * self.width
        actual_initial_tiles = min(self.num_initial_tiles, max_tiles)

        new_board = board
        for _ in range(actual_initial_tiles):
            # Directly call the tile adding logic without modifying instance state
            spawn_result = _add_random_tile_numba(new_board, self._spawn_values_np, self._spawn_weights_np)
            if spawn_result:
                r, c, val = spawn_result
                temp_board = new_board.copy() # Create copy before modification
                temp_board[r, c] = val
                new_board = temp_board # Update reference
            else:
                break # Should not happen with initial empty board if max_tiles respected

        return GameState(new_board, score)

    def apply_move(self, board: BoardType, direction: int) -> Tuple[BoardType, np.int64]:
        assert 0 <= direction <= 3
        if board.dtype != np.int64:
           board = board.astype(np.int64)
        return _perform_move_logic_numba(board, direction)

    def simulate_move(self, board: BoardType, direction: int) -> Tuple[bool, np.int64, BoardType]:
        assert 0 <= direction <= 3
        if board.dtype != np.int64:
           board = board.astype(np.int64)

        new_board, score_gain = _perform_move_logic_numba(board, direction)
        board_changed: bool = np.any(board != new_board)
        return board_changed, score_gain, new_board

    def get_valid_moves(self, board: BoardType) -> List[int]:
        valid_moves: List[int] = []
        for direction in self.DIRECTIONS:
            board_changed, _, _ = self.simulate_move(board, direction)
            if board_changed:
                valid_moves.append(direction)
        return valid_moves

    def is_terminal(self, board: BoardType) -> bool:
        return not _check_if_any_moves_possible_numba(board)

    def add_random_tiles(self, board: BoardType, return_action=False) -> Union[Tuple[BoardType, List[Tuple[int]]], BoardType]:
        """
        Either returns a new board with random tiles added, or the new board and a list of actions taken.
        :param board:
        :param return_action:
        :return:
        """
        new_board = board.copy()
        added_count = 0
        actions = []
        for _ in range(self.num_spawn_tiles_per_move):
            spawn_result = _add_random_tile_numba(new_board, self._spawn_values_np, self._spawn_weights_np)
            if spawn_result:
                r, c, val = spawn_result
                new_board[r, c] = val # Modify the copy
                added_count += 1
                if return_action:
                    actions.append((r, c, val))
            else:
                break # No empty space left
        if return_action:
            return new_board, actions
        else:
            return new_board

    def get_max_tile(self, board: BoardType) -> np.int64:
         if board.size == 0:
             return np.int64(0)
         return np.max(board)

    def render_ascii(self, board: BoardType, cell_width: int = 6) -> str:
        separator: str = "+" + ("-" * cell_width + "+") * self.width
        output: List[str] = [separator]
        for r in range(self.height):
            row_str: List[str] = ["|"]
            for c in range(self.width):
                val: np.int64 = board[r, c]
                cell_str: str = str(val) if val != 0 else "."
                row_str.append(cell_str.center(cell_width))
                row_str.append("|")
            output.append("".join(row_str))
            output.append(separator)
        return "\n".join(output)


class GameRunner:
    rules: GameRules
    current_board: BoardType
    current_score: np.int64
    game_over: bool

    def __init__(self, rules: GameRules):
        self.rules = rules
        initial_state = self.rules.get_initial_state()
        self.current_board = initial_state.board
        self.current_score = initial_state.score
        self.game_over = self.rules.is_terminal(self.current_board)

    def get_board(self) -> BoardType:
        return self.current_board.copy()

    def get_score(self) -> np.int64:
        return self.current_score

    def is_game_over(self) -> bool:
        if self.game_over:
            return True
        # Re-check in case tiles were generated into a terminal state
        self.game_over = self.rules.is_terminal(self.current_board)
        return self.game_over

    def move(self, direction: int) -> Tuple[np.int64, np.int64]:
        if self.is_game_over():
            return np.int64(0), self.current_score
        assert 0 <= direction <= 3

        board_changed, score_gain, new_board = self.rules.simulate_move(
            self.current_board, direction
        )

        if board_changed:
            self.current_board = new_board
            self.current_score += score_gain
            # Don't check game over here, check after potential tile generation
            return score_gain, self.current_score
        else:
            return np.int64(0), self.current_score

    def generate_tiles(self) -> bool:
        if self.is_game_over():
             return False # Cannot generate if already over

        initial_empty = np.count_nonzero(self.current_board == 0)
        if initial_empty == 0:
            self.game_over = self.rules.is_terminal(self.current_board)
            return False # Board is full

        self.current_board = self.rules.add_random_tiles(self.current_board)

        final_empty = np.count_nonzero(self.current_board == 0)
        tiles_generated = initial_empty - final_empty

        # Check game over *after* adding tiles
        self.game_over = self.rules.is_terminal(self.current_board)

        return tiles_generated >= min(initial_empty, self.rules.num_spawn_tiles_per_move)


    def clone(self) -> 'GameRunner':
        new_runner = GameRunner.__new__(GameRunner)
        new_runner.rules = self.rules # Share the rules object
        new_runner.current_board = self.current_board.copy()
        new_runner.current_score = self.current_score
        new_runner.game_over = self.game_over
        return new_runner

    def render_ascii(self, cell_width: int = 6) -> str:
        return self.rules.render_ascii(self.current_board, cell_width)

    def get_valid_moves(self) -> List[int]:
        if self.is_game_over():
            return []
        return self.rules.get_valid_moves(self.current_board)

    def get_max_tile(self) -> np.int64:
        return self.rules.get_max_tile(self.current_board)


if __name__ == "__main__":

    print("--- Standard 4x4 Game (1 tile/move, New Arch) ---")
    rules_4x4_1tile = GameRules(height=4, width=4, num_spawn_tiles_per_move=1)
    game = GameRunner(rules_4x4_1tile)
    print("Initial State:")
    print(game.render_ascii())
    print(f"Score: {game.get_score()}")
    print(f"Game Over: {game.is_game_over()}")
    print(f"Valid Moves: {game.get_valid_moves()}")

    valid_moves_initial = game.get_valid_moves()
    chosen_move = GameRules.DOWN if GameRules.DOWN in valid_moves_initial else (valid_moves_initial[0] if valid_moves_initial else -1)

    if chosen_move != -1:
        print(f"\nAttempting move: {chosen_move}")
        score_gain, total_score = game.move(chosen_move)
        print(f"Score Gain: {score_gain}, New Total Score: {total_score}")
        if score_gain > 0 or np.count_nonzero(game.current_board == 0) > 0:
           tiles_generated = game.generate_tiles()
           print(f"Tiles Generated: {tiles_generated}")
        else:
            print("Skipping tile generation as move had no effect or board full")


        print("\nAfter move + tile generation:")
        print(game.render_ascii())
        print(f"Score: {game.get_score()}")
        print(f"Game Over: {game.is_game_over()}")
        print(f"Valid Moves: {game.get_valid_moves()}")
    else:
        print("\nNo valid moves initially.")
    print("-" * 40)


    print("--- Standard 4x4 Game (2 tiles/move, New Arch) ---")
    rules_4x4_2tile = GameRules(height=4, width=4, num_spawn_tiles_per_move=2)
    game_2_tiles = GameRunner(rules_4x4_2tile)
    print("Initial State:")
    print(game_2_tiles.render_ascii())
    print(f"Score: {game_2_tiles.get_score()}")

    if GameRules.RIGHT in game_2_tiles.get_valid_moves():
        game_2_tiles.move(GameRules.RIGHT)
        game_2_tiles.generate_tiles()
        print("\nAfter RIGHT move + tile generation:")
        print(game_2_tiles.render_ascii())
        print(f"Score: {game_2_tiles.get_score()}")
    else:
       print("\nRIGHT move not valid.")


    if GameRules.UP in game_2_tiles.get_valid_moves():
        game_2_tiles.move(GameRules.UP)
        game_2_tiles.generate_tiles()
        print("\nAfter UP move + tile generation:")
        print(game_2_tiles.render_ascii())
        print(f"Score: {game_2_tiles.get_score()}")
    else:
       print("\nUP move not valid.")
    print("-" * 40)


    print("--- Custom 3x5 Game (More 4s, 1 tile/move, New Arch) ---")
    custom_rates: Dict[int, float] = {2: 0.5, 4: 0.5}
    rules_custom = GameRules(height=3, width=5, spawn_rates=custom_rates, num_spawn_tiles_per_move=1)
    game_custom = GameRunner(rules_custom)
    print("Initial State:")
    print(game_custom.render_ascii(cell_width=5))
    print(f"Score: {game_custom.get_score()}")

    move_names: Dict[int, str] = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
    turn: int = 0
    while not game_custom.is_game_over():
        turn += 1
        valid_moves: List[int] = game_custom.get_valid_moves()
        if not valid_moves:
             print(f"\nTurn {turn}: No valid moves detected! Game over.")
             break

        chosen_move: int = random.choice(valid_moves)
        print(f"\nTurn {turn}: Action: {move_names[chosen_move]} (Valid: {valid_moves})")

        board_before = game_custom.get_board()
        gain, _ = game_custom.move(chosen_move)
        board_after_move = game_custom.get_board()

        tiles_generated = False
        # Only generate tiles if the move changed the board
        if np.any(board_before != board_after_move):
            tiles_generated = game_custom.generate_tiles()

        print(game_custom.render_ascii(cell_width=5))
        print(f"Score Gain: {gain}, Total Score: {game_custom.get_score()}")
        if not tiles_generated and np.count_nonzero(game_custom.current_board) == game_custom.rules.height * game_custom.rules.width:
             print("Tile generation failed or skipped - board is full.")


    print(f"\n--- CUSTOM GAME FINISHED (Turn {turn}) ---")
    print("Final Board:")
    print(game_custom.render_ascii(cell_width=5))
    print(f"Final Score: {game_custom.get_score()}")
    print(f"Max Tile: {game_custom.get_max_tile()}")
    print(f"Is Game Over: {game_custom.is_game_over()}")


    print("\n--- Cloning Demo (2 tiles/move config, New Arch) ---")
    rules_clone_test = GameRules(4, 4, num_spawn_tiles_per_move=2)
    game_to_clone = GameRunner(rules_clone_test)
    print("Original Game (Turn 0):")
    print(game_to_clone.render_ascii())

    clone = game_to_clone.clone()
    print(f"Clone num_spawn_tiles_per_move: {clone.rules.num_spawn_tiles_per_move}")

    if GameRules.RIGHT in game_to_clone.get_valid_moves():
        game_to_clone.move(GameRules.RIGHT)
        game_to_clone.generate_tiles()
        print("\nOriginal Game After RIGHT + Generation:")
        print(game_to_clone.render_ascii())
    else:
        print("\nOriginal Game: RIGHT move not valid.")


    print("Clone State (Should be unchanged from Original Turn 0):")
    print(clone.render_ascii())

    if GameRules.UP in clone.get_valid_moves():
        clone.move(GameRules.UP)
        clone.generate_tiles()
        print("\nClone State After UP + Generation:")
        print(clone.render_ascii())
        print(f"Clone Score: {clone.get_score()}")
    else:
         print("\nClone Game: UP move not valid.")

    # Verify MCTS usage pattern: using rules directly
    print("\n--- MCTS Simulation Example ---")
    rules = GameRules(4, 4)
    start_state = rules.get_initial_state()
    print("MCTS Start State:")
    print(rules.render_ascii(start_state.board))
    print(f"Score: {start_state.score}")

    valid_moves = rules.get_valid_moves(start_state.board)
    if valid_moves:
        print(f"Valid moves from start state: {valid_moves}")
        # Simulate taking the first valid move
        chosen_move = valid_moves[0]
        changed, gain, board_after_move = rules.simulate_move(start_state.board, chosen_move)
        print(f"\nSimulating move {chosen_move}: Changed={changed}, Gain={gain}")
        print("Board after move:")
        print(rules.render_ascii(board_after_move))

        # Simulate adding random tiles (stochastic step)
        board_after_tiles = rules.add_random_tiles(board_after_move)
        print("\nBoard after random tile generation:")
        print(rules.render_ascii(board_after_tiles))
        print(f"Terminal state after tiles: {rules.is_terminal(board_after_tiles)}")
    else:
        print("No valid moves from start state.")