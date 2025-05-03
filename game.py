import numpy as np
import random
import math
import time # Can use time for internal logic if needed, but not Pygame time

# --- Constants for Logic (can be overridden) ---
DEFAULT_BOARD_SIZE = 4
DEFAULT_SPAWN_RATES = {2: 0.9, 4: 0.1}
WIN_TILE = 2048

class Game2048Animation:
    def __init__(self, size=DEFAULT_BOARD_SIZE, spawn_rates=DEFAULT_SPAWN_RATES, win_tile=WIN_TILE):
        self.size = size
        self.spawn_rates = spawn_rates
        self.win_tile = win_tile
        self.listeners = [] # For callbacks

        self.score = 0
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.game_over = False
        self.game_won = False # Track win state separately
        self.highest_tile = 0
        self._last_spawned_tile = None # Store (r, c, value) of the last spawned tile

        self._initialize_board()

    def add_listener(self, listener):
        if listener not in self.listeners:
            self.listeners.append(listener)

    def remove_listener(self, listener):
        self.listeners.remove(listener)

    def _notify(self, event_name, *args, **kwargs):
        for listener in self.listeners:
            if hasattr(listener, event_name):
                getattr(listener, event_name)(*args, **kwargs)

    def _initialize_board(self):
        self.score = 0
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.game_over = False
        self.game_won = False
        self.highest_tile = 0
        self._last_spawned_tile = None
        self._add_random_tile()
        self._add_random_tile()
        self._notify('on_reset', self.board.copy(), self.score) # Notify listeners of initial state

    def reset(self):
        self._initialize_board()

    def get_board(self):
        return self.board.copy() # Return a copy to prevent external modification

    def get_score(self):
        return self.score

    def is_game_over(self):
        return self.game_over

    def has_won(self):
        return self.game_won

    def get_highest_tile(self):
        return self.highest_tile

    def get_last_spawned_tile(self):
        return self._last_spawned_tile

    def _add_random_tile(self):
        empty_cells = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]
        if not empty_cells:
            return False

        r, c = random.choice(empty_cells)
        new_value = random.choices(list(self.spawn_rates.keys()), weights=list(self.spawn_rates.values()), k=1)[0]
        self.board[r, c] = new_value
        self.highest_tile = max(self.highest_tile, new_value)
        self._last_spawned_tile = (r, c, new_value)

        self._notify('on_tile_spawned', r, c, new_value)
        return True

    def _compress(self, line):
        new_line = []
        original_indices = []
        for i, x in enumerate(line):
            if x != 0:
                new_line.append(x)
                original_indices.append(i)
        return new_line, original_indices

    def _merge(self, line_values, original_indices):
        score_gain = 0
        merged_values = []
        merged_indices = [] # Tracks the original index that *results* in the merged value
        merge_animations_info = [] # List of ((start1_idx, start2_idx), end_idx, value) within the line axis

        i = 0
        while i < len(line_values):
            if i + 1 < len(line_values) and line_values[i] == line_values[i+1]:
                merged_value = line_values[i] * 2
                score_gain += merged_value
                self.highest_tile = max(self.highest_tile, merged_value)
                merged_values.append(merged_value)
                # Resulting tile ends up at the position corresponding to the second original tile
                merged_indices.append(original_indices[i+1])
                merge_animations_info.append(((original_indices[i], original_indices[i+1]), original_indices[i+1], merged_value))
                i += 2
            else:
                merged_values.append(line_values[i])
                merged_indices.append(original_indices[i])
                i += 1

        return merged_values, merged_indices, score_gain, merge_animations_info

    def move(self, direction):
        """
        Attempts to move tiles in the given direction ('up', 'down', 'left', 'right').

        Returns:
            tuple: (moved, score_gain, animation_steps)
            - moved (bool): True if the board state changed, False otherwise.
            - score_gain (int): Score increase from this move.
            - animation_steps (list or None): A list describing tile movements and merges
              for visualization, or None if no move occurred. Each step is a dict:
              {'value': v, 'start': (r,c), 'end': (r,c), 'type': 'move'/'merge_source'/'merge_result'}
        """
        if self.game_over:
            return False, 0, None

        original_board = self.board.copy()
        temp_board = self.board.copy()
        current_score_gain = 0
        animation_steps = []

        if direction in ['left', 'right']:
            for r in range(self.size):
                line = temp_board[r, :].copy()
                is_reversed = (direction == 'right')
                if is_reversed:
                    line = line[::-1]

                compressed_values, compressed_orig_indices = self._compress(line.tolist())
                merged_values, merged_orig_indices, line_score_gain, merges_info = self._merge(compressed_values, compressed_orig_indices)
                final_line_list = merged_values + [0] * (self.size - len(merged_values))

                final_line = np.array(final_line_list, dtype=int)
                if is_reversed:
                    final_line = final_line[::-1]

                temp_board[r, :] = final_line
                current_score_gain += line_score_gain

                # Convert line-based merge info to board coordinates and add to animation steps
                for (start1_idx, start2_idx), end_idx, value in merges_info:
                    s1_c = start1_idx if not is_reversed else self.size - 1 - start1_idx
                    s2_c = start2_idx if not is_reversed else self.size - 1 - start2_idx
                    e_c = end_idx if not is_reversed else self.size - 1 - end_idx
                    animation_steps.append({'value': value // 2, 'start': (r, s1_c), 'end': (r, e_c), 'type': 'merge_source'})
                    animation_steps.append({'value': value // 2, 'start': (r, s2_c), 'end': (r, e_c), 'type': 'merge_source'})
                    animation_steps.append({'value': value, 'end': (r, e_c), 'type': 'merge_result'})

        elif direction in ['up', 'down']:
            for c in range(self.size):
                line = temp_board[:, c].copy()
                is_reversed = (direction == 'down')
                if is_reversed:
                    line = line[::-1]

                compressed_values, compressed_orig_indices = self._compress(line.tolist())
                merged_values, merged_orig_indices, line_score_gain, merges_info = self._merge(compressed_values, compressed_orig_indices)
                final_line_list = merged_values + [0] * (self.size - len(merged_values))

                final_line = np.array(final_line_list, dtype=int)
                if is_reversed:
                    final_line = final_line[::-1]

                temp_board[:, c] = final_line
                current_score_gain += line_score_gain

                # Convert line-based merge info to board coordinates
                for (start1_idx, start2_idx), end_idx, value in merges_info:
                    s1_r = start1_idx if not is_reversed else self.size - 1 - start1_idx
                    s2_r = start2_idx if not is_reversed else self.size - 1 - start2_idx
                    e_r = end_idx if not is_reversed else self.size - 1 - end_idx
                    animation_steps.append({'value': value // 2, 'start': (s1_r, c), 'end': (e_r, c), 'type': 'merge_source'})
                    animation_steps.append({'value': value // 2, 'start': (s2_r, c), 'end': (e_r, c), 'type': 'merge_source'})
                    animation_steps.append({'value': value, 'end': (e_r, c), 'type': 'merge_result'})

        moved = not np.array_equal(original_board, temp_board)

        if moved:
            self.board = temp_board
            self.score += current_score_gain

            # Clear existing animation steps and rebuild with direction-aware logic
            animation_steps = []

            # First, add all merge animations (these are correct from existing code)
            merge_source_starts = set()
            merge_result_ends = {}

            for step in animation_steps:
                if step['type'] == 'merge_source':
                    merge_source_starts.add(step['start'])

                if step['type'] == 'merge_result':
                    merge_result_ends[step['end']] = step['value']

            # Direction-aware movement determination
            # For each non-zero tile in the final board that isn't a merge result
            direction_offsets = {
                'left': (0, -1),
                'right': (0, 1),
                'up': (-1, 0),
                'down': (1, 0)
            }
            dr, dc = direction_offsets[direction]

            # Track processed original positions to avoid duplication
            processed_origins = set(merge_source_starts)

            # For each cell in the final state
            for r_final in range(self.size):
                for c_final in range(self.size):
                    value = self.board[r_final, c_final]
                    if value == 0: continue

                    final_pos = (r_final, c_final)
                    if final_pos in merge_result_ends: continue

                    # Look backward along the direction vector for the source
                    found = False
                    r_check, c_check = r_final, c_final

                    # Search backward in direction of movement
                    while 0 <= r_check < self.size and 0 <= c_check < self.size:
                        r_check -= dr
                        c_check -= dc

                        # If we go out of bounds, break
                        if not (0 <= r_check < self.size and 0 <= c_check < self.size):
                            break

                        check_pos = (r_check, c_check)

                        # If we find a matching value in original board and it's not already processed
                        if original_board[check_pos] == value and check_pos not in processed_origins:
                            # We found our source
                            if check_pos != final_pos:  # Only add if actually moved
                                animation_steps.append({
                                    'value': value,
                                    'start': check_pos,
                                    'end': final_pos,
                                    'type': 'move'
                                })
                            processed_origins.add(check_pos)
                            found = True
                            break

                    # If no source found backward, use the same position if it had the value
                    if not found and original_board[final_pos] == value and final_pos not in processed_origins:
                        # Tile didn't move, but should be included in processed list
                        processed_origins.add(final_pos)

            # Now fix the new tile spawning timing in interface.py
            self._notify('on_move_complete', animation_steps)

            # IMPORTANT CHANGE: Move new tile generation to after animation completes
            # This gets handled through the animation callbacks now
            # The PygameInterface will call the appropriate method when animations finish

            # Check game state after adding tile
            if not self._any_moves_possible():
                self.game_over = True
                self._notify('on_game_over')

            if not self.game_won and self.highest_tile >= self.win_tile:
                self.game_won = True
                self._notify('on_win')

            return True, current_score_gain, animation_steps
        else:
            # Check for game over if no move was possible
            if not self._any_moves_possible():
                self.game_over = True
                self._notify('on_game_over')
            return False, 0, None

    def _any_moves_possible(self):
        # Check for empty cells
        if np.any(self.board == 0):
            return True
        # Check for possible merges horizontally
        for r in range(self.size):
            for c in range(self.size - 1):
                if self.board[r, c] == self.board[r, c+1]:
                    return True
        # Check for possible merges vertically
        for c in range(self.size):
            for r in range(self.size - 1):
                if self.board[r, c] == self.board[r+1, c]:
                    return True
        return False

# --- End of Game Logic ---