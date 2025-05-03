import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from game_alt import Simplified2048


class Simplified2048Adapter:
    """
    Adapter class to make Simplified2048 compatible with PygameInterface.
    Translates between the Simplified2048 implementation and the interface
    expected by PygameInterface.
    """

    # Direction mapping from string to int
    DIRECTION_MAP = {
        'up': 0,
        'right': 1,
        'down': 2,
        'left': 3
    }

    # Default win tile
    WIN_TILE = 2048

    def __init__(self, height=4, width=4, spawn_rates={2: 0.9, 4: 0.1},
                 num_spawn_tiles_per_move=1, num_initial_tiles=2, win_tile=WIN_TILE):
        """Initialize with Simplified2048 parameters"""
        self.game = Simplified2048(
            height=height,
            width=width,
            spawn_rates=spawn_rates,
            num_spawn_tiles_per_move=num_spawn_tiles_per_move,
            num_initial_tiles=num_initial_tiles
        )
        self.listeners = []
        self.size = height  # Assuming square board where width == height for interface compatibility
        self.win_tile = win_tile
        self.game_won = False
        self.last_spawned_tile = None
        self.last_board = self.game.get_board().copy()

    def add_listener(self, listener):
        if listener not in self.listeners:
            self.listeners.append(listener)

    def remove_listener(self, listener):
        if listener in self.listeners:
            self.listeners.remove(listener)

    def _notify(self, event_name, *args, **kwargs):
        for listener in self.listeners:
            if hasattr(listener, event_name):
                getattr(listener, event_name)(*args, **kwargs)

    def get_board(self):
        return self.game.get_board().astype(int)  # Convert from np.int64 to int

    def get_score(self):
        return int(self.game.get_score())  # Convert from np.int64 to int

    def is_game_over(self):
        return self.game.is_game_over()

    def has_won(self):
        # Check win state based on max tile
        if self.game_won:
            return True

        if int(self.game.get_max_tile()) >= self.win_tile:
            self.game_won = True
            self._notify('on_win')
            return True
        return False

    def reset(self):
        # Create a new game with same parameters
        height = self.game.height
        width = self.game.width
        spawn_rates = self.game._spawn_rates
        tiles_per_move = self.game.num_spawn_tiles_per_move
        init_tiles = self.game.num_initial_tiles

        # Create new game instance
        self.game = Simplified2048(
            height=height,
            width=width,
            spawn_rates=spawn_rates,
            num_spawn_tiles_per_move=tiles_per_move,
            num_initial_tiles=init_tiles
        )

        self.game_won = False
        self.last_board = self.game.get_board().copy()
        self.last_spawned_tile = None

        # Notify listeners
        self._notify('on_reset', self.get_board(), self.get_score())

    def move(self, direction):
        """
        Process move and generate animation info

        Args:
            direction (str): 'up', 'right', 'down', 'left'

        Returns:
            tuple: (moved, score_gain, animation_steps)
        """
        if self.is_game_over():
            return False, 0, None

        # Convert string direction to integer
        dir_int = self.DIRECTION_MAP.get(direction)
        if dir_int is None:
            return False, 0, None

        # Save board state before move for calculating animations
        original_board = self.game.get_board().copy()

        # Perform the move
        score_gain, _ = self.game.move(dir_int)
        board_changed = score_gain > 0 or np.any(original_board != self.game.get_board())

        if not board_changed:
            return False, 0, None

        # Generate animation steps based on board comparison
        animation_steps = self._generate_animation_steps(original_board, self.game.get_board(), direction)

        # Notify listeners of move completion
        self._notify('on_move_complete', animation_steps)

        # Check win condition
        if not self.game_won and int(self.game.get_max_tile()) >= self.win_tile:
            self.game_won = True
            self._notify('on_win')

        return True, int(score_gain), animation_steps

    def _add_random_tile(self):
        """Add a random tile and notify listeners"""
        current_board = self.game.get_board()
        empty_cells_before = set(zip(*np.where(current_board == 0)))

        if self.game.generate_tiles():
            new_board = self.game.get_board()

            # Find the new tile(s)
            for r in range(self.game.height):
                for c in range(self.game.width):
                    if new_board[r, c] > 0 and current_board[r, c] == 0:
                        # Found a new tile
                        tile_value = int(new_board[r, c])
                        self.last_spawned_tile = (r, c, tile_value)
                        self._notify('on_tile_spawned', r, c, tile_value)

            self.last_board = new_board.copy()

            # Check if game is over after generating tiles
            if self.game.is_game_over():
                self._notify('on_game_over')

            return True
        return False

    def get_highest_tile(self):
        return int(self.game.get_max_tile())

    def get_last_spawned_tile(self):
        return self.last_spawned_tile

    def _generate_animation_steps(self, old_board, new_board, direction):
        """
        Generate animation steps for tile movements and merges

        Args:
            old_board: Board state before move
            new_board: Board state after move
            direction: String direction of the move

        Returns:
            List of animation step dictionaries
        """
        animation_steps = []

        # Track processed tiles
        processed_old_positions = set()
        processed_new_positions = set()

        # Direction vector for movement tracing
        dir_vector = {
            'up': (-1, 0),
            'right': (0, 1),
            'down': (1, 0),
            'left': (0, -1)
        }[direction]

        # First pass: Find merges - look for positions where value in new board is double some value in old board
        for r in range(self.game.height):
            for c in range(self.game.width):
                new_val = new_board[r, c]
                if new_val == 0:
                    continue

                # If this is a power of 2 greater than 2
                if new_val > 2 and (new_val & (new_val - 1)) == 0:
                    half_val = new_val // 2

                    # Count half_vals in old board
                    half_val_count = np.sum(old_board == half_val)

                    # If there are at least 2 half_vals in old board, this is likely a merge
                    if half_val_count >= 2:
                        # Find the half_val tiles in old board
                        half_val_positions = []
                        for old_r in range(self.game.height):
                            for old_c in range(self.game.width):
                                if old_board[old_r, old_c] == half_val:
                                    half_val_positions.append((old_r, old_c))

                        # Only use the first two sources we find
                        if len(half_val_positions) >= 2:
                            # Sort by closest to destination for more natural animation
                            half_val_positions.sort(key=lambda pos: abs(pos[0] - r) + abs(pos[1] - c))
                            source1, source2 = half_val_positions[:2]

                            # Add move animations for the merging tiles
                            animation_steps.append({
                                'value': half_val,
                                'start': source1,
                                'end': (r, c),
                                'type': 'merge_source'
                            })
                            animation_steps.append({
                                'value': half_val,
                                'start': source2,
                                'end': (r, c),
                                'type': 'merge_source'
                            })

                            # Add the merge result animation
                            animation_steps.append({
                                'value': new_val,
                                'end': (r, c),
                                'type': 'merge_result'
                            })

                            # Mark these positions as processed
                            processed_old_positions.add(source1)
                            processed_old_positions.add(source2)
                            processed_new_positions.add((r, c))

        # Second pass: Process simple movements - tiles that moved but didn't merge
        # Process tiles in the direction of movement to avoid ambiguity
        rows = list(range(self.game.height))
        cols = list(range(self.game.width))

        # Adjust iteration order based on direction
        if direction == 'down':
            rows = rows[::-1]
        elif direction == 'right':
            cols = cols[::-1]

        for r in rows:
            for c in cols:
                if new_board[r, c] > 0 and (r, c) not in processed_new_positions:
                    new_val = new_board[r, c]

                    # Look for source tiles in the direction opposite to movement
                    found_source = False
                    search_r, search_c = r, c
                    dr, dc = -dir_vector[0], -dir_vector[1]

                    # Try to find source along the direction of movement
                    while True:
                        search_r += dr
                        search_c += dc

                        if (search_r < 0 or search_r >= self.game.height or
                                search_c < 0 or search_c >= self.game.width):
                            break

                        if (old_board[search_r, search_c] == new_val and
                                (search_r, search_c) not in processed_old_positions):
                            # Found a source, add move animation
                            animation_steps.append({
                                'value': new_val,
                                'start': (search_r, search_c),
                                'end': (r, c),
                                'type': 'move'
                            })

                            processed_old_positions.add((search_r, search_c))
                            processed_new_positions.add((r, c))
                            found_source = True
                            break

                    # If not found in line, search the whole board
                    if not found_source:
                        for search_r in range(self.game.height):
                            for search_c in range(self.game.width):
                                if (old_board[search_r, search_c] == new_val and
                                        (search_r, search_c) not in processed_old_positions):

                                    # Skip if it's the same position (didn't move)
                                    if (search_r, search_c) == (r, c):
                                        processed_old_positions.add((search_r, search_c))
                                        processed_new_positions.add((r, c))
                                        found_source = True
                                        break

                                    # Add move animation
                                    animation_steps.append({
                                        'value': new_val,
                                        'start': (search_r, search_c),
                                        'end': (r, c),
                                        'type': 'move'
                                    })

                                    processed_old_positions.add((search_r, search_c))
                                    processed_new_positions.add((r, c))
                                    found_source = True
                                    break

                            if found_source:
                                break

        return animation_steps