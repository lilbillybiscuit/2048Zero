from game import Simplified2048Adapter
from interface import PygameInterface
import random
import numpy as np

if __name__ == '__main__':
    # --- Configuration ---
    BOARD_HEIGHT = 4
    BOARD_WIDTH = 4
    SPAWN_RATES_CONFIG = {2: 0.9, 4: 0.1}
    WIN_TILE_CONFIG = 2048
    NUM_SPAWN_TILES_PER_MOVE = 1
    NUM_INITIAL_TILES = 2
    ENABLE_INTERFACE = True  # Set to False to run logic only

    # --- Create Game Logic with Adapter ---
    game = Simplified2048Adapter(
        height=BOARD_HEIGHT,
        width=BOARD_WIDTH,
        spawn_rates=SPAWN_RATES_CONFIG,
        num_spawn_tiles_per_move=NUM_SPAWN_TILES_PER_MOVE,
        num_initial_tiles=NUM_INITIAL_TILES,
        win_tile=WIN_TILE_CONFIG
    )

    # --- Optional: Attach Interface ---
    if ENABLE_INTERFACE:
        interface = PygameInterface(game)
        interface.run()  # Start the Pygame loop
    else:
        # --- Example: Programmatic Interaction (AI Agent Stub) ---
        print("Running Simplified2048 Logic without Interface...")
        print("Initial Board:")
        print(np.array2string(game.get_board()))
        print(f"Score: {game.get_score()}")

        # Simple agent: random valid moves
        move_count = 0
        max_moves = 1000  # Limit moves for example

        # Direction mapping from int to string
        move_names = {
            0: "up",
            1: "right",
            2: "down",
            3: "left"
        }

        while not game.is_game_over() and move_count < max_moves:
            possible_moves = ['left', 'right', 'up', 'down']
            random.shuffle(possible_moves)
            moved = False

            for move_dir in possible_moves:
                m, score_gain, _ = game.move(move_dir)
                if m:
                    print(f"\nMove {move_count + 1}: {move_dir}, Score Gain: {score_gain}")
                    # Add new tile
                    game._add_random_tile()
                    print(np.array2string(game.get_board()))
                    print(f"Score: {game.get_score()}")
                    moved = True
                    move_count += 1
                    break  # Go to next turn after successful move

            if not moved:
                # This should only happen if game becomes over unexpectedly
                print("No valid move found, but game not over? Error likely.")
                break

        print("\n--- Game Finished ---")
        print("Final Board:")
        print(np.array2string(game.get_board()))
        print(f"Final Score: {game.get_score()}")
        print(f"Highest Tile: {game.get_highest_tile()}")
        if game.has_won():
            print("Result: Won!")
        elif game.is_game_over():
            print("Result: Game Over!")
        else:
            print("Result: Max moves reached.")