from game import Game2048Animation
from interface import PygameInterface


if __name__ == '__main__':
    # --- Configuration ---
    BOARD_SIZE_CONFIG = 4
    SPAWN_RATES_CONFIG = {2: 0.9, 4: 0.1}
    WIN_TILE_CONFIG = 2048
    ENABLE_INTERFACE = True # Set to False to run logic only

    # --- Create Game Logic ---
    game = Game2048Animation(size=BOARD_SIZE_CONFIG,
                             spawn_rates=SPAWN_RATES_CONFIG,
                             win_tile=WIN_TILE_CONFIG)

    # --- Optional: Attach Interface ---
    if ENABLE_INTERFACE:
        interface = PygameInterface(game)
        interface.run() # Start the Pygame loop
    else:
        # --- Example: Programmatic Interaction (AI Agent Stub) ---
        print("Running 2048 Logic without Interface...")
        print("Initial Board:")
        print(game.get_board())
        print(f"Score: {game.get_score()}")

        # Simple agent: random valid moves
        move_count = 0
        max_moves = 1000 # Limit moves for example
        while not game.is_game_over() and move_count < max_moves:
            possible_moves = ['left', 'right', 'up', 'down']
            random.shuffle(possible_moves)
            moved = False
            for move_dir in possible_moves:
                m, score_gain, _ = game.move(move_dir)
                if m:
                    print(f"\nMove {move_count+1}: {move_dir}, Score Gain: {score_gain}")
                    print(game.get_board())
                    print(f"Score: {game.get_score()}")
                    moved = True
                    move_count += 1
                    break # Go to next turn after successful move
            if not moved:
                # This should only happen if game becomes over unexpectedly
                print("No valid move found, but game not over? Error likely.")
                break

        print("\n--- Game Finished ---")
        print("Final Board:")
        print(game.get_board())
        print(f"Final Score: {game.get_score()}")
        print(f"Highest Tile: {game.get_highest_tile()}")
        if game.has_won():
            print("Result: Won!")
        elif game.is_game_over():
            print("Result: Game Over!")
        else:
            print("Result: Max moves reached.")