import pygame
import numpy as np
import math
import time

# --- Constants for Interface ---
TILE_SIZE = 100
MARGIN = 10
SCORE_HEIGHT = 80
BG_COLOR = (250, 248, 239)
GRID_COLOR = (187, 173, 160)
EMPTY_CELL_COLOR = (205, 193, 180)
TEXT_COLOR_LIGHT = (249, 246, 242)
TEXT_COLOR_DARK = (119, 110, 101)
TILE_COLORS = {
    0: EMPTY_CELL_COLOR, 2: (238, 228, 218), 4: (237, 224, 200),
    8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95),
    64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
    512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46),
    4096: (60, 58, 50), 8192: (60, 58, 50), 16384: (60, 58, 50),
    32768: (60, 58, 50), 65536: (60, 58, 50)
}
ANIMATION_DURATION_MS = 100  # Duration for move/merge animation
NEW_TILE_ANIMATION_DURATION_MS = 150  # Duration for new tile appearance
FPS = 60

class PygameInterface:
    def __init__(self, game_logic):
        self.game = game_logic
        self.board_size = game_logic.size

        # Calculate dimensions based on board size
        self.grid_width = self.board_size * TILE_SIZE + (self.board_size + 1) * MARGIN
        self.grid_height = self.board_size * TILE_SIZE + (self.board_size + 1) * MARGIN
        self.window_width = self.grid_width
        self.window_height = self.grid_height + SCORE_HEIGHT

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("2048 Simplified")
        self.clock = pygame.time.Clock()

        self.font_score = pygame.font.SysFont("Arial", 40, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 60, bold=True)

        self.game.add_listener(self)  # Register for callbacks

        self._animating_move = False
        self._animation_start_time = 0
        self._animation_steps = []
        self._animating_new_tile = False
        self._new_tile_info = None  # (r, c, value)
        self._new_tile_start_time = 0

        self._show_win_overlay = False
        self._win_continue = False  # Has the player chosen to continue after winning?

    # --- Game Logic Listener Methods ---
    def on_reset(self, board, score):
        self._animating_move = False
        self._animating_new_tile = False
        self._show_win_overlay = False
        self._win_continue = False
        # No visual feedback needed here, draw loop handles board state

    def on_move_complete(self, animation_steps):
        if animation_steps:
            self._animation_steps = animation_steps
            self._animating_move = True
            self._animation_start_time = pygame.time.get_ticks()
            # Clear the new tile info until animation completes
            self._new_tile_info = None
            self._animating_new_tile = False
        else:
            # Ensure no lingering animation state if move was invalid but triggered checks
            self._animating_move = False
            self._animation_steps = []

    def on_tile_spawned(self, r, c, value):
        # Trigger new tile animation *after* move animation finishes
        self._new_tile_info = (r, c, value)
        # Start time will be set when move animation ends

    def on_game_over(self):
        # The draw loop will check game.is_game_over()
        pass

    def on_win(self):
        if not self._win_continue:
             self._show_win_overlay = True
        # The draw loop will check game.has_won() and _show_win_overlay

    # --- Pygame Drawing Helpers ---
    def _get_tile_color(self, value):
        log_value = int(math.log2(value)) if value > 0 else 0
        color_key = 2**min(log_value, 17)
        return TILE_COLORS.get(color_key, TILE_COLORS[65536])

    def _get_text_color(self, value):
        return TEXT_COLOR_DARK if value <= 8 else TEXT_COLOR_LIGHT

    def _get_tile_font_size(self, value):
        s = str(value)
        if len(s) <= 2: return 55
        if len(s) == 3: return 45
        if len(s) == 4: return 35
        if len(s) == 5: return 30
        return 25

    def _get_cell_pixel_center(self, r, c):
         x = MARGIN + c * (TILE_SIZE + MARGIN) + TILE_SIZE // 2
         y = SCORE_HEIGHT + MARGIN + r * (TILE_SIZE + MARGIN) + TILE_SIZE // 2
         return x, y

    def _draw_tile_at_pos(self, value, center_pixel, alpha=255, scale=1.0):
        color = self._get_tile_color(value)
        base_rect = pygame.Rect(0, 0, TILE_SIZE, TILE_SIZE)
        scaled_size = max(1, int(TILE_SIZE * scale))
        scaled_rect = pygame.Rect(0, 0, scaled_size, scaled_size)
        scaled_rect.center = center_pixel

        tile_surface = pygame.Surface((scaled_size, scaled_size), pygame.SRCALPHA)
        pygame.draw.rect(tile_surface, (*color, alpha), (0, 0, scaled_size, scaled_size), border_radius=int(5 * scale))
        self.screen.blit(tile_surface, scaled_rect.topleft)

        font_size = self._get_tile_font_size(value)
        display_font_size = max(10, int(font_size * min(1.0, scale * 1.5)))
        font = pygame.font.SysFont("Arial", display_font_size, bold=True)
        text_surf = font.render(str(value), True, self._get_text_color(value))
        text_surf.set_alpha(alpha)
        text_rect = text_surf.get_rect(center=scaled_rect.center)
        self.screen.blit(text_surf, text_rect)

    # --- Main Drawing Method ---
    def _draw_board(self):
        self.screen.fill(BG_COLOR)
        grid_rect = (0, SCORE_HEIGHT, self.window_width, self.grid_height)
        pygame.draw.rect(self.screen, GRID_COLOR, grid_rect)

        # Draw empty cells
        for r in range(self.board_size):
            for c in range(self.board_size):
                cell_rect = pygame.Rect(
                    MARGIN + c * (TILE_SIZE + MARGIN),
                    SCORE_HEIGHT + MARGIN + r * (TILE_SIZE + MARGIN),
                    TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(self.screen, EMPTY_CELL_COLOR, cell_rect, border_radius=5)

        current_time = pygame.time.get_ticks()
        move_anim_progress = 0.0
        new_tile_anim_progress = 0.0

        # Calculate animation progress
        if self._animating_move:
            elapsed = current_time - self._animation_start_time
            move_anim_progress = min(1.0, elapsed / ANIMATION_DURATION_MS)
            if move_anim_progress >= 1.0:
                self._animating_move = False
                self._animation_steps = []
                if not self.game.is_game_over():
                    # Call the adapter's method to generate new tiles
                    self.game._add_random_tile()

                # If a new tile was spawned, start its animation now
                if self._new_tile_info:
                    self._animating_new_tile = True
                    self._new_tile_start_time = current_time

        if self._animating_new_tile:
             elapsed = current_time - self._new_tile_start_time
             new_tile_anim_progress = min(1.0, elapsed / NEW_TILE_ANIMATION_DURATION_MS)
             if new_tile_anim_progress >= 1.0:
                  self._animating_new_tile = False
                  self._new_tile_info = None  # Clear after animation

        # --- Draw Tiles ---
        board_state = self.game.get_board()  # Get current state for drawing static tiles

        # Track positions involved in animations
        animated_positions = set()

        # Gather positions involved in move/merge animation
        if self._animating_move:
            # Collect all positions that are part of animations
            for step in self._animation_steps:
                if 'start' in step:
                    animated_positions.add(step['start'])
                if 'end' in step:
                    animated_positions.add(step['end'])

            # Draw the animations
            for step in self._animation_steps:
                value = step['value']
                start_pos = step.get('start')
                end_pos = step['end']
                tile_type = step['type']

                if tile_type in ['move', 'merge_source']:
                    start_pixel = self._get_cell_pixel_center(*start_pos)
                    end_pixel = self._get_cell_pixel_center(*end_pos)
                    current_x = start_pixel[0] + (end_pixel[0] - start_pixel[0]) * move_anim_progress
                    current_y = start_pixel[1] + (end_pixel[1] - start_pixel[1]) * move_anim_progress
                    alpha = 255
                    if tile_type == 'merge_source':
                        alpha = max(0, 255 - int(255 * move_anim_progress * 2))  # Fade out faster
                    self._draw_tile_at_pos(value, (current_x, current_y), alpha=alpha)

                elif tile_type == 'merge_result':
                    # Draw merge results appearing towards the end
                    if move_anim_progress > 0.5:
                        appear_progress = min(1.0, (move_anim_progress - 0.5) / 0.5)
                        end_pixel = self._get_cell_pixel_center(*end_pos)
                        scale = 0.5 + 0.5 * appear_progress
                        alpha = int(255 * appear_progress)
                        self._draw_tile_at_pos(value, end_pixel, alpha=alpha, scale=scale)

        # Draw static tiles (always draw, but skip those being animated)
        for r in range(self.board_size):
            for c in range(self.board_size):
                value = board_state[r, c]
                pos = (r, c)

                # Skip empty cells and cells involved in animations
                if value == 0 or (self._animating_move and pos in animated_positions):
                    continue

                pixel_center = self._get_cell_pixel_center(r, c)
                scale = 1.0
                alpha = 255

                # Animate newly spawned tile
                if self._animating_new_tile and self._new_tile_info[:2] == (r, c):
                    scale = 0.1 + 0.9 * new_tile_anim_progress  # Scale from 0.1 to 1
                    alpha = int(255 * new_tile_anim_progress)  # Fade in

                self._draw_tile_at_pos(value, pixel_center, alpha=alpha, scale=scale)

        # --- Draw Score ---
        score_text = self.font_score.render(f"Score: {self.game.get_score()}", True, TEXT_COLOR_DARK)
        score_rect = score_text.get_rect(center=(self.window_width // 2, SCORE_HEIGHT // 2))
        self.screen.blit(score_text, score_rect)

    def _draw_game_over(self):
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        overlay.fill((238, 228, 218, 180))
        self.screen.blit(overlay, (0, 0))
        msg = self.font_large.render("Game Over!", True, TEXT_COLOR_DARK)
        score_msg = self.font_large.render(f"Final Score: {self.game.get_score()}", True, TEXT_COLOR_DARK)
        msg_rect = msg.get_rect(center=(self.window_width // 2, self.window_height // 2 - 50))
        score_rect = score_msg.get_rect(center=(self.window_width // 2, self.window_height // 2 + 20))
        self.screen.blit(msg, msg_rect)
        self.screen.blit(score_msg, score_rect)
        restart_msg = pygame.font.SysFont("Arial", 30, bold=True).render("Press R to Restart", True, TEXT_COLOR_DARK)
        restart_rect = restart_msg.get_rect(center=(self.window_width // 2, self.window_height - 50))
        self.screen.blit(restart_msg, restart_rect)

    def _draw_win_message(self):
         overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
         overlay.fill((237, 194, 46, 180))
         self.screen.blit(overlay, (0, 0))
         msg = self.font_large.render("You Win!", True, TEXT_COLOR_LIGHT)
         score_msg = self.font_large.render(f"Score: {self.game.get_score()}", True, TEXT_COLOR_LIGHT)
         msg_rect = msg.get_rect(center=(self.window_width // 2, self.window_height // 2 - 50))
         score_rect = score_msg.get_rect(center=(self.window_width // 2, self.window_height // 2 + 20))
         self.screen.blit(msg, msg_rect)
         self.screen.blit(score_msg, score_rect)
         continue_msg = pygame.font.SysFont("Arial", 30, bold=True).render("Press any key to continue or R to Restart", True, TEXT_COLOR_DARK)
         continue_rect = continue_msg.get_rect(center=(self.window_width // 2, self.window_height - 50))
         self.screen.blit(continue_msg, continue_rect)

    # --- Main Loop ---
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    # Handle input regardless of animation state, but game logic will block if busy
                    current_game_over = self.game.is_game_over()
                    current_game_won = self.game.has_won()

                    if current_game_over:
                        if event.key == pygame.K_r:
                            self.game.reset()
                    elif current_game_won and self._show_win_overlay:
                         if event.key == pygame.K_r:
                              self.game.reset()  # Reset will hide overlay via callback
                         else:
                              # Continue playing
                              self._show_win_overlay = False
                              self._win_continue = True
                    elif not self._animating_move and not self._animating_new_tile:  # Only accept move input if not animating
                        direction = None
                        if event.key == pygame.K_LEFT: direction = 'left'
                        elif event.key == pygame.K_RIGHT: direction = 'right'
                        elif event.key == pygame.K_UP: direction = 'up'
                        elif event.key == pygame.K_DOWN: direction = 'down'

                        if direction:
                            self.game.move(direction)  # Logic handles state update and notifies interface

            # Drawing
            self._draw_board()

            if self.game.is_game_over():
                self._draw_game_over()
            elif self.game.has_won() and self._show_win_overlay:
                 self._draw_win_message()

            pygame.display.flip()

        pygame.quit()

# --- End of Pygame Interface ---