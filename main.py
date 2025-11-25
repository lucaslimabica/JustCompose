from cam import Camera
import pygame
import sys

# -- | Constants | --

WINDOW_SIZE = (720, 720)
BG_COLOR = (60, 25, 60)

COLOR_WHITE = (255, 255, 255)
COLOR_BUTTON_LIGHT = (170, 170, 170)
COLOR_BUTTON_DARK = (100, 100, 100)

BUTTON_SIZE = (250, 60)


def main():
    """
    Main entry point for the JustCompose UI.
    Opens a simple Pygame window with a single button.
    """
    pygame.init()
    pygame.display.set_caption("JustCompose | Main Menu")

    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    width, height = screen.get_size()
    # Background
    screen.fill(BG_COLOR)
    
    # Button setup
    button_rect = pygame.Rect(0, 0, *BUTTON_SIZE)
    button_rect.center = (width // 2, height // 2)
    font = pygame.font.SysFont("Corbel", 32)
    button_text = font.render("Start Composing", True, COLOR_WHITE)
    # Center text inside the button
    text_rect = button_text.get_rect(center=button_rect.center)
    screen.blit(button_text, text_rect)
    # Button hover effect
    if button_rect.collidepoint(mouse_pos):
        pygame.draw.rect(screen, COLOR_BUTTON_LIGHT, button_rect)
    else:
        pygame.draw.rect(screen, COLOR_BUTTON_DARK, button_rect)

    # Prepare camera instance
    cam = Camera(device=0)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            # Window close (X)
            if event.type == pygame.QUIT:
                running = False

            # ESC key quits
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            # Mouse click
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if button_rect.collidepoint(mouse_pos):
                    # Start motion capture (blocks until camera window closes)
                    cam.capture()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
