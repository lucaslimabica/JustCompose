from just_composing import Camera
from just_recording import Recorder 
import pygame
import sys

# -- | Constants | --

WINDOW_SIZE = (720, 720)
BG_COLOR = (60, 25, 60)

COLOR_WHITE = (255, 255, 255)
COLOR_BUTTON_LIGHT = (170, 170, 170)
COLOR_BUTTON_DARK = (100, 100, 100)

BUTTON_SIZE = (250, 60)

BTN_COMPOSE_Y = 250
BTN_RECORD_Y = 350 


def draw_button(screen, rect, font, text_content, mouse_pos):
    if rect.collidepoint(mouse_pos):
        color = COLOR_BUTTON_LIGHT
    else:
        color = COLOR_BUTTON_DARK
    
    pygame.draw.rect(screen, color, rect)
    
    text_surface = font.render(text_content, True, COLOR_WHITE)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)
    
    return rect

def main():
    """
    Main entry point for the JustCompose UI.
    Opens a simple Pygame window with two buttons.
    """
    pygame.init()
    pygame.display.set_caption("JustCompose | Main Menu")

    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    width, height = screen.get_size()
    
    font = pygame.font.SysFont("Corbel", 32)
    
    # Start Composing
    btn_compose_rect = pygame.Rect(0, 0, *BUTTON_SIZE)
    btn_compose_rect.center = (width // 2, BTN_COMPOSE_Y)
    
    # Start Recording
    btn_record_rect = pygame.Rect(0, 0, *BUTTON_SIZE)
    btn_record_rect.center = (width // 2, BTN_RECORD_Y)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            # Mouse click
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                
                if btn_compose_rect.collidepoint(mouse_pos):
                    cam = Camera(device=0)
                    pygame.display.quit()
                    cam.capture()
                    main()
                    
                if btn_record_rect.collidepoint(mouse_pos):
                    recorder = Recorder()
                    pygame.display.quit()
                    recorder.capture()
                    main()
                    
                    
        
        screen.fill(BG_COLOR)
        draw_button(screen, btn_compose_rect, font, "Start Composing", mouse_pos)
        draw_button(screen, btn_record_rect, font, "Start Recording", mouse_pos)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()