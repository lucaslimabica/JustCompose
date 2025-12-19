from just_composing import Camera
from just_recording import Recorder
import pygame
import sys

WINDOW_SIZE = (720, 720)
BUTTON_SIZE = (250, 60)

BTN_COMPOSE_Y = 250
BTN_RECORD_Y = 350
BTN_DEBBUG_Y = 450

def draw_hover_glow(screen, rect):
    s = pygame.Surface(rect.size, pygame.SRCALPHA)
    s.fill((255, 255, 255, 40))
    screen.blit(s, rect.topleft)

def main():
    pygame.init()
    pygame.display.set_caption("JustCompose | Main Menu")
    pygame.mixer.init()
    pygame.mixer.music.load("assets/menu.mp3")
    pygame.mixer.music.set_volume(0.35)
    pygame.mixer.music.play(-1) # loop 
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    width, height = screen.get_size()

    bg = pygame.image.load("assets/menu_bg_v2.png").convert()
    bg = pygame.transform.smoothscale(bg, WINDOW_SIZE)

    btn_compose_rect = pygame.Rect(0, 0, *BUTTON_SIZE); btn_compose_rect.center = (width // 2, BTN_COMPOSE_Y)
    btn_debbug_rect = pygame.Rect(0, 0, *BUTTON_SIZE); btn_debbug_rect.center = (width // 2, BTN_DEBBUG_Y)
    btn_record_rect = pygame.Rect(0, 0, *BUTTON_SIZE); btn_record_rect.center = (width // 2, BTN_RECORD_Y)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        hover_compose = btn_compose_rect.collidepoint(mouse_pos)
        hover_debug = btn_debbug_rect.collidepoint(mouse_pos)
        hover_record = btn_record_rect.collidepoint(mouse_pos)

        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if (hover_compose or hover_debug or hover_record) else pygame.SYSTEM_CURSOR_ARROW)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if hover_compose:
                    cam = Camera(device=0)
                    pygame.mixer.music.stop()
                    pygame.display.quit()
                    cam.capture()
                    main()
                elif hover_debug:
                    cam = Camera(device=0, capture_mode="bounding_box")
                    pygame.mixer.music.stop()
                    pygame.display.quit()
                    cam.capture()
                    main()
                elif hover_record:
                    recorder = Recorder()
                    pygame.mixer.music.stop()
                    pygame.display.quit()
                    recorder.capture()
                    main()

        screen.blit(bg, (0, 0))

        if hover_compose: draw_hover_glow(screen, btn_compose_rect)
        if hover_debug: draw_hover_glow(screen, btn_debbug_rect)
        if hover_record: draw_hover_glow(screen, btn_record_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
