import os
import json
import pygame

UI_STATE_PATH = os.path.expanduser("~/emoq_ui/ui_state.json")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SOFT = (220, 220, 220)
GREEN = (170, 255, 170)

def safe_read_json(path):
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}

def wrap_text(text, font, max_width):
    words = text.split()
    if not words:
        return []
    lines = []
    cur = words[0]
    for w in words[1:]:
        trial = cur + " " + w
        if font.size(trial)[0] <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines

def draw_box(screen, title, text, top, height, title_font, body_font):
    w, h = screen.get_size()
    x = int(w * 0.06)
    box_w = int(w * 0.88)

    pygame.draw.rect(screen, WHITE, (x, top, box_w, height), 2, border_radius=26)
    screen.blit(title_font.render(title, True, WHITE), (x + 18, top + 14))

    lines = wrap_text(text or "", body_font, box_w - 36)
    y = top + 58
    for line in lines[:5]:
        screen.blit(body_font.render(line, True, SOFT), (x + 18, y))
        y += 34

def draw_face(screen):
    w, h = screen.get_size()

    cx = w // 2
    eye_y = int(h * 0.36)
    eye_gap = int(w * 0.18)
    eye_r = int(min(w, h) * 0.075)

    left = (cx - eye_gap, eye_y)
    right = (cx + eye_gap, eye_y)

    pygame.draw.circle(screen, WHITE, left, eye_r)
    pygame.draw.circle(screen, WHITE, right, eye_r)

    pygame.draw.circle(screen, BLACK, left, max(8, eye_r // 3))
    pygame.draw.circle(screen, BLACK, right, max(8, eye_r // 3))

    mouth_y = int(h * 0.62)
    mouth_w = int(w * 0.18)
    pygame.draw.line(
        screen,
        WHITE,
        (cx - mouth_w // 2, mouth_y),
        (cx + mouth_w // 2, mouth_y),
        max(6, eye_r // 4)
    )

def main():
    pygame.init()

    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.NOFRAME)
    pygame.display.set_caption("EmoQ Screen")
    pygame.mouse.set_visible(False)

    title_font = pygame.font.SysFont("Arial", 28, True)
    body_font = pygame.font.SysFont("Arial", 28)
    small_font = pygame.font.SysFont("Arial", 20)

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state = safe_read_json(UI_STATE_PATH)

        screen.fill(BLACK)

        w, h = screen.get_size()
        mode = state.get("mode", "idle")
        user_text = state.get("user_text", "")
        bot_text = state.get("bot_text", "")
        prompt_text = state.get("prompt_text", "")

        screen.blit(small_font.render(mode.upper(), True, GREEN), (20, 20))

        if mode == "idle":
            draw_face(screen)
            draw_box(screen, "Standby", "Press button to talk", int(h * 0.72), int(h * 0.22), title_font, body_font)

        elif mode == "prompt":
            draw_box(screen, "EmoQ", prompt_text, int(h * 0.22), int(h * 0.56), title_font, body_font)

        else:
            draw_box(screen, "You", user_text, int(h * 0.16), int(h * 0.32), title_font, body_font)
            draw_box(screen, "EmoQ", bot_text, int(h * 0.54), int(h * 0.38), title_font, body_font)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
