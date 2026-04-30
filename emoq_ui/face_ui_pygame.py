import pygame
import json
import os
import time

WIDTH = 800
HEIGHT = 480
JSON_PATH = os.path.expanduser("~/emoq_ui/emotion_state.json")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def normalize_emotion(raw):
    s = str(raw).strip().lower()
    if "happy" in s:
        return "happy"
    if "sad" in s:
        return "sad"
    if "think" in s:
        return "thinking"
    if "sleep" in s:
        return "sleepy"
    if "neutral" in s:
        return "neutral"
    return "neutral"

def read_emotion():
    if not os.path.exists(JSON_PATH):
        return None
    try:
        with open(JSON_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "emotion" in data:
                return normalize_emotion(data["emotion"])
            if "state" in data:
                return normalize_emotion(data["state"])
    except Exception as e:
        print("JSON read error:", e)
    return None

def draw_eye_open(screen, center, r=34):
    pygame.draw.circle(screen, WHITE, center, r)

def draw_eye_closed(screen, center, w=70):
    x, y = center
    pygame.draw.line(screen, WHITE, (x - w // 2, y), (x + w // 2, y), 8)

def draw_text(screen, text, font, y=430):
    surf = font.render(text, True, WHITE)
    rect = surf.get_rect(center=(WIDTH // 2, y))
    screen.blit(surf, rect)

def draw_happy(screen, font):
    screen.fill(BLACK)
    draw_eye_open(screen, (260, 170))
    draw_eye_open(screen, (540, 170))
    pygame.draw.arc(screen, WHITE, (280, 240, 240, 120), 0, 3.14159, 8)
    draw_text(screen, "HAPPY", font)

def draw_neutral(screen, font):
    screen.fill(BLACK)
    draw_eye_open(screen, (260, 170))
    draw_eye_open(screen, (540, 170))
    pygame.draw.line(screen, WHITE, (290, 305), (510, 305), 8)
    draw_text(screen, "NEUTRAL", font)

def draw_sad(screen, font):
    screen.fill(BLACK)
    draw_eye_open(screen, (260, 170))
    draw_eye_open(screen, (540, 170))
    pygame.draw.arc(screen, WHITE, (280, 300, 240, 120), 3.14159, 6.28318, 8)
    draw_text(screen, "SAD", font)

def draw_thinking(screen, font, frame_idx):
    screen.fill(BLACK)
    if (frame_idx // 20) % 2 == 0:
        draw_eye_open(screen, (260, 170))
        draw_eye_closed(screen, (540, 170))
    else:
        draw_eye_closed(screen, (260, 170))
        draw_eye_open(screen, (540, 170))
    pygame.draw.circle(screen, WHITE, (320, 300), 10)
    pygame.draw.circle(screen, WHITE, (400, 300), 10)
    pygame.draw.circle(screen, WHITE, (480, 300), 10)
    draw_text(screen, "THINKING", font)

def draw_sleepy(screen, font):
    screen.fill(BLACK)
    draw_eye_closed(screen, (260, 170), 75)
    draw_eye_closed(screen, (540, 170), 75)
    pygame.draw.arc(screen, WHITE, (330, 270, 140, 70), 0, 3.14159, 8)
    draw_text(screen, "SLEEPY", font)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("EmoQ Face")
    font = pygame.font.SysFont("Arial", 32)
    clock = pygame.time.Clock()

    current_state = "neutral"
    auto_mode = True
    last_read = 0
    frame_idx = 0
    running = True

    print("Controls: h n s t y a q")

    while running:
        now = time.time()
        if auto_mode and now - last_read > 0.5:
            last_read = now
            emo = read_emotion()
            if emo:
                current_state = emo

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_h:
                    current_state = "happy"
                    auto_mode = False
                elif event.key == pygame.K_n:
                    current_state = "neutral"
                    auto_mode = False
                elif event.key == pygame.K_s:
                    current_state = "sad"
                    auto_mode = False
                elif event.key == pygame.K_t:
                    current_state = "thinking"
                    auto_mode = False
                elif event.key == pygame.K_y:
                    current_state = "sleepy"
                    auto_mode = False
                elif event.key == pygame.K_a:
                    auto_mode = not auto_mode

        if current_state == "happy":
            draw_happy(screen, font)
        elif current_state == "sad":
            draw_sad(screen, font)
        elif current_state == "thinking":
            draw_thinking(screen, font, frame_idx)
        elif current_state == "sleepy":
            draw_sleepy(screen, font)
        else:
            draw_neutral(screen, font)

        pygame.display.flip()
        frame_idx += 1
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
