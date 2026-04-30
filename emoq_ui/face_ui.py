import cv2
import numpy as np
import json
import os
import time

WIDTH = 800
HEIGHT = 480
JSON_PATH = os.path.expanduser("~/emoq_ui/emotion_state.json")

BG_COLOR = (0, 0, 0)
FG_COLOR = (255, 255, 255)

def blank():
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

def draw_text(img, text, y=430, scale=0.9, thickness=2):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    x = (WIDTH - text_size[0]) // 2
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale, FG_COLOR, thickness, cv2.LINE_AA
    )

def draw_eye_open(img, center, r=36):
    cv2.circle(img, center, r, FG_COLOR, -1)

def draw_eye_closed(img, center, w=70):
    x, y = center
    cv2.line(img, (x - w // 2, y), (x + w // 2, y), FG_COLOR, 8)

def make_happy():
    img = blank()
    draw_eye_open(img, (260, 170), 34)
    draw_eye_open(img, (540, 170), 34)
    cv2.ellipse(img, (400, 300), (120, 55), 0, 0, 180, FG_COLOR, 8)
    draw_text(img, "HAPPY")
    return img

def make_neutral():
    img = blank()
    draw_eye_open(img, (260, 170), 34)
    draw_eye_open(img, (540, 170), 34)
    cv2.line(img, (290, 305), (510, 305), FG_COLOR, 8)
    draw_text(img, "NEUTRAL")
    return img

def make_sad():
    img = blank()
    draw_eye_open(img, (260, 170), 34)
    draw_eye_open(img, (540, 170), 34)
    cv2.ellipse(img, (400, 360), (120, 55), 0, 180, 360, FG_COLOR, 8)
    draw_text(img, "SAD")
    return img

def make_thinking(frame_idx=0):
    img = blank()
    if (frame_idx // 15) % 2 == 0:
        draw_eye_open(img, (260, 170), 34)
        draw_eye_closed(img, (540, 170), 70)
    else:
        draw_eye_closed(img, (260, 170), 70)
        draw_eye_open(img, (540, 170), 34)
    cv2.circle(img, (320, 300), 10, FG_COLOR, -1)
    cv2.circle(img, (400, 300), 10, FG_COLOR, -1)
    cv2.circle(img, (480, 300), 10, FG_COLOR, -1)
    draw_text(img, "THINKING")
    return img

def make_sleepy():
    img = blank()
    draw_eye_closed(img, (260, 170), 75)
    draw_eye_closed(img, (540, 170), 75)
    cv2.ellipse(img, (400, 305), (70, 28), 0, 0, 180, FG_COLOR, 8)
    draw_text(img, "SLEEPY")
    return img

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

def read_emotion_from_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "emotion" in data:
                return normalize_emotion(data["emotion"])
            if "state" in data:
                return normalize_emotion(data["state"])
        return None
    except Exception as e:
        print("JSON read error:", e)
        return None

def render_face(state, frame_idx):
    if state == "happy":
        return make_happy()
    if state == "sad":
        return make_sad()
    if state == "thinking":
        return make_thinking(frame_idx)
    if state == "sleepy":
        return make_sleepy()
    return make_neutral()

def main():
    cv2.namedWindow("EmoQ Face", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("EmoQ Face", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    current_state = "neutral"
    frame_idx = 0
    last_json_read = 0.0
    auto_mode = True

    print("Controls:")
    print("  h = happy")
    print("  n = neutral")
    print("  s = sad")
    print("  t = thinking")
    print("  y = sleepy")
    print("  a = toggle auto JSON mode")
    print("  q = quit")
    print(f"Auto JSON path: {JSON_PATH}")

    while True:
        now = time.time()
        if auto_mode and now - last_json_read > 0.5:
            last_json_read = now
            json_state = read_emotion_from_json(JSON_PATH)
            if json_state is not None:
                current_state = json_state

        img = render_face(current_state, frame_idx)
        cv2.imshow("EmoQ Face", img)

        key = cv2.waitKey(33) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("h"):
            current_state = "happy"
            auto_mode = False
        elif key == ord("n"):
            current_state = "neutral"
            auto_mode = False
        elif key == ord("s"):
            current_state = "sad"
            auto_mode = False
        elif key == ord("t"):
            current_state = "thinking"
            auto_mode = False
        elif key == ord("y"):
            current_state = "sleepy"
            auto_mode = False
        elif key == ord("a"):
            auto_mode = not auto_mode
            print("auto_mode =", auto_mode)

        frame_idx += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

