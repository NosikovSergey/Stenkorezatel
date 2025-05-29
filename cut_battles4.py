import cv2
import numpy as np
import os
import subprocess
from glob import glob

# === НАСТРОЙКИ ===
FRAME_STEP = 60
START_THRESHOLD = 0.9
END_THRESHOLD = 0.9
EXPECTED_WIDTH = 1920
EXPECTED_HEIGHT = 888

TEMPLATE_PATH = "start_frame.jpg"
END_TEMPLATE_NAMES = ["win", "lose", "chat_exit", "lobby_exit", "character_exit"]

# === ЗАГРУЗКА ВИДЕО ===
mp4_files = glob("*.mp4")
if len(mp4_files) != 1:
    raise Exception("Ошибка: В папке должен быть ровно один .mp4 файл")
VIDEO_PATH = mp4_files[0]

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

rotate_needed = False
if frame_height == EXPECTED_WIDTH and frame_width == EXPECTED_HEIGHT:
    print("[~] Обнаружен вертикальный формат. Будет произведён поворот кадров.")
    rotate_needed = True
elif frame_width != EXPECTED_WIDTH or frame_height != EXPECTED_HEIGHT:
    raise Exception(f"Ожидается видео {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}, но получено {frame_width}x{frame_height}")

# === ЗАГРУЗКА ШАБЛОНОВ ===
start_template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
if start_template is None:
    raise Exception(f"Не удалось загрузить шаблон: {TEMPLATE_PATH}")

end_templates = []
for name in END_TEMPLATE_NAMES:
    path = f"{name}_frame.jpg"
    tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        raise Exception(f"Не найден шаблон: {path}")
    end_templates.append((name, tmpl))

frame_idx = 0
timestamps = []
possible_start = None
start_confirmed = None
found_battle = False

print(f"[~] Анализ видео: {VIDEO_PATH}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if rotate_needed:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if frame_idx % FRAME_STEP == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if start_confirmed is None:
            score = np.sum(cv2.absdiff(gray, start_template) < 20) / gray.size
            print(f"[{frame_idx:05d}] Сходство с заставкой: {score:.3f}")
            if score >= START_THRESHOLD:
                possible_start = frame_idx / fps
            elif possible_start:
                start_confirmed = possible_start
                print(f"[+] Найдено начало боя: {start_confirmed:.2f} сек")
                possible_start = None
        else:
            for name, tmpl in end_templates:
                score = np.sum(cv2.absdiff(gray, tmpl) < 20) / gray.size
                print(f"[{frame_idx:05d}] Сравнение с {name}: {score:.3f}")
                if score >= END_THRESHOLD:
                    end_time = frame_idx / fps
                    print(f"[+] Найден конец боя ({name}): {end_time:.2f} сек")
                    timestamps.append((start_confirmed, end_time))
                    found_battle = True
                    break
            if found_battle:
                start_confirmed = None
                found_battle = False

    frame_idx += 1

cap.release()

if timestamps:
    os.makedirs("clips", exist_ok=True)
    for idx, (start, end) in enumerate(timestamps):
        output_file = f"clips/battle_{idx+1:02d}.mp4"
        cmd = f'ffmpeg -y -ss {start:.2f} -to {end:.2f} -i "{VIDEO_PATH}" -c:v libx264 -preset veryfast -crf 18 -c:a copy "{output_file}"'
        print(f"[~] Режем: {output_file}")
        subprocess.call(cmd, shell=True)
    print(f"\n[✓] Готово. Сохранено {len(timestamps)} фрагмент(ов).")
else:
    print("\n[!] Бой не найден.")
