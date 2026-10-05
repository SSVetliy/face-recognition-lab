"""
Этап 1. Проверка камеры.

Что делает скрипт:
  1. Открывает камеру с индексом CAMERA_INDEX из config.py.
  2. Задаёт разрешение.
  3. В цикле читает кадры и показывает окно.
  4. Считает реальный FPS и рисует его на кадре.
  5. Выходит по ESC.

Запуск (из корня проекта, с активным .venv):
  python scripts/test_camera.py
"""

import sys
import time
from pathlib import Path

import cv2

# Добавляем корень проекта в sys.path, чтобы можно было импортировать src.*
# Это нужно, потому что скрипт лежит в scripts/, а модуль config — в src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402


def main() -> None:
    # Открываем камеру. cv2.CAP_DSHOW — Windows-специфичный бэкенд,
    # он быстрее открывает камеру и меньше тормозит.
    # На Linux/macOS этот флаг игнорируется — OpenCV подберёт бэкенд сам.
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть камеру с индексом {config.CAMERA_INDEX}.")
        print("Проверь, что камера не занята другим приложением (Zoom, Teams и т.д.).")
        sys.exit(1)

    # Задаём разрешение. Камера может округлить до ближайшего поддерживаемого.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Камера открыта. Реальное разрешение: {real_w}x{real_h}")
    print("[INFO] Нажми ESC в окне, чтобы выйти.")

    # Переменные для расчёта FPS.
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Не удалось прочитать кадр. Пропускаем.")
                continue

            # Считаем FPS с экспоненциальным сглаживанием,
            # чтобы цифра не прыгала на каждом кадре.
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                instant_fps = 1.0 / dt
                # Сглаживание: 0.9 — старая оценка, 0.1 — новая.
                fps = 0.9 * fps + 0.1 * instant_fps if fps > 0 else instant_fps

            # Рисуем FPS в левом верхнем углу.
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(config.WINDOW_NAME, frame)

            # waitKey(1) — обрабатываем события окна и ждём 1 мс.
            # ESC (27) — выход.
            key = cv2.waitKey(1) & 0xFF
            if key == config.EXIT_KEY:
                print("[INFO] Выход по ESC.")
                break

    finally:
        # Освобождаем ресурсы в любом случае — даже если была ошибка.
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Ресурсы освобождены.")


if __name__ == "__main__":
    main()