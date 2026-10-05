"""
Этап 2. Детекция лиц в реальном времени.

Что делает:
  1. Открывает камеру.
  2. На каждом кадре запускает детектор лиц.
  3. Рисует рамки, ключевые точки и confidence.
  4. Показывает FPS и количество найденных лиц.
  5. Выход по ESC.

Первый запуск скачает модель (~300 МБ) в ~/.insightface/models/buffalo_sc/.
Это одноразово — при следующих запусках модель берётся с диска.

Запуск (из корня проекта, с активным .venv):
  python scripts/test_detection.py
"""

import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                  # noqa: E402
from src.detector import FaceDetector   # noqa: E402


def draw_face(frame, face) -> None:
    """Нарисовать рамку, ключевые точки и score одного лица."""
    x1, y1, x2, y2 = face.bbox

    # Рамка лица.
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        config.BOX_COLOR,
        config.BOX_THICKNESS,
    )

    # 5 ключевых точек: глаза, нос, уголки рта.
    for (px, py) in face.landmarks:
        cv2.circle(
            frame,
            (int(px), int(py)),
            config.LANDMARK_RADIUS,
            config.LANDMARK_COLOR,
            -1,
        )

    # Подпись с уверенностью над рамкой.
    label = f"{face.det_score:.2f}"
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 8, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.TEXT_SCALE,
        config.TEXT_COLOR,
        config.TEXT_THICKNESS,
        cv2.LINE_AA,
    )


def main() -> None:
    print("[INFO] Загружаю модель детекции (первый раз — скачивание ~300 МБ)...")
    detector = FaceDetector()
    print("[INFO] Модель загружена.")

    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть камеру {config.CAMERA_INDEX}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    print("[INFO] Камера открыта. Нажми ESC для выхода.")

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # Детекция лиц.
            faces = detector.detect(frame)

            # Отрисовка.
            for face in faces:
                draw_face(frame, face)

            # FPS с экспоненциальным сглаживанием.
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                instant = 1.0 / dt
                fps = 0.9 * fps + 0.1 * instant if fps > 0 else instant

            # Инфо-панель сверху.
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Faces: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == config.EXIT_KEY:
                print("[INFO] Выход по ESC.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Ресурсы освобождены.")


if __name__ == "__main__":
    main()