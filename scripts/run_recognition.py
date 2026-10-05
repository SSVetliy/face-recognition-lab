"""
Этап 6. Распознавание в реальном времени + логирование визитов.

Что делает:
  1. Загружает базу эталонов из data/embeddings/base.npz.
  2. Открывает камеру.
  3. На каждом кадре:
     - находит все лица,
     - строит эмбеддинги,
     - идентифицирует через matcher,
     - рисует зелёную (свой) или красную (UNKNOWN) рамку с именем и dist.
  4. Логирует первое появление каждого человека в data/logs/visits.csv
     (с антиспамом: повторно — не раньше чем через VISIT_COOLDOWN_SECONDS).
  5. Выход по ESC.

Запуск:
  python scripts/run_recognition.py
"""

import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                      # noqa: E402
from src.embedder import FaceEmbedder       # noqa: E402
from src.storage import load_base           # noqa: E402
from src.matcher import FaceMatcher         # noqa: E402
from src.visits import VisitLogger          # noqa: E402


def draw_result(frame, face, match) -> None:
    """
    Нарисовать рамку, имя, distance и статус.

    face  — FaceEmbedding (bbox, landmarks, det_score).
    match — MatchResult (name, distance, is_known, top_matches).
    """
    x1, y1, x2, y2 = face.bbox

    # Цвет рамки зависит от статуса.
    color = config.BOX_COLOR if match.is_known else config.UNKNOWN_BOX_COLOR

    # Рамка.
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)

    # Ключевые точки.
    for (px, py) in face.landmarks:
        cv2.circle(
            frame, (int(px), int(py)),
            config.LANDMARK_RADIUS, config.LANDMARK_COLOR, -1,
        )

    # Подпись: имя + dist.
    name = match.name if match.is_known else "UNKNOWN"
    label = f"{name} ({match.distance:.2f})"

    # Фон под текстом — для читаемости на любом фоне.
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX,
        config.TEXT_SCALE, config.TEXT_THICKNESS,
    )
    cv2.rectangle(
        frame,
        (x1, max(y1 - th - 10, 0)),
        (x1 + tw + 10, max(y1, th + 10)),
        color,
        -1,
    )
    # Текст белым — контрастно на зелёном/красном.
    cv2.putText(
        frame, label,
        (x1 + 5, max(y1 - 5, th + 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.TEXT_SCALE,
        (255, 255, 255),
        config.TEXT_THICKNESS,
        cv2.LINE_AA,
    )


def main() -> None:
    # ─── 1. База ──────────────────────────────────────────────
    print("[INFO] Загружаю базу эталонов...")
    records = load_base()
    if not records:
        print("[ERROR] База пустая. Сначала: python scripts/build_base.py")
        sys.exit(1)
    print(f"[INFO] Людей в базе: {len(records)}")
    for r in records:
        print(f"       - {r.name}: {r.n_photos} фото")

    # ─── 2. Модель ────────────────────────────────────────────
    print("\n[INFO] Загружаю модель...")
    embedder = FaceEmbedder()
    matcher = FaceMatcher(records)
    logger = VisitLogger()
    print(f"[INFO] Порог идентификации: {matcher.threshold:.3f}")
    print(f"[INFO] Лог визитов: {logger._path}")
    print("[INFO] Модель готова.\n")

    # ─── 3. Камера ────────────────────────────────────────────
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть камеру {config.CAMERA_INDEX}.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    print("[INFO] Камера открыта. Нажми ESC для выхода.\n")

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # Детекция + эмбеддинги.
            faces = embedder.embed_frame(frame)

            # Идентификация каждого лица.
            for face in faces:
                match = matcher.match(face.embedding)
                draw_result(frame, face, match)

                # Логирование (только если узнали — UNKNOWN не пишем).
                if match.is_known:
                    logged = logger.log(
                        name=match.name,
                        distance=match.distance,
                        is_known=True,
                    )
                    if logged:
                        print(
                            f"[VISIT] {match.name} "
                            f"(dist={match.distance:.3f})"
                        )

            # FPS.
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                instant = 1.0 / dt
                fps = 0.9 * fps + 0.1 * instant if fps > 0 else instant

            # Инфо-панель.
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Faces: {len(faces)} | Threshold: {matcher.threshold:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
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