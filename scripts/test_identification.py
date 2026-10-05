"""
Этап 5, шаг 2. Тест идентификации 1:N.

Что делает:
  1. Загружает базу из data/embeddings/base.npz.
  2. Берёт все одиночные фото из КОРНЯ data/photos/ (не из подпапок) —
     это тестовый набор. У тебя там групповые md/sd/sv.
  3. Для каждого лица на каждом фото делает идентификацию.
  4. Выводит таблицу: имя-предсказание, расстояние, известный/неизвестный.

Запуск:
  python scripts/test_identification.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                              # noqa: E402
from src.embedder import FaceEmbedder               # noqa: E402
from src.storage import load_base                   # noqa: E402
from src.matcher import FaceMatcher                 # noqa: E402


def list_test_photos() -> list[Path]:
    """Фото из корня data/photos/, БЕЗ подпапок — это тестовый набор."""
    return sorted(
        p for p in config.PHOTOS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_IMAGE_EXTENSIONS
    )


def main() -> None:
    print("[INFO] Загружаю базу эталонов...")
    records = load_base()
    if not records:
        print("[ERROR] База пустая. Сначала запусти: python scripts/build_base.py")
        return

    print(f"[INFO] Людей в базе: {len(records)}")
    for r in records:
        print(f"       - {r.name}: {r.n_photos} фото")
    print()

    print("[INFO] Загружаю модель...")
    embedder = FaceEmbedder()
    print("[INFO] Модель загружена.\n")

    matcher = FaceMatcher(records)
    print(f"[INFO] Порог идентификации: {matcher.threshold:.3f}\n")

    test_photos = list_test_photos()
    if not test_photos:
        print(f"[WARN] В корне {config.PHOTOS_DIR} нет фото для теста.")
        return

    print(f"[INFO] Тестовых фото в корне: {len(test_photos)}\n")
    print("=" * 78)
    print(f"{'ФАЙЛ':<18} {'ЛИЦО':<6} {'PREDICTED':<12} {'DIST':<8} {'TOP-3 (dist)':<30}")
    print("=" * 78)

    for photo in test_photos:
        try:
            results = embedder.embed_image(photo)
        except Exception as e:
            print(f"[WARN] {photo.name}: ошибка — {e}")
            continue

        if not results:
            print(f"{photo.name:<18} лиц не найдено")
            continue

        for i, r in enumerate(results):
            m = matcher.match(r.embedding)
            predicted = m.name if m.is_known else "UNKNOWN"
            top3 = ", ".join(f"{n}:{d:.2f}" for n, d in m.top_matches)
            print(
                f"{photo.name:<18} #{i:<5} {predicted:<12} "
                f"{m.distance:<8.3f} {top3:<30}"
            )

    print("=" * 78)
    print("\n[INFO] Готово.")


if __name__ == "__main__":
    main()