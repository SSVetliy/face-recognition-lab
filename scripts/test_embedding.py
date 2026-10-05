"""
Этап 3. Проверка эмбеддингов на фото.

Что делает:
  1. Читает все картинки из data/photos/.
  2. Для каждой строит эмбеддинги (все лица на фото).
  3. Выводит в консоль:
     - сколько лиц нашлось на каждом фото,
     - размер эмбеддинга (должен быть 512),
     - матрицу косинусных расстояний между всеми эмбеддингами.
  4. По матрице видно:
     - маленькое расстояние (~0.2-0.4) — один и тот же человек,
     - большое расстояние (> 0.6) — разные люди.

Запуск (из корня проекта, с активным .venv):
  python scripts/test_embedding.py
"""

import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                          # noqa: E402
from src.embedder import FaceEmbedder, cosine_distance          # noqa: E402


def find_photos() -> list[Path]:
    """Собрать все картинки из data/photos/, отсортированные по имени."""
    photos = sorted(
        p for p in config.PHOTOS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_IMAGE_EXTENSIONS
    )
    return photos


def main() -> None:
    print("[INFO] Загружаю модель (детектор + распознаватель)...")
    embedder = FaceEmbedder()
    print("[INFO] Модель загружена.\n")

    photos = find_photos()
    if not photos:
        print(f"[ERROR] В {config.PHOTOS_DIR} нет картинок.")
        print("Положи туда фото и перезапусти.")
        return

    print(f"[INFO] Найдено фото: {len(photos)}")
    for p in photos:
        print(f"       - {p.name}")
    print()

    # ─────────────────────────────────────────────────────────
    # Шаг 1. Собираем эмбеддинги всех лиц со всех фото.
    # ─────────────────────────────────────────────────────────
    # Каждый элемент: (имя_файла, индекс_лица_на_фото, эмбеддинг, det_score)
    all_embeddings: list[tuple[str, int, "np.ndarray", float]] = []

    for photo in photos:
        try:
            results = embedder.embed_image(photo)
        except Exception as e:
            print(f"[WARN] {photo.name}: не удалось обработать — {e}")
            continue

        if not results:
            print(f"[WARN] {photo.name}: лица не найдены")
            continue

        print(f"[INFO] {photo.name}: лиц найдено — {len(results)}")
        for i, r in enumerate(results):
            print(
                f"        лицо #{i}: det_score={r.det_score:.3f}, "
                f"размер эмбеддинга={r.embedding.shape[0]}"
            )
            # Проверка длины эмбеддинга.
            if r.embedding.shape[0] != config.EXPECTED_EMBEDDING_DIM:
                print(
                    f"[WARN]  Ожидался эмбеддинг длины "
                    f"{config.EXPECTED_EMBEDDING_DIM}, а получен "
                    f"{r.embedding.shape[0]}"
                )
            all_embeddings.append((photo.name, i, r.embedding, r.det_score))

    if len(all_embeddings) < 2:
        print("\n[WARN] Меньше двух эмбеддингов — нечего сравнивать.")
        return

    # ─────────────────────────────────────────────────────────
    # Шаг 2. Считаем косинусные расстояния между всеми парами.
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("КОСИНУСНЫЕ РАССТОЯНИЯ МЕЖДУ ПАРАМИ ЛИЦ")
    print("=" * 70)
    print("Ориентиры:")
    print(f"  < {config.SAME_PERSON_THRESHOLD:.2f}  — почти наверняка один человек")
    print(f"  {config.SAME_PERSON_THRESHOLD:.2f}..{config.DIFFERENT_PERSON_THRESHOLD:.2f}"
          f"  — серая зона")
    print(f"  > {config.DIFFERENT_PERSON_THRESHOLD:.2f}  — почти наверняка разные")
    print("=" * 70)

    for (name_a, idx_a, emb_a, _), (name_b, idx_b, emb_b, _) in combinations(all_embeddings, 2):
        dist = cosine_distance(emb_a, emb_b)
        label_a = f"{name_a}#{idx_a}"
        label_b = f"{name_b}#{idx_b}"

        # Подсказка по порогам.
        if dist < config.SAME_PERSON_THRESHOLD:
            verdict = "ОДИН ЧЕЛОВЕК"
        elif dist < config.DIFFERENT_PERSON_THRESHOLD:
            verdict = "серая зона"
        else:
            verdict = "РАЗНЫЕ"

        print(f"{label_a:>18}  ↔  {label_b:<18}  dist={dist:.4f}   [{verdict}]")

    print("\n[INFO] Готово.")


if __name__ == "__main__":
    import numpy as np  # noqa: F401  (для type hints в аннотации)
    main()