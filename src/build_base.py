"""
Этап 5, шаг 1. Построение базы эталонов.

Что делает:
  1. Проходит по data/photos/<имя>/, где <имя> — имя человека.
  2. Для каждого человека строит эмбеддинги всех его фото.
  3. Усредняет их в один вектор.
  4. Сохраняет базу в data/embeddings/base.npz.

Запуск:
  python scripts/build_base.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedder import FaceEmbedder         # noqa: E402
from src.storage import build_base, save_base  # noqa: E402
from src import config                         # noqa: E402


def main() -> None:
    print("[INFO] Загружаю модель...")
    embedder = FaceEmbedder()
    print("[INFO] Модель загружена.\n")

    records = build_base(embedder)

    if not records:
        print("[ERROR] База пустая. Проверь, что в data/photos/ есть подпапки "
              "с именами людей и фото внутри.")
        return

    print()
    for r in records:
        print(f"  {r.name:<10} — {r.n_photos} фото, embedding.shape={r.embedding.shape}")

    print()
    save_base(records, config.BASE_FILE)


if __name__ == "__main__":
    main()