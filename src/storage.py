"""
Хранилище базы эталонов (Этап 5).

Что делает:
  - Строит базу: для каждого человека из data/photos/<имя>/ берёт все фото,
    строит эмбеддинги, УСРЕДНЯЕТ их в один вектор, сохраняет.
  - Сохраняет базу в один .npz-файл: вектора + имена + метаданные.
  - Загружает базу обратно.

Формат .npz:
  - "embeddings": матрица (N, 512) — по одному вектору на человека
  - "names":      массив строк (N,) — имена в порядке строк матрицы
  - "n_photos":   массив int (N,) — сколько фото усреднено для каждого

Почему .npz: это нативный формат numpy, компактный, быстрый,
без внешних зависимостей (в отличие от pickle или json).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from src import config
from src.embedder import FaceEmbedder


@dataclass
class PersonRecord:
    """Один человек в базе: имя, усреднённый эмбеддинг, число фото."""
    name: str
    embedding: np.ndarray      # (512,)
    n_photos: int


# ─────────────────────────────────────────────────────────────
# Построение базы из папок
# ─────────────────────────────────────────────────────────────

def _list_person_dirs() -> List[Path]:
    """
    Найти подпапки в data/photos/, которые считаем людьми.
    Каждая подпапка = один человек. Имя папки = имя в базе.
    """
    if not config.PHOTOS_DIR.exists():
        return []

    dirs = [
        d for d in config.PHOTOS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    return sorted(dirs)


def _list_images_in_dir(folder: Path) -> List[Path]:
    """Все картинки в папке (не рекурсивно), отсортированные по имени."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_IMAGE_EXTENSIONS
    )


def build_base(embedder: FaceEmbedder) -> List[PersonRecord]:
    """
    Пройти по всем подпапкам data/photos/, построить эмбеддинги
    и усреднить их для каждого человека.

    Args:
        embedder: уже инициализированный FaceEmbedder.

    Returns:
        Список PersonRecord.

    Логика усреднения:
      - для каждого фото берём САМОЕ уверенное лицо (det_score max),
        потому что на фото может быть несколько лиц;
      - усредняем эмбеддинги по L2-норме (они уже нормализованы ArcFace,
        но после усреднения нормируем ещё раз — чтобы косинусное
        расстояние оставалось корректным).
    """
    records: List[PersonRecord] = []

    person_dirs = _list_person_dirs()
    if not person_dirs:
        print(f"[WARN] В {config.PHOTOS_DIR} нет подпапок с людьми.")
        return records

    print(f"[INFO] Найдено людей (папок): {len(person_dirs)}")

    for person_dir in person_dirs:
        name = person_dir.name
        images = _list_images_in_dir(person_dir)

        if not images:
            print(f"[WARN] {name}: нет картинок, пропускаю.")
            continue

        embeddings_for_person: List[np.ndarray] = []

        for img_path in images:
            try:
                results = embedder.embed_image(img_path)
            except Exception as e:
                print(f"[WARN] {name}/{img_path.name}: ошибка — {e}")
                continue

            if not results:
                print(f"[WARN] {name}/{img_path.name}: лиц не найдено")
                continue

            # Берём лицо с максимальным det_score.
            best = max(results, key=lambda r: r.det_score)
            embeddings_for_person.append(best.embedding)

        if not embeddings_for_person:
            print(f"[WARN] {name}: ни одного валидного лица, пропускаю.")
            continue

        # Усредняем по L2-норме.
        stacked = np.stack(embeddings_for_person, axis=0)  # (K, 512)
        mean_emb = stacked.mean(axis=0)
        # Нормируем обратно к единичной длине.
        norm = np.linalg.norm(mean_emb) + 1e-10
        mean_emb = mean_emb / norm

        print(
            f"[INFO] {name}: усреднено {len(embeddings_for_person)} фото "
            f"(из {len(images)} файлов)"
        )

        records.append(
            PersonRecord(
                name=name,
                embedding=mean_emb.astype(np.float32),
                n_photos=len(embeddings_for_person),
            )
        )

    return records


# ─────────────────────────────────────────────────────────────
# Сохранение / загрузка
# ─────────────────────────────────────────────────────────────

def save_base(records: List[PersonRecord], path: Path = None) -> None:
    """Сохранить базу в .npz-файл."""
    if path is None:
        path = config.BASE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        print("[WARN] База пустая — сохранять нечего.")
        return

    embeddings = np.stack([r.embedding for r in records], axis=0)
    names = np.array([r.name for r in records], dtype=object)
    n_photos = np.array([r.n_photos for r in records], dtype=np.int32)

    np.savez(path, embeddings=embeddings, names=names, n_photos=n_photos)
    print(f"[INFO] База сохранена: {path} ({len(records)} человек)")


def load_base(path: Path = None) -> List[PersonRecord]:
    """Загрузить базу из .npz-файла. Если файла нет — вернуть пустой список."""
    if path is None:
        path = config.BASE_FILE

    if not path.exists():
        print(f"[WARN] Файл базы не найден: {path}")
        return []

    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]
    names = data["names"]
    n_photos = data["n_photos"]

    records = [
        PersonRecord(
            name=str(names[i]),
            embedding=embeddings[i].astype(np.float32),
            n_photos=int(n_photos[i]),
        )
        for i in range(len(names))
    ]
    print(f"[INFO] База загружена: {path} ({len(records)} человек)")
    return records