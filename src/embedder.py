"""
Извлечение эмбеддингов лиц (Этап 3).

Что делает модуль:
  - Берёт кадр (numpy BGR) или путь к фото.
  - Находит лица (через FaceDetector).
  - Для каждого лица строит эмбеддинг (512 чисел) — «математический отпечаток».
  - Возвращает список: эмбеддинг + исходное лицо (рамка, ключевые точки, score).

Эмбеддинг — это вектор, в котором:
  - у одного человека на разных фото значения близки,
  - у разных людей — далеки.
Именно по этому вектору позже будем сравнивать «кто это».

Модуль НЕ хранит базу и НЕ сравнивает лица. Только строит эмбеддинги.
Сравнение — в matcher.py (будет на Этапе 5).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from src import config


@dataclass
class FaceEmbedding:
    """Результат для одного лица: эмбеддинг + метаданные детекции."""
    embedding: np.ndarray     # вектор длины EXPECTED_EMBEDDING_DIM
    bbox: np.ndarray          # [x1, y1, x2, y2]
    landmarks: np.ndarray     # 5 точек
    det_score: float          # уверенность детектора


class FaceEmbedder:
    """
    Обёртка над insightface FaceAnalysis, но с фокусом на эмбеддинги.

    Пример:
        embedder = FaceEmbedder()
        results = embedder.embed_image("data/photos/img-s.jpg")
        for r in results:
            print(r.embedding.shape, r.det_score)
    """

    def __init__(self) -> None:
        # Та же модель, что и в детекторе. Insightface сам подгрузит
        # и детектор (SCRFD), и распознаватель (ArcFace).
        self._app = FaceAnalysis(
            name=config.MODEL_PACK,
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(
            ctx_id=-1,
            det_size=config.DETECTION_SIZE,
            det_thresh=config.DETECTION_THRESHOLD,
        )

    def embed_frame(self, frame: np.ndarray) -> List[FaceEmbedding]:
        """
        Найти лица на кадре и построить эмбеддинги для каждого.

        Args:
            frame: numpy array в формате BGR.

        Returns:
            Список FaceEmbedding, отсортированный по уверенности детектора.
        """
        raw_faces = self._app.get(frame)

        results: List[FaceEmbedding] = []
        for f in raw_faces:
            # f.embedding — уже L2-нормализованный вектор (ArcFace).
            # Если вдруг None (бывает при плохом детекте) — пропускаем.
            if getattr(f, "embedding", None) is None:
                continue

            results.append(
                FaceEmbedding(
                    embedding=f.embedding.astype(np.float32),
                    bbox=f.bbox.astype(int),
                    landmarks=f.kps.astype(int),
                    det_score=float(f.det_score),
                )
            )

        results.sort(key=lambda x: x.det_score, reverse=True)
        return results

    def embed_image(self, image_path: Union[str, Path]) -> List[FaceEmbedding]:
        """
        Прочитать картинку с диска и построить эмбеддинги для всех лиц на ней.

        Args:
            image_path: путь к .jpg/.png и т.п.

        Returns:
            Список FaceEmbedding. Пустой, если лиц не найдено или файл не прочитан.

        Raises:
            FileNotFoundError: если файла нет.
            ValueError: если файл не читается как изображение.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        # cv2.imread корректно читает пути с кириллицей только через
        # np.fromfile + cv2.imdecode. У нас пути латиницей, но сделаем
        # надёжно — на будущее.
        data = np.fromfile(str(path), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Не удалось прочитать изображение: {path}")

        return self.embed_frame(frame)


# ─────────────────────────────────────────────────────────────
# Утилиты для сравнения эмбеддингов
# ─────────────────────────────────────────────────────────────
# Они пригодятся и в скрипте теста, и позже в matcher.py.

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Косинусное сходство двух векторов. Диапазон [-1..1], ближе к 1 — похожее.

    Insightface уже нормализует эмбеддинги, но на всякий случай
    нормируем ещё раз — это дешёвая операция.
    """
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a_norm, b_norm))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Косинусное расстояние: 1 - cosine_similarity.
    Диапазон [0..2], но для лиц на практике [0..1].
        ~0    — один и тот же человек
        > 0.6 — разные люди
    """
    return 1.0 - cosine_similarity(a, b)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Евклидово расстояние между векторами.

    Для нормализованных эмбеддингов insightface:
        euclidean ≈ sqrt(2 * cosine_distance)
    Так что использовать можно любое, но косинусное — стандарт.
    """
    return float(np.linalg.norm(a - b))