"""
Детектор лиц на базе insightface (Этап 2).

Задача модуля:
  - Инициализировать модель один раз (загрузка весов — дорогая операция).
  - Принимать кадр (numpy array BGR из OpenCV).
  - Возвращать список найденных лиц: рамка, ключевые точки, уверенность.

Детектор НЕ распознаёт "кто это". Только находит лица.
Распознавание (эмбеддинги) будет отдельным модулем — embedder.py.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from insightface.app import FaceAnalysis

from src import config


@dataclass
class DetectedFace:
    """Одно найденное лицо на кадре."""
    bbox: np.ndarray          # [x1, y1, x2, y2] — координаты рамки
    landmarks: np.ndarray     # 5 точек: глаза, нос, уголки рта. Форма (5, 2)
    det_score: float          # уверенность детектора, 0..1


class FaceDetector:
    """
    Обёртка над insightface FaceAnalysis.

    Пример:
        detector = FaceDetector()
        faces = detector.detect(frame)
        for f in faces:
            print(f.bbox, f.det_score)
    """

    def __init__(self) -> None:
        # providers=["CPUExecutionProvider"] — считаем на CPU.
        # Если появится GPU, заменим на "CUDAExecutionProvider".
        self._app = FaceAnalysis(
            name=config.MODEL_PACK,
            providers=["CPUExecutionProvider"],
        )

        # prepare(): загружает веса и настраивает входной размер.
        # ctx_id=-1 — CPU, 0/1/... — номер GPU.
        self._app.prepare(
            ctx_id=-1,
            det_size=config.DETECTION_SIZE,
            det_thresh=config.DETECTION_THRESHOLD,
        )

    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        """
        Найти все лица на кадре.

        Args:
            frame: numpy array в формате BGR (как отдаёт cv2.VideoCapture).

        Returns:
            Список DetectedFace, отсортированный по уверенности (сильнее — первее).
        """
        # insightface сам внутри конвертирует BGR → RGB.
        raw_faces = self._app.get(frame)

        result: List[DetectedFace] = []
        for f in raw_faces:
            result.append(
                DetectedFace(
                    bbox=f.bbox.astype(int),
                    landmarks=f.kps.astype(int),
                    det_score=float(f.det_score),
                )
            )

        # Сортируем: самые уверенные — первыми.
        result.sort(key=lambda x: x.det_score, reverse=True)
        return result