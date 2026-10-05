"""
Идентификация 1:N (Этап 5).

Что делает:
  - Принимает эмбеддинг запроса (512 чисел).
  - Сравнивает его с каждым эталоном в базе.
  - Возвращает лучший матч + расстояние, или «неизвестный».

Логика порога:
  - если лучший матч ближе порога → считаем «это он»;
  - иначе → «неизвестный».

Порог задаётся в config.IDENTIFICATION_THRESHOLD.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src import config
from src.embedder import cosine_distance
from src.storage import PersonRecord


@dataclass
class MatchResult:
    """Результат идентификации одного лица."""
    name: Optional[str]      # имя или None, если неизвестный
    distance: float          # расстояние до лучшего эталона
    is_known: bool           # True, если distance < threshold
    top_matches: List[tuple] # [(имя, расстояние), ...] — для отладки, топ-3


class FaceMatcher:
    """
    Сравнивает эмбеддинг запроса с базой эталонов.

    Пример:
        matcher = FaceMatcher(records)
        result = matcher.match(embedding)
        print(result.name, result.distance)
    """

    def __init__(self, records: List[PersonRecord]) -> None:
        self._records = records
        self._threshold = config.IDENTIFICATION_THRESHOLD

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = float(value)

    def match(self, embedding: np.ndarray) -> MatchResult:
        """
        Идентифицировать одно лицо.

        Args:
            embedding: (512,) numpy array.

        Returns:
            MatchResult.
        """
        if not self._records:
            return MatchResult(
                name=None, distance=float("inf"),
                is_known=False, top_matches=[],
            )

        # Считаем расстояние до каждого эталона.
        distances = [
            (rec.name, cosine_distance(embedding, rec.embedding))
            for rec in self._records
        ]
        # Сортируем по возрастанию расстояния (ближе — лучше).
        distances.sort(key=lambda x: x[1])

        best_name, best_dist = distances[0]
        is_known = best_dist < self._threshold

        return MatchResult(
            name=best_name if is_known else None,
            distance=float(best_dist),
            is_known=is_known,
            top_matches=distances[:3],
        )

    def match_many(self, embeddings: List[np.ndarray]) -> List[MatchResult]:
        """Идентифицировать сразу пачку эмбеддингов (например, все лица на кадре)."""
        return [self.match(e) for e in embeddings]