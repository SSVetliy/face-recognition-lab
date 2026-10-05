"""
Логирование визитов (Этап 6).

Что делает:
  - Принимает имя распознанного человека и расстояние.
  - Решает, писать ли запись в CSV (антиспам через cooldown).
  - Пишет строку: timestamp, name, distance, is_known.

Логика cooldown:
  - Для каждого имени храним timestamp последней записи.
  - Если с тех пор прошло < VISIT_COOLDOWN_SECONDS — не пишем.
  - Иначе — пишем и обновляем timestamp.

Это даёт: «человек зашёл в 9:15, следующий раз залогируем не раньше 9:20».
Именно то, что нужно для системы контроля прихода сотрудников.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict

from src import config


class VisitLogger:
    """
    Логирует визиты в CSV с антиспамом.

    Пример:
        logger = VisitLogger()
        logger.log("stas", distance=0.25, is_known=True)  # запишет
        logger.log("stas", distance=0.27, is_known=True)  # не запишет (cooldown)
    """

    def __init__(self, path: Path = None, cooldown_seconds: int = None) -> None:
        self._path = path or config.VISITS_FILE
        self._cooldown = cooldown_seconds or config.VISIT_COOLDOWN_SECONDS
        # Храним время последнего логирования каждого имени.
        self._last_seen: Dict[str, datetime] = {}

        # Создаём папку и файл с заголовком, если их нет.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            with open(self._path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(config.VISITS_CSV_HEADER)
            print(f"[INFO] Создан файл лога визитов: {self._path}")

    def log(self, name: str, distance: float, is_known: bool) -> bool:
        """
        Записать визит, если прошёл cooldown.

        Args:
            name: имя (или "UNKNOWN").
            distance: расстояние до лучшего матча.
            is_known: True, если distance < threshold.

        Returns:
            True, если запись сделана. False, если пропущена по cooldown.
        """
        now = datetime.now()

        # Проверяем cooldown.
        last = self._last_seen.get(name)
        if last is not None:
            elapsed = (now - last).total_seconds()
            if elapsed < self._cooldown:
                return False

        # Пишем в CSV.
        with open(self._path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                now.strftime("%Y-%m-%d %H:%M:%S"),
                name,
                f"{distance:.4f}",
                int(is_known),
            ])

        self._last_seen[name] = now
        return True

    def reset(self, name: str = None) -> None:
        """
        Сбросить cooldown для имени (или для всех, если name=None).
        Полезно для теста.
        """
        if name is None:
            self._last_seen.clear()
        else:
            self._last_seen.pop(name, None)