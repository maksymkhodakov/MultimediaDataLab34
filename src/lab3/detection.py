"""Виявлення та відстеження об'єктів: YOLO11 (COCO) + ByteTrack.

YOLO11 (You Only Look Once, v11, Ultralytics) за один прохід мережі
повертає рамки, класи та впевненість для 80 класів набору COCO. Ми
залишаємо лише класи, що стосуються завдання (люди, автомобілі, тварини,
рюкзаки/сумки).

ByteTrack пов'язує детекції сусідніх кадрів в «треки»: кожен об'єкт
отримує номер, що зберігається між кадрами. Це дозволяє рахувати
статистику по об'єкту (скільки кадрів видно, як змістився, чиє обличчя).
Особливість ByteTrack — він використовує і детекції з низькою
впевненістю для продовження вже існуючих треків, тому об'єкт не
«губиться», коли його частково перекриває інший.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from . import config as C


@dataclass
class Detection:
    """Один об'єкт в одному кадрі."""
    track_id: int
    cls: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 у пікселях кадру

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2


class ObjectTracker:
    """Обгортка над ``YOLO.track`` з фільтром класів і зручним форматом виходу."""

    def __init__(self, model_path=C.COCO_MODEL, conf: float = C.DET_CONF,
                 device: str = C.DEVICE, classes=C.TARGET_CLASSES):
        self.model = YOLO(str(model_path))
        self.names = self.model.names  # id класу → назва
        self.class_ids = [i for i, n in self.names.items() if n in classes]
        self.conf = conf
        self.device = device

    def update(self, frame: np.ndarray) -> list[Detection]:
        """Обробляє черговий кадр і повертає список відстежуваних об'єктів.

        ``persist=True`` зберігає стан трекера між викликами — саме так
        ByteTrack «пам'ятає» об'єкти з попередніх кадрів.
        """
        r = self.model.track(frame, persist=True, tracker=C.TRACKER_CFG, conf=self.conf,
                             classes=self.class_ids, device=self.device, verbose=False)[0]
        if r.boxes.id is None:  # у кадрі немає жодного підтвердженого треку
            return []
        ids = r.boxes.id.int().cpu().tolist()
        cls = r.boxes.cls.int().cpu().tolist()
        conf = r.boxes.conf.cpu().tolist()
        xyxy = r.boxes.xyxy.cpu().numpy()
        return [Detection(t, self.names[c], float(p), tuple(map(float, b)))
                for t, c, p, b in zip(ids, cls, conf, xyxy)]
