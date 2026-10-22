"""Виявлення облич і ідентифікація водія: YuNet + SFace (OpenCV Zoo).

* **YuNet** — легкий згортковий детектор облич (≈75 тис. параметрів).
  Повертає рамку обличчя, 5 ключових точок (очі, ніс, кутики рота) і
  впевненість.
* **SFace** — мережа розпізнавання облич (MobileFaceNet-подібна,
  навчена з функцією втрат SFace на MS1M). Обличчя спершу *вирівнюється*
  за 5 ключовими точками (``alignCrop`` — афінне перетворення до 112×112),
  а потім перетворюється на вектор-ознаку з 128 чисел.

Дві фотографії вважаються однією людиною, якщо косинусна схожість їх
векторів перевищує 0.363 (поріг, рекомендований авторами моделі).
Порівнюємо кожне обличчя з відео з **окремим** еталонним фото водія.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config as C


@dataclass
class FaceMatch:
    """Обличчя, знайдене в кадрі, та його схожість з еталоном водія."""
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float                            # впевненість YuNet
    similarity: float                       # косинусна схожість з еталоном

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2


class FaceIdentifier:
    def __init__(self, yunet=C.YUNET_MODEL, sface=C.SFACE_MODEL, score_thr: float = C.FACE_SCORE):
        self.detector = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), score_thr, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface), "")
        self.score_thr = score_thr
        self.reference: np.ndarray | None = None
        self.reference_face: np.ndarray | None = None  # вирівняне еталонне обличчя (для звіту)

    def detect(self, img: np.ndarray, score_thr: float | None = None) -> np.ndarray:
        """Повертає масив N×15: x, y, w, h, 5 пар координат точок, впевненість."""
        self.detector.setInputSize((img.shape[1], img.shape[0]))
        self.detector.setScoreThreshold(score_thr or self.score_thr)
        _, faces = self.detector.detect(img)
        return np.zeros((0, 15), np.float32) if faces is None else faces

    def embed(self, img: np.ndarray, face: np.ndarray) -> np.ndarray:
        """Вирівнює обличчя за ключовими точками і рахує 128-вимірний вектор SFace."""
        aligned = self.recognizer.alignCrop(img, face)
        return self.recognizer.feature(aligned)

    def set_reference(self, path: Path) -> None:
        """Завантажує еталонне фото водія; береться найбільше обличчя на фото."""
        img = cv2.imread(str(path))
        faces = self.detect(img, score_thr=0.5)
        if len(faces) == 0:
            raise RuntimeError(f"На еталонному фото {path} не знайдено обличчя")
        face = max(faces, key=lambda f: f[2] * f[3])
        self.reference_face = self.recognizer.alignCrop(img, face)
        self.reference = self.recognizer.feature(self.reference_face)

    def similarity(self, feat: np.ndarray) -> float:
        """Косинусна схожість вектора з еталоном водія."""
        return float(self.recognizer.match(self.reference, feat, cv2.FaceRecognizerSF_FR_COSINE))

    def process(self, frame: np.ndarray) -> list[FaceMatch]:
        """Знаходить усі обличчя в кадрі та порівнює кожне з еталоном."""
        out = []
        for f in self.detect(frame):
            x, y, w, h = f[:4]
            if w < 24 or h < 24:  # занадто дрібні обличчя SFace розпізнає ненадійно
                continue
            sim = self.similarity(self.embed(frame, f))
            out.append(FaceMatch((float(x), float(y), float(x + w), float(y + h)), float(f[14]), sim))
        return out
