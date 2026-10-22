"""Розпізнавання номерного знака: детектор YOLOv11-LP + EasyOCR + постобробка.

Конвеєр для одного кадру:

1. **Пошук знака.** YOLOv11s, донавчена на номерних знаках
   (``morsetechlab/yolov11-license-plate-detection``, Hugging Face), повертає
   рамку знака. Загальна COCO-модель такого класу не має.
2. **Підготовка.** Рамку трохи розширюємо, вирізаємо знак, збільшуємо в
   ``PLATE_UPSCALE`` разів (бікубічна інтерполяція) і переводимо у відтінки
   сірого з вирівнюванням контрасту CLAHE — так OCR краще бачить дрібні символи.
3. **OCR.** EasyOCR = детектор тексту CRAFT + розпізнавач CRNN
   (згорткова рекурентна мережа з CTC-декодуванням). Обмежуємо алфавіт
   латинськими великими літерами й цифрами.
4. **Постобробка.** OCR «не знає» формату номерів: прапорець і «UA» на
   синій смузі, сплутування O/0, I/1, B/8 тощо. Тому шукаємо в рядку вікно
   з 8 символів, яке після мінімальної кількості виправлень (з урахуванням
   позиції: літери — на місцях 1–2 і 7–8, цифри — на 3–6) відповідає
   шаблону українського номера ``LLDDDDLL``.
5. **Голосування.** Окремі кадри можуть читатися з помилками, тому
   результати з усіх кадрів об'єднуються: голосуванням за цілий рядок і
   посимвольним голосуванням (зважено на впевненість OCR).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

from . import config as C

# Типові помилки OCR: що «насправді» означає символ на позиції цифри / літери
_TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "U": "0", "I": "1", "L": "1", "J": "1", "T": "7",
             "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4"}
_TO_LETTER = {"0": "O", "1": "I", "8": "B", "4": "A", "6": "G", "7": "T", "5": "S", "2": "Z"}
_PATTERN = re.compile(C.UA_PLATE_REGEX)
_LETTERS = set(C.UA_PLATE_LETTERS)


@dataclass
class PlateRead:
    """Результат обробки одного знака в одному кадрі."""
    box: tuple[float, float, float, float]
    det_conf: float
    raw_text: str            # що повернув OCR (усі фрагменти зліва направо)
    ocr_conf: float          # середня впевненість OCR
    plate: str | None        # нормалізований номер або None, якщо шаблон не знайдено
    corrections: int = 0     # скільки символів довелося виправити


def normalize_plate(raw: str) -> tuple[str | None, int]:
    """Знаходить у сирому рядку OCR український номер ``LLDDDDLL``.

    Перебирає всі вікна довжиною 8 символів, виправляє символи відповідно
    до позиції (цифра/літера) і повертає варіант з найменшою кількістю
    виправлень. Якщо жодне вікно не вдається привести до шаблону — ``None``.
    """
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    best: tuple[str | None, int] = (None, 99)
    for i in range(len(s) - 7):
        win, fixed, n_fix = s[i:i + 8], [], 0
        for pos, ch in enumerate(win):
            if pos in (2, 3, 4, 5):          # позиції цифр
                if not ch.isdigit():
                    ch, n_fix = _TO_DIGIT.get(ch, ch), n_fix + 1
            else:                             # позиції літер
                if ch.isdigit():
                    ch, n_fix = _TO_LETTER.get(ch, ch), n_fix + 1
                if ch not in _LETTERS:
                    n_fix += 5                # недопустима літера — сильний штраф
            fixed.append(ch)
        cand = "".join(fixed)
        if _PATTERN.fullmatch(cand) and n_fix < best[1]:
            best = (cand, n_fix)
    # Більше двох виправлень — читання ненадійне, не довіряємо йому
    return best if best[1] <= 2 else (None, best[1])


class PlateReader:
    """Детектор номерних знаків + OCR."""

    def __init__(self, model_path=C.PLATE_MODEL, conf: float = C.PLATE_CONF,
                 device: str = C.DEVICE, upscale: int = C.PLATE_UPSCALE):
        self.det = YOLO(str(model_path))
        self.ocr = easyocr.Reader(["en"], gpu=False, verbose=False)
        self.conf, self.device, self.upscale = conf, device, upscale
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8))

    def detect(self, frame: np.ndarray) -> list[tuple[tuple[float, ...], float]]:
        """Повертає список (рамка, впевненість) знайдених номерних знаків."""
        r = self.det(frame, conf=self.conf, device=self.device, verbose=False)[0]
        return [(tuple(map(float, b)), float(c))
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]

    def preprocess(self, frame: np.ndarray, box) -> np.ndarray:
        """Вирізає знак з невеликим запасом, збільшує, сірий + CLAHE."""
        x1, y1, x2, y2 = box
        pw, ph = 0.04 * (x2 - x1), 0.12 * (y2 - y1)
        H, W = frame.shape[:2]
        crop = frame[max(int(y1 - ph), 0):min(int(y2 + ph), H), max(int(x1 - pw), 0):min(int(x2 + pw), W)]
        crop = cv2.resize(crop, None, fx=self.upscale, fy=self.upscale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)

    def read(self, frame: np.ndarray) -> list[PlateRead]:
        """Знаходить і читає всі номери в кадрі."""
        out = []
        for box, det_conf in self.detect(frame):
            img = self.preprocess(frame, box)
            res = self.ocr.readtext(img, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                                    paragraph=False, detail=1)
            res.sort(key=lambda r: min(p[0] for p in r[0]))  # фрагменти зліва направо
            raw = " ".join(r[1] for r in res)
            conf = float(np.mean([r[2] for r in res])) if res else 0.0
            plate, n_fix = normalize_plate(raw)
            out.append(PlateRead(box, det_conf, raw, conf, plate, n_fix))
        return out


def vote(reads: list[PlateRead]) -> dict:
    """Об'єднує читання з різних кадрів.

    * ``by_string`` — найчастіший нормалізований рядок (голоси = кількість кадрів);
    * ``by_char``   — посимвольне голосування, зважене на впевненість OCR:
      навіть якщо кожен кадр має одну помилку в різних місцях, разом вони
      дають правильний номер.
    """
    valid = [r for r in reads if r.plate]
    counts = Counter(r.plate for r in valid)
    by_char = None
    if valid:
        pos_votes: list[defaultdict] = [defaultdict(float) for _ in range(8)]
        for r in valid:
            for i, ch in enumerate(r.plate):
                pos_votes[i][ch] += max(r.ocr_conf, 0.05)
        by_char = "".join(max(v, key=v.get) for v in pos_votes)
    return {
        "frames_with_plate": len(reads),
        "valid_reads": len(valid),
        "counts": dict(counts.most_common()),
        "by_string": counts.most_common(1)[0][0] if counts else None,
        "by_char": by_char,
    }
