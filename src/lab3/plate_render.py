"""Рендеринг українського номерного знака та накладання його на фото авто.

Номерний знак у синтетичній сцені — власний (``KA 7429 MB``), а не той, що
був у демонстрації. Він малюється «з нуля» за пропорціями ДСТУ 4278:2019
(520 × 112 мм, синя смуга з прапором і кодом «UA» зліва) і переноситься на
місце знака на фото за допомогою перспективного перетворення (гомографії),
тож виглядає як частина автомобіля, а не як плоска наклейка.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Шрифт, близький до DIN 1451, яким набирають символи на номерах.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "DejaVuSans-Bold.ttf",
]

PX_PER_MM = 2            # роздільна здатність рендеру: 2 пікселі на мм
PLATE_MM = (520, 112)    # розмір стандартного знака (ширина, висота), мм


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Повертає перший доступний у системі жирний шрифт потрібного розміру."""
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise RuntimeError("Не знайдено жодного TrueType-шрифту для рендеру номера")


def render_ua_plate(text: str) -> np.ndarray:
    """Малює номерний знак і повертає його як BGR-зображення.

    ``text`` — номер у форматі «KA 7429 MB» (пробіли лише для відступів).
    """
    w, h = PLATE_MM[0] * PX_PER_MM, PLATE_MM[1] * PX_PER_MM
    img = Image.new("RGB", (w, h), (245, 245, 240))
    d = ImageDraw.Draw(img)

    # Чорна облямівка з заокругленими кутами
    d.rounded_rectangle([2, 2, w - 3, h - 3], radius=14, outline=(20, 20, 20), width=6)

    # Синя смуга зліва: прапор України + код країни «UA»
    band_w = 44 * PX_PER_MM
    d.rounded_rectangle([8, 8, band_w, h - 9], radius=8, fill=(0, 61, 165))
    fx0, fy0, fx1 = 22, 26, band_w - 14
    fh = 34
    d.rectangle([fx0, fy0, fx1, fy0 + fh // 2], fill=(0, 87, 183))        # синя половина
    d.rectangle([fx0, fy0 + fh // 2, fx1, fy0 + fh], fill=(255, 215, 0))  # жовта половина
    ua_font = _load_font(52)
    tw = d.textlength("UA", font=ua_font)
    d.text(((8 + band_w - tw) / 2, h - 86), "UA", font=ua_font, fill=(255, 255, 255))

    # Символи номера: підбираємо розмір шрифту так, щоб текст займав
    # усю ширину робочого поля праворуч від синьої смуги.
    field_x0, field_x1 = band_w + 18, w - 22
    size = 165
    font = _load_font(size)
    while d.textlength(text, font=font) > field_x1 - field_x0 and size > 40:
        size -= 2
        font = _load_font(size)
    tw = d.textlength(text, font=font)
    bbox = d.textbbox((0, 0), text, font=font)
    ty = (h - (bbox[3] - bbox[1])) / 2 - bbox[1]
    d.text((field_x0 + (field_x1 - field_x0 - tw) / 2, ty), text, font=font, fill=(15, 15, 15))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def paste_plate(photo: np.ndarray, plate: np.ndarray, quad: np.ndarray,
                brightness: float = 0.78, blur: float = 1.2) -> np.ndarray:
    """Переносить знак ``plate`` у чотирикутник ``quad`` на фото ``photo``.

    ``quad`` — 4 точки (верх-ліво, верх-право, низ-право, низ-ліво) у
    координатах фото. Знак затемнюється (``brightness``) і трохи
    розмивається (``blur``), щоб збігатися з освітленням і різкістю знімка.
    """
    h, w = plate.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    H = cv2.getPerspectiveTransform(src, np.float32(quad))
    size = (photo.shape[1], photo.shape[0])

    warped = cv2.warpPerspective(plate, H, size, flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, size)
    mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32)[..., None] / 255.0

    # Освітлення: знак рівномірно затемнюємо та додаємо легкий вертикальний
    # градієнт (верх знака трохи в тіні від бампера).
    warped = warped.astype(np.float32) * brightness
    ys = np.linspace(0.92, 1.03, size[1], dtype=np.float32)[:, None, None]
    warped = np.clip(warped * ys, 0, 255)
    if blur > 0:
        warped = cv2.GaussianBlur(warped, (0, 0), blur)

    out = photo.astype(np.float32) * (1 - mask) + warped * mask
    return out.astype(np.uint8)


if __name__ == "__main__":  # швидка перевірка рендеру
    out = Path("plate_preview.png")
    cv2.imwrite(str(out), render_ua_plate("KA 7429 MB"))
    print("saved", out)
