"""Генерація синтетичного відео для лабораторної роботи.

Замість зйомки на телефон (дозволено викладачем) сцена збирається з вільних
фотографій Wikimedia Commons та тестового зображення Ultralytics:

* фон і головний автомобіль — ЗАЗ Sens на шкільному подвір'ї (поруч стоїть
  другий, синій автомобіль); на місце номера наноситься власний знак
  ``KA 7429 MB`` (див. :mod:`lab3.plate_render`);
* водій — людина, вирізана з фото 2012 р., стоїть біля машини ближче до камери;
* пасажир — людина, що йде зліва направо (**єдиний рухомий об'єкт**);
* додатковий об'єкт — собака (цуценя), що сидить біля машин.

Людей і собаку вирізаємо сегментаційною мережею YOLO11m-seg (маска
екземпляра → альфа-канал). Віртуальна камера імітує зйомку з руки:
плавне наближення до номера й водія, панорамування, дрібне тремтіння,
шум сенсора. Тривалість — 40 с, 1280×720, 25 кадрів/с.

Оскільки сцену побудовано програмно, для кожного кадру відомі істинні
рамки об'єктів, їх видимість, номер, хто водій і хто рухається. Це
зберігається у JSON як еталонна розмітка (ground truth) і дає змогу
кількісно оцінити якість розпізнавання.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

from . import config as C
from .plate_render import paste_plate, render_ua_plate

# --------------------------------------------------------------------------
# Параметри сцени
# --------------------------------------------------------------------------
OUT_W, OUT_H = 1280, 720
FPS = 25
DURATION_S = 40
PLATE_TEXT = "KA 7429 MB"
# Чотирикутник, у який вписується номер на фото авто (верх-ліво, верх-право,
# низ-право, низ-ліво), — визначено вручну за зачорненою ділянкою знака.
PLATE_QUAD = np.float32([[430, 930], [752, 944], [750, 1016], [430, 1004]])

PASSENGER_HEIGHT = 800   # висота пасажира у «світових» пікселях (фото 1920×1440)
DOG_HEIGHT = 300
DRIVER_HEAD_Y = 620      # де у світі опиняється верх голови водія
WALK_START_S, WALK_END_S = 3.0, 33.0  # пасажир іде з 3-ї по 33-ю секунду


@dataclass
class Cutout:
    """Вирізаний об'єкт: BGR-зображення + альфа-маска (0…1) + клас COCO."""
    name: str
    cls: str
    rgb: np.ndarray
    alpha: np.ndarray
    meta: dict = field(default_factory=dict)

    def scaled(self, s: float) -> "Cutout":
        size = (max(1, round(self.rgb.shape[1] * s)), max(1, round(self.rgb.shape[0] * s)))
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        return Cutout(self.name, self.cls, cv2.resize(self.rgb, size, interpolation=interp),
                      cv2.resize(self.alpha, size, interpolation=cv2.INTER_LINEAR), dict(self.meta))


# --------------------------------------------------------------------------
# Вирізання об'єктів сегментаційною мережею
# --------------------------------------------------------------------------
def cut_instance(seg: YOLO, path: Path, cls: str, name: str, pick: str = "conf",
                 hint_xy: tuple[float, float] | None = None) -> Cutout:
    """Знаходить на фото екземпляр класу ``cls`` і вирізає його за маскою.

    Якщо на фото кілька екземплярів, обирається найвпевненіший або (за
    наявності ``hint_xy``) найближчий до вказаної точки.
    """
    img = cv2.imread(str(path))
    r = seg(img, verbose=False, retina_masks=True)[0]
    names = r.names
    cand = [i for i, c in enumerate(r.boxes.cls.cpu().numpy()) if names[int(c)] == cls]
    if not cand:
        raise RuntimeError(f"{path.name}: не знайдено об'єкт класу {cls}")
    boxes = r.boxes.xyxy.cpu().numpy()
    if hint_xy is not None:
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        idx = min(cand, key=lambda i: (cx[i] - hint_xy[0]) ** 2 + (cy[i] - hint_xy[1]) ** 2)
    else:
        idx = max(cand, key=lambda i: float(r.boxes.conf[i]))

    # Маска з полігона в повній роздільній здатності (точніше за растрову маску мережі)
    mask = np.zeros(img.shape[:2], np.uint8)
    poly = r.masks.xy[idx].astype(np.int32)
    cv2.fillPoly(mask, [poly], 255)
    # Прибираємо «ореол» фону по краях: звужуємо маску й розмиваємо межу
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    alpha = cv2.GaussianBlur(mask, (0, 0), 1.6).astype(np.float32) / 255.0

    x, y, w, h = cv2.boundingRect(mask)
    return Cutout(name, cls, img[y:y + h, x:x + w].copy(), alpha[y:y + h, x:x + w].copy(),
                  {"source": path.name, "conf": float(r.boxes.conf[idx])})


def match_colors(cut: Cutout, target: np.ndarray, strength: float = 0.5) -> Cutout:
    """Наближає колірну статистику вирізаного об'єкта до фону (перенос у LAB).

    Фото зроблені за різного освітлення; часткове вирівнювання середнього та
    дисперсії каналів L*a*b* робить вставку природнішою.
    """
    src = cv2.cvtColor(cut.rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = cut.alpha > 0.5
    s_mean, s_std = src[m].mean(0), src[m].std(0) + 1e-6
    t_mean, t_std = tgt.reshape(-1, 3).mean(0), tgt.reshape(-1, 3).std(0)
    moved = (src - s_mean) / s_std * (s_std + (t_std - s_std) * strength) + \
        s_mean + (t_mean - s_mean) * strength
    rgb = cv2.cvtColor(np.clip(moved, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return Cutout(cut.name, cut.cls, rgb, cut.alpha, cut.meta)


# --------------------------------------------------------------------------
# Композиція шарів
# --------------------------------------------------------------------------
def paste(canvas: np.ndarray, cut: Cutout, x: int, y: int,
          occupancy: np.ndarray | None = None) -> None:
    """Накладає ``cut`` на ``canvas`` (на місці) з лівим верхнім кутом (x, y).

    Якщо передано ``occupancy`` (bool-маска світу), у ній позначаються
    пікселі, які зайняв об'єкт, — це потрібно для розрахунку видимості.
    """
    H, W = canvas.shape[:2]
    h, w = cut.alpha.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    a = cut.alpha[y0 - y:y1 - y, x0 - x:x1 - x][..., None]
    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (roi * (1 - a) + cut.rgb[y0 - y:y1 - y, x0 - x:x1 - x] * a).astype(np.uint8)
    if occupancy is not None:
        occupancy[y0:y1, x0:x1] |= a[..., 0] > 0.5


def drop_shadow(canvas: np.ndarray, cx: int, cy: int, rx: int, ry: int, strength=0.35) -> None:
    """Малює м'яку тінь-еліпс під ногами/лапами — «прив'язує» об'єкт до землі."""
    sh = np.zeros(canvas.shape[:2], np.float32)
    cv2.ellipse(sh, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), ry * 0.6 + 2)[..., None] * strength
    canvas[:] = (canvas.astype(np.float32) * (1 - sh)).astype(np.uint8)


# --------------------------------------------------------------------------
# Віртуальна камера
# --------------------------------------------------------------------------
def _smooth(t: float) -> float:
    """Плавний перехід 0→1 (smoothstep) — камера не смикається на ключових точках."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


# Ключові положення камери: (час, x центру, y центру, наближення) у світі 1920×1440
CAMERA_KEYS = [
    (0.0, 960, 800, 1.00),   # загальний план
    (7.0, 960, 800, 1.00),
    (14.0, 1000, 900, 1.35),  # наближення до номера та водія
    (25.0, 1120, 870, 1.35),  # повільна панорама вправо
    (32.0, 960, 800, 1.00),   # повернення до загального плану
    (40.0, 960, 800, 1.00),
]


def camera_at(t: float, world_wh: tuple[int, int]) -> dict:
    """Стан камери в момент ``t``: центр вікна, наближення, поворот, матриця.

    Повертає матрицю 2×3 ``M`` перетворення «світ → кадр» (масштаб +
    поворот + зсув). До ключової траєкторії додається тремтіння рук —
    сума синусоїд різних частот.
    """
    for (t0, x0, y0, z0), (t1, x1, y1, z1) in zip(CAMERA_KEYS, CAMERA_KEYS[1:]):
        if t0 <= t <= t1:
            k = _smooth((t - t0) / (t1 - t0))
            cx, cy, z = x0 + (x1 - x0) * k, y0 + (y1 - y0) * k, z0 + (z1 - z0) * k
            break
    else:
        _, cx, cy, z = CAMERA_KEYS[-1]

    # Тремтіння рук (у світових пікселях) і невеликий поворот кадру
    cx += 7 * math.sin(2 * math.pi * 0.63 * t) + 4 * math.sin(2 * math.pi * 1.37 * t + 1.1)
    cy += 5 * math.sin(2 * math.pi * 0.51 * t + 0.4) + 3 * math.sin(2 * math.pi * 1.71 * t)
    angle = 0.35 * math.sin(2 * math.pi * 0.29 * t)  # градуси

    W, H = world_wh
    win_w, win_h = 1920 / z, 1080 / z
    # Не дозволяємо вікну камери вийти за межі «світу»
    cx = min(max(cx, win_w / 2 + 2), W - win_w / 2 - 2)
    cy = min(max(cy, win_h / 2 + 2), H - win_h / 2 - 2)

    s = OUT_W / win_w
    M = cv2.getRotationMatrix2D((cx, cy), angle, s)
    M[0, 2] += OUT_W / 2 - cx
    M[1, 2] += OUT_H / 2 - cy
    return {"cx": cx, "cy": cy, "zoom": z, "angle": angle, "scale": s, "M": M}


# --------------------------------------------------------------------------
# Еталонна розмітка
# --------------------------------------------------------------------------
def _visible_box(mask: np.ndarray, M: np.ndarray) -> tuple[list[float] | None, float]:
    """Переводить видиму частину маски об'єкта у кадр.

    Повертає рамку видимої частини в координатах кадру та частку видимих
    пікселів відносно всієї маски об'єкта.
    """
    # Беремо кожен 3-й піксель по обох осях — точність рамки ±3 px світу,
    # зате розрахунок у ~9 разів швидший.
    step = 3
    ys, xs = np.nonzero(mask[::step, ::step])
    if len(xs) == 0:
        return None, 0.0
    xs, ys = xs * step, ys * step
    pts = np.stack([xs, ys, np.ones_like(xs)], 1).astype(np.float32) @ M.T
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < OUT_W) & (pts[:, 1] >= 0) & (pts[:, 1] < OUT_H)
    if inside.sum() < 30:
        return None, 0.0
    p = pts[inside]
    box = [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]
    return box, float(inside.sum()) * step * step


def build_scene(out_video: Path = C.SCENE_VIDEO, out_gt: Path = C.SCENE_GT,
                preview_dir: Path | None = None, duration_s: float = DURATION_S) -> dict:
    """Будує сцену, рендерить відео та зберігає еталонну розмітку.

    Повертає словник з метаданими сцени (той самий, що записується у JSON).
    """
    out_video.parent.mkdir(parents=True, exist_ok=True)
    seg = YOLO(str(C.SEG_MODEL))

    # 1. Фон: фото авто з нанесеним власним номерним знаком
    world = cv2.imread(str(C.ASSETS_DIR / "car_zaz_sens.jpg"))
    world = paste_plate(world, render_ua_plate(PLATE_TEXT), PLATE_QUAD)
    WH = (world.shape[1], world.shape[0])

    # Маски автомобілів на фоні — для розрахунку їх видимості
    r = seg(world, verbose=False, retina_masks=True)[0]
    car_masks = []
    for i, c in enumerate(r.boxes.cls.cpu().numpy()):
        if r.names[int(c)] == "car":
            m = np.zeros(world.shape[:2], np.uint8)
            cv2.fillPoly(m, [r.masks.xy[i].astype(np.int32)], 1)
            car_masks.append((cv2.boundingRect(m)[2] * cv2.boundingRect(m)[3], m.astype(bool)))
    car_masks.sort(key=lambda t: -t[0])
    main_car_mask, second_car_mask = car_masks[0][1], car_masks[1][1]
    plate_mask = np.zeros(world.shape[:2], np.uint8)
    cv2.fillPoly(plate_mask, [PLATE_QUAD.astype(np.int32)], 1)
    plate_mask = plate_mask.astype(bool)

    # 2. Вирізаємо водія, пасажира та собаку
    driver = cut_instance(seg, C.ASSETS_DIR / "driver_in_scene_2012.jpg", "person", "driver")
    passenger = cut_instance(seg, C.ASSETS_DIR / "passenger_source.jpg", "person", "passenger",
                             hint_xy=(283, 630))
    dog = cut_instance(seg, C.ASSETS_DIR / "dog_puppy.jpg", "dog", "dog")

    driver = match_colors(driver, world, 0.45)
    passenger = match_colors(passenger, world, 0.35)
    dog = match_colors(dog, world, 0.35)

    # Масштабуємо до правдоподібних розмірів відносно авто
    driver = driver.scaled((WH[1] - DRIVER_HEAD_Y) / driver.rgb.shape[0])
    passenger = passenger.scaled(PASSENGER_HEIGHT / passenger.rgb.shape[0])
    dog = dog.scaled(DOG_HEIGHT / dog.rgb.shape[0])

    # Розміщення статичних об'єктів (лівий верхній кут у світі)
    dog_xy = (430 - dog.rgb.shape[1] // 2, 1290 - dog.rgb.shape[0])
    driver_xy = (1650 - driver.rgb.shape[1] // 2, WH[1] - driver.rgb.shape[0])

    # Статичний задній план: фон + тінь і собака
    static_bg = world.copy()
    drop_shadow(static_bg, 430, 1288, int(dog.rgb.shape[1] * 0.45), 18)
    dog_occ = np.zeros(world.shape[:2], bool)
    paste(static_bg, dog, *dog_xy, occupancy=dog_occ)
    driver_occ = np.zeros(world.shape[:2], bool)
    paste(np.zeros_like(world), driver, *driver_xy, occupancy=driver_occ)

    n_frames = int(FPS * duration_s)
    writer = imageio_ffmpeg.write_frames(str(out_video), (OUT_W, OUT_H), fps=FPS, codec="libx264",
                                         pix_fmt_out="yuv420p", quality=None, bitrate="8M",
                                         macro_block_size=1)  # ~8 Мбіт/с, як у телефона
    writer.send(None)
    rng = np.random.default_rng(7)  # фіксоване зерно — відтворюваний шум
    frames_gt = []

    for f in range(n_frames):
        t = f / FPS
        canvas = static_bg.copy()

        # 3. Рухомий об'єкт: пасажир іде зліва направо з легким «підстрибуванням» кроку
        k = (t - WALK_START_S) / (WALK_END_S - WALK_START_S)
        pass_occ = np.zeros(world.shape[:2], bool)
        pass_box_world = None
        if 0.0 <= k <= 1.0:
            x_feet = -260 + (2180 + 260) * k
            y_feet = 1345 + 25 * k + 7 * abs(math.sin(math.pi * 1.8 * t))
            px = int(x_feet - passenger.rgb.shape[1] / 2)
            py = int(y_feet - passenger.rgb.shape[0])
            drop_shadow(canvas, int(x_feet), int(y_feet) - 4, int(passenger.rgb.shape[1] * 0.5), 16, 0.3)
            paste(canvas, passenger, px, py, occupancy=pass_occ)
            pass_box_world = [px, py, px + passenger.rgb.shape[1], py + passenger.rgb.shape[0]]

        # 4. Водій — найближчий до камери (перекриває все інше);
        #    ледь помітне «дихання» ±2 px, але він нерухомий.
        sway = int(round(2 * math.sin(2 * math.pi * 0.25 * t)))
        paste(canvas, driver, driver_xy[0] + sway, driver_xy[1])

        # 5. Камера: вікно → кадр 1280×720, потім ефекти «телефонної» зйомки
        cam = camera_at(t, WH)
        frame = cv2.warpAffine(canvas, cam["M"], (OUT_W, OUT_H), flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_REFLECT)
        frame = cv2.GaussianBlur(frame, (0, 0), 0.6)
        gain = 1.0 + 0.025 * math.sin(2 * math.pi * 0.17 * t)  # повільна зміна експозиції
        noise = rng.normal(0, 2.2, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) * gain + noise, 0, 255).astype(np.uint8)
        writer.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())

        # 6. Еталонна розмітка: видимі рамки з урахуванням перекриттів (порядок шарів:
        #    другий авто < головний авто < собака < пасажир < водій)
        drv = np.roll(driver_occ, sway, axis=1)
        layers = {
            "car_second": ("car", second_car_mask & ~main_car_mask & ~dog_occ & ~pass_occ & ~drv, second_car_mask),
            "car_main": ("car", main_car_mask & ~dog_occ & ~pass_occ & ~drv, main_car_mask),
            "plate": ("plate", plate_mask & ~dog_occ & ~pass_occ & ~drv, plate_mask),
            "dog": ("dog", dog_occ & ~pass_occ & ~drv, dog_occ),
            "passenger": ("person", pass_occ & ~drv, pass_occ),
            "driver": ("person", drv, drv),
        }
        objs = []
        for name, (cls, vis_mask, full_mask) in layers.items():
            if name == "passenger" and pass_box_world is None:
                continue
            box, n_vis = _visible_box(vis_mask, cam["M"])
            if box is None:
                continue
            objs.append({"name": name, "cls": cls, "box": [round(v, 1) for v in box],
                         "visibility": round(n_vis / max(full_mask.sum(), 1), 3)})
        frames_gt.append({"frame": f, "t": round(t, 3),
                          "camera": {k2: round(float(v), 4) for k2, v in cam.items() if k2 != "M"},
                          "M": [[round(float(v), 6) for v in row] for row in cam["M"]],
                          "objects": objs})

        if preview_dir is not None and f in (0, 250, 450, 650, 900):
            preview_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_dir / f"scene_frame_{f:04d}.jpg"), frame)

    writer.close()

    meta = {
        "video": out_video.name, "width": OUT_W, "height": OUT_H, "fps": FPS,
        "frames": n_frames, "duration_s": duration_s,
        "plate_text": PLATE_TEXT.replace(" ", ""),
        "driver": "driver", "moving_object": "passenger",
        "objects": {
            "car_main": "ЗАЗ Sens з номером KA 7429 MB (головний автомобіль)",
            "car_second": "синій автомобіль ліворуч (другий автомобіль)",
            "driver": "водій біля автомобіля (фото 2012 р.)",
            "passenger": "пасажир, що йде зліва направо (рухомий об'єкт)",
            "dog": "собака (цуценя) біля автомобілів",
        },
        "sources": {c.name: c.meta for c in (driver, passenger, dog)},
        "frames_gt": frames_gt,
    }
    out_gt.write_text(json.dumps(meta, ensure_ascii=False))
    return meta
