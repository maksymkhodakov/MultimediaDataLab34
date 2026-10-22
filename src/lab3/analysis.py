"""Аналіз результатів по треках: рухомий об'єкт, водій, номер автомобіля.

Покадрові результати детекторів «шумні»: об'єкт може бути не знайдений в
окремому кадрі, обличчя — повернуте, номер — прочитаний з помилкою.
Тому всі рішення приймаються **по треку** (по всій історії об'єкта), а не
по одному кадру.

**Рухомий об'єкт.** Положення рамок переводяться в систему координат
першого кадру (компенсація руху камери, див. :mod:`camera_motion`). Для
кожної з чотирьох сторін рамки рахуємо розмах положень (90-й мінус 10-й
перцентиль), беремо медіану по сторонах і ділимо на характерний розмір
об'єкта √(w·h). Об'єкт рухається, якщо ця величина > 0.25. Медіана по
сторонах робить оцінку стійкою до часткових перекриттів (змінюється одна
сторона рамки, а не всі), а сторони, що впираються в край кадру
(обрізаний об'єкт), не враховуються.

Для порівняння рахується і «наївна» оцінка з демонстрації — зміщення
центру рамки від першої до останньої появи у координатах кадру, без
компенсації руху камери.

Перед аналізом треки «очищуються»: фрагменти одного нерухомого об'єкта,
що отримали різні номери (ID switch під час наближення камери),
об'єднуються, а треки, що весь час лежать усередині рамки іншого об'єкта
того самого класу, відкидаються. Так само відкидаються «крайові» треки:
низька середня впевненість (< 0.5) і ≥ 80 % кадрів рамка впирається в
край кадру (напр. піднята рука людини, що потрапила в кадр лише частково).

**Водій** — трек класу ``person``, обличчя якого в середньому найбільш
схоже на еталонне фото, якщо середня схожість вища за поріг SFace.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

from . import config as C
from .camera_motion import CameraMotionEstimator
from .plates import vote

ROLE_DRIVER = "ВОДІЙ"
ROLE_PERSON = "інша людина (пасажир)"
ROLE_CAR_PLATE = "автомобіль з номером"
ROLE_CAR = "інший автомобіль"
ROLE_EXTRA = "додатковий об'єкт"


@dataclass
class TrackSummary:
    track_id: int
    cls: str
    n_frames: int
    first_frame: int
    last_frame: int
    mean_conf: float
    move_score: float          # з компенсацією руху камери
    move_score_naive: float    # як у демонстрації: без компенсації (перша → остання поява)
    move_score_naive_max: float  # без компенсації, максимальне зміщення від першої появи
    moving: bool
    moving_naive: bool
    face_samples: int = 0
    mean_similarity: float | None = None
    role: str = ""
    similarities: list = field(default_factory=list, repr=False)


def _box_to_ref(box, T) -> np.ndarray:
    """Переводить рамку в систему координат першого кадру (обхоплююча рамка 4 кутів)."""
    x1, y1, x2, y2 = box
    p = CameraMotionEstimator.apply(T, [[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])


def movement_score(samples, frame_wh) -> float:
    """Нормований розмах положення рамки після компенсації руху камери.

    ``samples`` — список пар (рамка в кадрі, матриця «кадр → перший кадр»).
    """
    W, H = frame_wh
    m = C.BORDER_MARGIN
    refs = [_box_to_ref(b, T) for b, T in samples]
    areas = np.array([max(r[2] - r[0], 1) * max(r[3] - r[1], 1) for r in refs])
    # Спостереження, де об'єкт видно лише частково (обрізаний краєм кадру
    # чи перекритий) мають помітно меншу площу — їх рамка ненадійна.
    full = areas >= 0.7 * np.percentile(areas, 90)
    edges: list[list[float]] = [[], [], [], []]
    sizes = []
    for (box, _), ref, ok in zip(samples, refs, full):
        if not ok:
            continue
        sizes.append(np.sqrt(max(ref[2] - ref[0], 1) * max(ref[3] - ref[1], 1)))
        x1, y1, x2, y2 = box
        truncated = [x1 <= m, y1 <= m, x2 >= W - m, y2 >= H - m]
        for i in range(4):
            if not truncated[i]:
                edges[i].append(ref[i])
    size = float(np.median(sizes))
    ranges = [np.percentile(e, 90) - np.percentile(e, 10) for e in edges if len(e) >= 5]
    if not ranges:  # об'єкт увесь час обрізаний краєм кадру — беремо центри
        c = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in (_box_to_ref(b, T) for b, T in samples)])
        ranges = [float(np.ptp(c[:, 0])), float(np.ptp(c[:, 1]))]
    return float(np.median(ranges) / size)


def naive_movement_score(samples) -> float:
    """Оцінка з демонстрації: зміщення центру (перша → остання поява) / розмір."""
    (b0, _), (b1, _) = samples[0], samples[-1]
    c0 = np.array([(b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2])
    c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
    size = np.sqrt((b0[2] - b0[0]) * (b0[3] - b0[1]))
    return float(np.linalg.norm(c1 - c0) / size)


def naive_max_movement_score(samples) -> float:
    """Без компенсації: максимальне зміщення центру від першої появи / розмір."""
    b0 = samples[0][0]
    c0 = np.array([(b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2])
    size = np.sqrt((b0[2] - b0[0]) * (b0[3] - b0[1]))
    c = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b, _ in samples])
    return float(np.linalg.norm(c - c0, axis=1).max() / size)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def merge_fragments(records: list[dict], max_overlap: float = 0.2, min_iou: float = 0.6) -> dict[int, int]:
    """Об'єднує фрагменти треку одного нерухомого об'єкта.

    Два треки одного класу вважаються одним об'єктом, якщо вони майже не
    співіснують у часі (перекриття ≤ ``max_overlap`` коротшого треку), а їх
    медіанні рамки в системі координат першого кадру збігаються (IoU ≥
    ``min_iou``). Номери треків у ``records`` замінюються на номер
    першого фрагмента. Повертає відображення «старий id → новий id».
    """
    info = defaultdict(lambda: {"frames": set(), "boxes": [], "cls": Counter()})
    for rec in records:
        for d in rec["dets"]:
            t = info[d.track_id]
            t["frames"].add(rec["frame"])
            t["boxes"].append(_box_to_ref(d.box, rec["T"]))
            t["cls"][d.cls] += 1
    med = {k: np.median(v["boxes"], axis=0) for k, v in info.items()}
    cls = {k: v["cls"].most_common(1)[0][0] for k, v in info.items()}
    mapping: dict[int, int] = {}
    ids = sorted(info, key=lambda k: min(info[k]["frames"]))
    for i, a in enumerate(ids):
        root_a = mapping.get(a, a)
        for b in ids[i + 1:]:
            if b in mapping or cls[a] != cls[b]:
                continue
            fa, fb = info[root_a]["frames"], info[b]["frames"]
            overlap = len(fa & fb) / min(len(fa), len(fb))
            if overlap <= max_overlap and _iou(med[root_a], med[b]) >= min_iou:
                mapping[b] = root_a
                info[root_a]["frames"] |= fb
    for rec in records:
        for d in rec["dets"]:
            d.track_id = mapping.get(d.track_id, d.track_id)
    return mapping


def part_of_other(per_track: dict, containment: float = 0.9, share: float = 0.8) -> set[int]:
    """Треки, що в ≥ ``share`` своїх кадрів лежать усередині (≥ ``containment``
    площі) рамки іншого, довшого треку того самого класу — це частина
    іншого об'єкта (напр. рука людини), а не окремий об'єкт."""
    by_frame = defaultdict(list)
    for tid, items in per_track.items():
        for f, d, _ in items:
            by_frame[f].append(d)
    length = {tid: len(items) for tid, items in per_track.items()}
    parts = set()
    for tid, items in per_track.items():
        inside = 0
        for f, d, _ in items:
            a = d.box
            area = (a[2] - a[0]) * (a[3] - a[1])
            for o in by_frame[f]:
                if o.track_id == tid or o.cls != d.cls or length[o.track_id] <= length[tid]:
                    continue
                b = o.box
                ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                if ix * iy >= containment * area:
                    inside += 1
                    break
        if inside >= share * len(items):
            parts.add(tid)
    return parts


def _contains(box, pt) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def summarize(records: list[dict], frame_wh: tuple[int, int]) -> dict:
    """Агрегує покадрові записи конвеєра в підсумкові рішення.

    Кожен запис ``records[i]`` містить: ``dets`` (список Detection),
    ``T`` (матриця «кадр → перший кадр»), ``plates`` і ``faces``
    (списки або ``None``, якщо кадр не аналізувався на номер/обличчя).
    """
    n_frames = len(records)
    merged = merge_fragments(records)
    per_track = defaultdict(list)                 # id → [(frame, Detection, T)]
    sims = defaultdict(list)                      # id → [схожість обличчя]
    plate_owner = Counter()                       # id авто → скільки разів номер у його рамці
    plate_reads = []

    for rec in records:
        for d in rec["dets"]:
            per_track[d.track_id].append((rec["frame"], d, rec["T"]))

        # Обличчя прив'язуємо до найменшої рамки person, що містить центр обличчя
        for face in rec["faces"] or []:
            owners = [d for d in rec["dets"] if d.cls == "person" and _contains(d.box, face.center)]
            if owners:
                owner = min(owners, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
                sims[owner.track_id].append(face.similarity)

        # Номер прив'язуємо до автомобіля, в рамці якого він знаходиться
        for pr in rec["plates"] or []:
            plate_reads.append(pr)
            c = ((pr.box[0] + pr.box[2]) / 2, (pr.box[1] + pr.box[3]) / 2)
            cars = [d for d in rec["dets"] if d.cls in ("car", "truck", "bus") and _contains(d.box, c)]
            if cars:
                plate_owner[min(cars, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])).track_id] += 1

    min_len = C.MIN_TRACK_FRACTION * n_frames
    parts = part_of_other(per_track)
    W, H = frame_wh
    m = C.BORDER_MARGIN
    edge_tracks = set()
    for tid, items in per_track.items():
        at_border = np.mean([d.box[0] <= m or d.box[1] <= m or d.box[2] >= W - m or d.box[3] >= H - m
                             for _, d, _ in items])
        if np.mean([d.conf for _, d, _ in items]) < 0.5 and at_border >= 0.8:
            edge_tracks.add(tid)
    tracks: list[TrackSummary] = []
    for tid, items in sorted(per_track.items()):
        if len(items) < min_len or tid in parts or tid in edge_tracks:
            continue
        cls = Counter(d.cls for _, d, _ in items).most_common(1)[0][0]
        samples = [(d.box, T) for _, d, T in items]
        score = movement_score(samples, frame_wh)
        naive = naive_movement_score(samples)
        naive_max = naive_max_movement_score(samples)
        s = sims.get(tid, [])
        tracks.append(TrackSummary(
            track_id=tid, cls=cls, n_frames=len({f for f, _, _ in items}), first_frame=items[0][0], last_frame=items[-1][0],
            mean_conf=float(np.mean([d.conf for _, d, _ in items])),
            move_score=score, move_score_naive=naive, move_score_naive_max=naive_max,
            moving=score > C.MOVING_THRESHOLD, moving_naive=naive > C.MOVING_THRESHOLD,
            face_samples=len(s), mean_similarity=float(np.mean(s)) if s else None, similarities=s))

    # Водій: найбільша середня схожість серед людей з достатньою кількістю облич
    candidates = [t for t in tracks if t.cls == "person" and t.face_samples >= C.MIN_FACE_SAMPLES]
    driver = max(candidates, key=lambda t: t.mean_similarity, default=None)
    if driver is not None and driver.mean_similarity <= C.SFACE_COSINE_THRESHOLD:
        driver = None

    plate = vote(plate_reads)
    plate_text = plate["by_char"] or plate["by_string"]
    plate_car = plate_owner.most_common(1)[0][0] if plate_owner else None

    for t in tracks:
        if t.cls == "person":
            t.role = ROLE_DRIVER if driver is not None and t.track_id == driver.track_id else ROLE_PERSON
        elif t.cls in ("car", "truck", "bus"):
            t.role = f"{ROLE_CAR_PLATE} {plate_text}" if t.track_id == plate_car else ROLE_CAR
        else:
            t.role = ROLE_EXTRA

    return {
        "n_frames": n_frames,
        "n_tracks_total": len(per_track),
        "merged_tracks": {int(k): int(v) for k, v in merged.items()},
        "part_tracks": sorted(int(t) for t in parts),
        "edge_tracks": sorted(int(t) for t in edge_tracks),
        "tracks": tracks,
        "moving_ids": [t.track_id for t in tracks if t.moving],
        "moving_ids_naive": [t.track_id for t in tracks if t.moving_naive],
        "driver_id": driver.track_id if driver else None,
        "driver_similarity": driver.mean_similarity if driver else None,
        "plate": plate,
        "plate_text": plate_text,
        "plate_car_id": plate_car,
    }


def to_jsonable(summary: dict) -> dict:
    """Готує підсумок до запису в JSON (dataclass → dict)."""
    out = dict(summary)
    out["tracks"] = [asdict(t) for t in summary["tracks"]]
    return out
