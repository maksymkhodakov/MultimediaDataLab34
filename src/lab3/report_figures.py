"""Побудова ілюстрацій для звіту та ноутбука (matplotlib → PNG у report/figures).

Кольори — фіксований категоріальний порядок (синій, помаранчевий,
бірюзовий, жовтий, пурпуровий); пороги — сірі штрихові лінії. Кожен
графік має одну вісь значень; різні за масштабом величини — на окремих
панелях.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402
from .camera_motion import CameraMotionEstimator  # noqa: E402
from .faces import FaceIdentifier  # noqa: E402
from .plates import PlateReader  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, THR = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "lines.linewidth": 2,
})


def _save(fig, name: str) -> Path:
    path = C.FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _track_color(summary) -> dict:
    """Колір закріплений за треком (а не за рангом) — однаковий на всіх графіках."""
    return {t.track_id: SERIES[i % len(SERIES)] for i, t in enumerate(summary["tracks"])}


def _label(t) -> str:
    cls_uk = {"dog": "собака", "cat": "кіт", "backpack": "рюкзак", "handbag": "сумка", "bicycle": "велосипед"}
    names = {"ВОДІЙ": "водій", "інша людина (пасажир)": "пасажир", "додатковий об'єкт": cls_uk.get(t.cls, t.cls),
             "інший автомобіль": "другий авто"}
    role = names.get(t.role, "авто з номером" if t.role.startswith("автомобіль") else t.role)
    return f"#{t.track_id} {role}"


# --------------------------------------------------------------------------
def fig_confidence(records, summary) -> Path:
    """Впевненість детектора по кадрах для кожного треку."""
    col = _track_color(summary)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for t in summary["tracks"]:
        xs, ys = [], []
        for r in records:
            d = next((d for d in r["dets"] if d.track_id == t.track_id), None)
            xs.append(r["t"])
            ys.append(d.conf if d else np.nan)
        ax.plot(xs, ys, color=col[t.track_id], lw=1.6, label=_label(t))
    ax.set_xlabel("час, с")
    ax.set_ylabel("впевненість YOLO11m")
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    return _save(fig, "fig_confidence.png")


def fig_similarity(records, summary) -> Path:
    """Косинусна схожість облич з еталоном водія в часі (по треках людей)."""
    col = _track_color(summary)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    owners = {}
    for r in records:
        for f in r["faces"] or []:
            people = [d for d in r["dets"] if d.cls == "person" and d.box[0] <= f.center[0] <= d.box[2]
                      and d.box[1] <= f.center[1] <= d.box[3]]
            if people:
                o = min(people, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
                owners.setdefault(o.track_id, []).append((r["t"], f.similarity))
    for t in summary["tracks"]:
        if t.track_id in owners:
            xs, ys = zip(*owners[t.track_id])
            ax.plot(xs, ys, "o", ms=3, color=col[t.track_id], alpha=0.8,
                    label=f"{_label(t)} (середнє {np.mean(ys):.3f})")
    ax.axhline(C.SFACE_COSINE_THRESHOLD, color=THR, ls="--", lw=1.4)
    ax.text(0.3, C.SFACE_COSINE_THRESHOLD + 0.02, f"поріг SFace {C.SFACE_COSINE_THRESHOLD}", color=INK2, fontsize=9)
    ax.set_xlabel("час, с")
    ax.set_ylabel("косинусна схожість")
    ax.set_ylim(-0.2, 1.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
    return _save(fig, "fig_similarity.png")


def fig_movement(summary) -> Path:
    """Три оцінки руху по треках — на окремих панелях (різний масштаб)."""
    tracks = summary["tracks"]
    col = _track_color(summary)
    labels = [_label(t) for t in tracks]
    measures = [("move_score", "з компенсацією руху камери\n(наш метод)"),
                ("move_score_naive", "без компенсації: перша → остання\nпоява (як у демонстрації)"),
                ("move_score_naive_max", "без компенсації:\nмаксимальне зміщення")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (key, title) in zip(axes, measures):
        vals = [getattr(t, key) for t in tracks]
        lim = 1.6
        ax.barh(range(len(vals)), np.minimum(vals, lim), color=[col[t.track_id] for t in tracks], height=0.6)
        for i, v in enumerate(vals):
            ax.text(min(v, lim) + 0.03, i, f"{v:.2f}" + (" →" if v > lim else ""), va="center", fontsize=9,
                    color=INK)
        ax.axvline(C.MOVING_THRESHOLD, color=THR, ls="--", lw=1.4)
        ax.set_xlim(0, lim + 0.45)
        ax.set_title(title, fontsize=10, color=INK)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    axes[1].set_xlabel(f"зміщення / розмір об'єкта (поріг {C.MOVING_THRESHOLD})")
    return _save(fig, "fig_movement.png")


def fig_camera(records, gt: dict, evaluation: dict) -> Path:
    """Оцінений і справжній масштаб кадру відносно першого кадру + похибка."""
    t = [r["t"] for r in records]
    est = [1 / np.sqrt(abs(np.linalg.det(r["T"][:2, :2]))) for r in records]
    true = [fg["camera"]["scale"] / gt["frames_gt"][0]["camera"]["scale"] for fg in gt["frames_gt"][:len(records)]]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(t, true, color=SERIES[0], lw=3, alpha=0.45, label="справжнє (з розмітки сцени)")
    a1.plot(t, est, color=SERIES[1], lw=1.4, label="оцінка: оптичний потік + RANSAC")
    a1.set_ylabel("наближення камери\nвідносно 1-го кадру")
    a1.legend(loc="upper left", fontsize=9)
    a2.plot(t, evaluation["camera"]["errors"], color=SERIES[2], lw=1.4)
    a2.set_ylabel("похибка, px")
    a2.set_xlabel("час, с")
    return _save(fig, "fig_camera.png")


def fig_trajectory(records, summary, first_frame: np.ndarray) -> Path:
    """Центри всіх треків у координатах першого кадру поверх першого кадру."""
    col = _track_color(summary)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB), alpha=0.55)
    for t in summary["tracks"]:
        pts = []
        for r in records:
            d = next((d for d in r["dets"] if d.track_id == t.track_id), None)
            if d is not None:
                pts.append(CameraMotionEstimator.apply(r["T"], [d.center])[0])
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], "-", color=col[t.track_id], lw=2 if t.moving else 1,
                label=_label(t) + (" — рухається" if t.moving else ""))
        ax.plot(pts[:, 0], pts[:, 1], "o", ms=2, color=col[t.track_id])
    ax.set_xlim(-120, first_frame.shape[1] + 40)
    ax.set_ylim(first_frame.shape[0] + 30, -30)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9)
    return _save(fig, "fig_trajectory.png")


def fig_plate_examples(video: Path, records, frames=(60, 300, 520, 800)) -> Path:
    """Вирізаний знак → підготовлене зображення → сирий текст OCR → номер."""
    reader = PlateReader()
    cap = cv2.VideoCapture(str(video))
    rows = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        reads = reader.read(frame)
        if not reads:
            continue
        pr = reads[0]
        x1, y1, x2, y2 = map(int, pr.box)
        rows.append((f, frame[y1:y2, x1:x2], reader.preprocess(frame, pr.box), pr))
    cap.release()
    fig, axes = plt.subplots(len(rows), 2, figsize=(9, 1.45 * len(rows)),
                             gridspec_kw={"width_ratios": [1, 1.4]})
    for (f, crop, prep, pr), (a, b) in zip(rows, axes):
        a.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        a.set_title(f"кадр {f}: рамка {crop.shape[1]}×{crop.shape[0]} px, conf {pr.det_conf:.2f}", fontsize=9)
        b.imshow(prep, cmap="gray")
        b.set_title(f"OCR: «{pr.raw_text}» → {pr.plate} (виправлень: {pr.corrections})", fontsize=9)
        for ax in (a, b):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    return _save(fig, "fig_plate_examples.png")


def fig_faces(video: Path, records, summary, driver_photo: Path) -> Path:
    """Еталонне обличчя і вирівняні обличчя з відео зі значеннями схожості."""
    fi = FaceIdentifier()
    fi.set_reference(driver_photo)
    cap = cv2.VideoCapture(str(video))
    wanted = {"driver": summary["driver_id"], "other": next((t.track_id for t in summary["tracks"]
                                                             if t.cls == "person" and t.track_id != summary["driver_id"]), None)}
    crops = {"driver": [], "other": []}
    for r in records[::40]:
        if r["faces"] is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, frame = cap.read()
        for face in fi.detect(frame):
            cx, cy = face[0] + face[2] / 2, face[1] + face[3] / 2
            for key, tid in wanted.items():
                d = next((d for d in r["dets"] if d.track_id == tid), None)
                if d and d.box[0] <= cx <= d.box[2] and d.box[1] <= cy <= d.box[3] and len(crops[key]) < 4:
                    aligned = fi.recognizer.alignCrop(frame, face)
                    crops[key].append((r["frame"], aligned, fi.similarity(fi.recognizer.feature(aligned))))
    cap.release()
    n = 1 + len(crops["driver"]) + len(crops["other"])
    fig, axes = plt.subplots(1, n, figsize=(1.55 * n, 2.2))
    items = [("еталон\n(фото 2016 р.)", fi.reference_face, None)] + \
            [(f"водій, кадр {f}", im, s) for f, im, s in crops["driver"]] + \
            [(f"пасажир, кадр {f}", im, s) for f, im, s in crops["other"]]
    for ax, (title, im, s) in zip(axes, items):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title + ("" if s is None else f"\nsim = {s:.3f}"), fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    return _save(fig, "fig_faces.png")


def fig_pipeline() -> Path:
    """Схема конвеєра обробки."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                                       fc=color + "22", ec=color, lw=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6, color=INK)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))

    box(0.1, 1.9, 1.4, 0.8, "Відео\n1280×720, 25 к/с\n40 с", SERIES[6])
    box(2.0, 3.5, 2.3, 0.85, "YOLO11m (COCO)\n+ ByteTrack", SERIES[0])
    box(2.0, 2.35, 2.3, 0.85, "Оптичний потік LK\n+ RANSAC (рух камери)", SERIES[2])
    box(2.0, 1.2, 2.3, 0.85, "YOLOv11-LP → CLAHE\n→ EasyOCR (CRAFT+CRNN)", SERIES[3])
    box(2.0, 0.05, 2.3, 0.85, "YuNet → вирівнювання\n→ SFace (128-D)", SERIES[4])
    for y in (3.92, 2.77, 1.62, 0.47):
        arrow(1.5, 2.3, 2.0, y)
    box(4.9, 3.5, 2.4, 0.85, "Треки: об'єднання\nфрагментів, фільтрація", SERIES[0])
    box(4.9, 2.35, 2.4, 0.85, "Координати 1-го кадру\n→ оцінка руху об'єкта", SERIES[2])
    box(4.9, 1.2, 2.4, 0.85, "Шаблон LLDDDDLL\n+ голосування по кадрах", SERIES[3])
    box(4.9, 0.05, 2.4, 0.85, "Обличчя → трек person\nсередня схожість", SERIES[4])
    for y in (3.92, 2.77, 1.62, 0.47):
        arrow(4.3, y, 4.9, y)
    arrow(4.3, 3.92, 4.9, 2.95)
    box(7.9, 3.5, 3.0, 0.85, "Об'єкти та їх класи\n(авто, люди, собака)", SERIES[0])
    box(7.9, 2.35, 3.0, 0.85, "Рухомий об'єкт", SERIES[2])
    box(7.9, 1.2, 3.0, 0.85, "Номер автомобіля\n+ прив'язка до треку авто", SERIES[3])
    box(7.9, 0.05, 3.0, 0.85, "Ідентифікований водій", SERIES[4])
    for y in (3.92, 2.77, 1.62, 0.47):
        arrow(7.3, y, 7.9, y)
    return _save(fig, "fig_pipeline.png")


def build_all(video: Path, records, summary, result: dict, gt: dict | None, driver_photo: Path) -> list[Path]:
    """Будує всі ілюстрації; повертає список шляхів."""
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    _, first = cap.read()
    cap.release()
    out = [fig_pipeline(), fig_confidence(records, summary), fig_similarity(records, summary),
           fig_movement(summary), fig_trajectory(records, summary, first),
           fig_plate_examples(video, records), fig_faces(video, records, summary, driver_photo)]
    if gt is not None and "evaluation" in result:
        out.append(fig_camera(records, gt, result["evaluation"]))
    return out
