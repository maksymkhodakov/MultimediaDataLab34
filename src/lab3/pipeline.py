"""Повний конвеєр обробки відео.

Схема (для кожного кадру):

    кадр ─┬─► YOLO11m + ByteTrack ─► об'єкти з номерами треків
          ├─► оптичний потік + RANSAC ─► рух камери (кадр → перший кадр)
          └─► (кожен PROCESS_EVERY-й кадр)
                ├─► YOLOv11-LP ─► рамка номера ─► збільшення, CLAHE ─► EasyOCR ─► шаблон UA
                └─► YuNet ─► вирівнювання ─► SFace ─► схожість з еталоном водія

Після проходу по відео — агрегування по треках (:mod:`analysis`),
анотоване відео (:mod:`annotate`) і, якщо є еталонна розмітка, оцінка
якості (:mod:`evaluation`).
"""
from __future__ import annotations

import csv
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .analysis import summarize, to_jsonable
from .annotate import write_annotated
from .camera_motion import CameraMotionEstimator
from .detection import ObjectTracker
from .evaluation import evaluate
from .faces import FaceIdentifier
from .plates import PlateReader
from .report_figures import build_all

SNAPSHOT_FRAMES = (60, 300, 380, 520, 700, 900)


def process_video(video: Path, driver_photo: Path, verbose: bool = True) -> tuple[list[dict], dict]:
    """Перший прохід: покадрова обробка. Повертає (записи по кадрах, метадані відео)."""
    tracker = ObjectTracker()
    plates = PlateReader()
    faces = FaceIdentifier()
    faces.set_reference(driver_photo)
    camera = CameraMotionEstimator()

    cap = cv2.VideoCapture(str(video))
    meta = {"fps": cap.get(cv2.CAP_PROP_FPS), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    records, f, t0 = [], 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = tracker.update(frame)
        T = camera.update(frame, [d.box for d in dets])
        rec = {"frame": f, "t": f / meta["fps"], "dets": dets, "T": T.copy(), "plates": None, "faces": None}
        if f % C.PROCESS_EVERY == 0:
            rec["plates"] = plates.read(frame)
            rec["faces"] = faces.process(frame)
        records.append(rec)
        f += 1
        if verbose and f % 100 == 0:
            print(f"  оброблено {f}/{meta['frames']} кадрів ({f / (time.time() - t0):.1f} кадр/с)")
    cap.release()
    meta["processing_s"] = time.time() - t0
    meta["camera_inliers_mean"] = float(np.mean(camera.inliers[1:])) if len(camera.inliers) > 1 else 0.0
    meta["reference_face"] = faces.reference_face
    return records, meta


def save_frame_log(records: list[dict], path: Path) -> None:
    """Зберігає всі покадрові результати у CSV (об'єкти, номери, обличчя)."""
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t", "kind", "track_id", "cls", "conf", "x1", "y1", "x2", "y2", "text_raw",
                    "plate", "similarity"])
        for r in records:
            for d in r["dets"]:
                w.writerow([r["frame"], f"{r['t']:.3f}", "object", d.track_id, d.cls, f"{d.conf:.3f}",
                            *[f"{v:.1f}" for v in d.box], "", "", ""])
            for p in r["plates"] or []:
                w.writerow([r["frame"], f"{r['t']:.3f}", "plate", "", "plate", f"{p.det_conf:.3f}",
                            *[f"{v:.1f}" for v in p.box], p.raw_text, p.plate or "", ""])
            for fc in r["faces"] or []:
                w.writerow([r["frame"], f"{r['t']:.3f}", "face", "", "face", f"{fc.score:.3f}",
                            *[f"{v:.1f}" for v in fc.box], "", "", f"{fc.similarity:.4f}"])


def print_summary(video: Path, summary: dict) -> None:
    """Текстовий підсумок у стилі демонстрації."""
    print(f"\nПІДСУМОК ОБРОБКИ ВІДЕО {video.name}\n" + "-" * 70)
    print(f"Треків всього: {summary['n_tracks_total']}, об'єднано фрагментів: {summary['merged_tracks']}, "
          f"частини інших об'єктів: {summary['part_tracks']}, крайові: {summary['edge_tracks']}, "
          f"залишилось: {len(summary['tracks'])}")
    print(f"{'трек':>4} {'клас':8} {'кадрів':>6} {'впевн.':>6} {'рух*':>6} {'наїв.1':>7} {'наїв.2':>7} "
          f"{'рухається':>9}  {'схожість з водієм':22} роль")
    for t in summary["tracks"]:
        sim = f"{t.mean_similarity:.3f} (у {t.face_samples} кадрах)" if t.mean_similarity is not None else "-"
        print(f"{t.track_id:>4} {t.cls:8} {t.n_frames:>6} {t.mean_conf:6.2f} {t.move_score:6.2f} "
              f"{t.move_score_naive:7.2f} {t.move_score_naive_max:7.2f} {'так' if t.moving else 'ні':>9}  "
              f"{sim:22} {t.role}")
    print("* рух — з компенсацією руху камери; наїв.1 — як у демонстрації (перша→остання поява);")
    print("  наїв.2 — максимальне зміщення без компенсації; поріг", C.MOVING_THRESHOLD)
    p = summary["plate"]
    print(f"\nНомерний знак: голоси {p['counts']}")
    print(f"  за рядком: {p['by_string']}, посимвольно: {p['by_char']} -> {summary['plate_text']}")
    if summary["driver_id"] is not None:
        print(f"Водій: трек #{summary['driver_id']}, середня схожість {summary['driver_similarity']:.3f} "
              f"> {C.SFACE_COSINE_THRESHOLD}")
    else:
        print("Водія не ідентифіковано")


RECORDS_CACHE = C.DATA_DIR / "records.pkl"


def run_pipeline(video: Path = C.SCENE_VIDEO, driver_photo: Path = C.DRIVER_REFERENCE,
                 gt_path: Path | None = C.SCENE_GT, verbose: bool = True, reuse: bool = False) -> dict:
    """Запускає всі кроки і зберігає результати в ``data/`` та ``report/figures/``.

    ``reuse=True`` бере покадрові результати нейромереж з кешу
    (``data/records.pkl``) і перераховує лише аналіз — зручно, коли
    змінюються пороги чи правила агрегування.
    """
    C.DATA_DIR.mkdir(exist_ok=True)
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if reuse and RECORDS_CACHE.exists():
        records, meta = pickle.loads(RECORDS_CACHE.read_bytes())
        if verbose:
            print(f"Покадрові результати взято з кешу {RECORDS_CACHE.name}")
    else:
        if verbose:
            print(f"Обробка {video} ...")
        records, meta = process_video(video, driver_photo, verbose)
        RECORDS_CACHE.write_bytes(pickle.dumps((records, meta)))
    if verbose:
        print(f"Оброблено {len(records)} кадрів за {meta['processing_s']:.0f} с "
              f"({len(records) / meta['processing_s']:.1f} кадр/с)")

    summary = summarize(records, (meta["width"], meta["height"]))
    write_annotated(video, C.ANNOTATED_VIDEO, records, summary, SNAPSHOT_FRAMES, C.FIGURES_DIR)
    save_frame_log(records, C.FRAMES_CSV)
    if meta["reference_face"] is not None:
        cv2.imwrite(str(C.FIGURES_DIR / "reference_face_aligned.png"), meta["reference_face"])

    result = {"video": str(video.name), "meta": {k: v for k, v in meta.items() if k != "reference_face"},
              "summary": to_jsonable(summary)}
    gt = None
    if gt_path is not None and gt_path.exists():
        gt = json.loads(gt_path.read_text())
        if gt["video"] == video.name:
            result["evaluation"] = evaluate(records, summary, gt)
        else:
            gt = None
    C.RESULTS_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float))
    figures = build_all(video, records, summary, result, gt, driver_photo)
    if verbose:
        print("Ілюстрації:", ", ".join(f.name for f in figures))

    if verbose:
        print_summary(video, summary)
        if "evaluation" in result:
            ev = result["evaluation"]
            print("\nОЦІНКА ЗА ЕТАЛОННОЮ РОЗМІТКОЮ")
            for name, d in ev["detection"].items():
                print(f"  {name:11} recall@IoU0.5={d['recall_iou50']}  mean IoU={d['mean_iou']}  треки={d['track_ids']}")
            print(f"  номер: {ev['plate']}")
            print(f"  водій: {ev['driver']['predicted_is']} (правильно: {ev['driver']['correct']})")
            print(f"  рухомий: {ev['moving']['predicted']} (правильно: {ev['moving']['correct']}); "
                  f"наїв.1: {ev['moving']['naive_predicted']}; наїв.2: {ev['moving']['naive_max_predicted']}")
            print(f"  похибка оцінки руху камери: середня {ev['camera']['mean_error_px']} px, "
                  f"макс. {ev['camera']['max_error_px']} px")
        print(f"\nАнотоване відео: {C.ANNOTATED_VIDEO}")
    return {"records": records, "summary": summary, "meta": meta, "result": result}
