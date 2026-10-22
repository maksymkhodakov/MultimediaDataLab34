"""Кількісна оцінка результатів за еталонною розміткою синтетичної сцени.

Оскільки сцену згенеровано програмно, для кожного кадру відомі істинні
(видимі) рамки об'єктів, текст номера, хто водій, хто рухається і навіть
точний рух камери. Це дає змогу порахувати метрики, а не лише «подивитися
очима»:

* **Детекція/трекінг:** частка кадрів, де об'єкт знайдено з IoU ≥ 0.5
  (recall), середній IoU, кількість різних номерів треків на один об'єкт
  (фрагментація треку).
* **Номер:** частка аналізованих кадрів, де знак знайдено, і де OCR
  після постобробки прочитав його *точно*; правильність голосування.
* **Водій і рухомий об'єкт:** чи збігаються рішення з еталоном.
* **Рух камери:** середня похибка (у пікселях) перенесення точок кадру в
  систему координат першого кадру.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .camera_motion import CameraMotionEstimator

CLASS_OF = {"car_main": "car", "car_second": "car", "driver": "person", "passenger": "person", "dog": "dog"}
VEHICLES = {"car", "truck", "bus"}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _same_kind(det_cls: str, gt_cls: str) -> bool:
    return det_cls == gt_cls or (det_cls in VEHICLES and gt_cls == "car")


def evaluate(records: list[dict], summary: dict, gt: dict, min_visibility: float = 0.3) -> dict:
    frames_gt = gt["frames_gt"]
    n = min(len(records), len(frames_gt))

    # ---------------- 1. Відповідність трек → еталонний об'єкт ----------------
    track_iou = defaultdict(lambda: defaultdict(list))  # track → gt name → [IoU]
    per_obj = {name: {"visible": 0, "hit": 0, "ious": [], "track_ids": Counter()} for name in CLASS_OF}
    for rec, fg in zip(records[:n], frames_gt[:n]):
        objs = {o["name"]: o for o in fg["objects"]}
        for name, cls in CLASS_OF.items():
            o = objs.get(name)
            if o is None or o["visibility"] < min_visibility:
                continue
            per_obj[name]["visible"] += 1
            cands = [(iou(d.box, o["box"]), d) for d in rec["dets"] if _same_kind(d.cls, cls)]
            best = max(cands, key=lambda c: c[0], default=(0.0, None))
            if best[0] >= 0.5:
                per_obj[name]["hit"] += 1
                per_obj[name]["ious"].append(best[0])
                per_obj[name]["track_ids"][best[1].track_id] += 1
        for d in rec["dets"]:
            for name, cls in CLASS_OF.items():
                o = objs.get(name)
                if o is not None and _same_kind(d.cls, cls):
                    track_iou[d.track_id][name].append(iou(d.box, o["box"]))

    track_to_gt = {}
    for tid, by_name in track_iou.items():
        name, vals = max(by_name.items(), key=lambda kv: np.mean(kv[1]))
        if np.mean(vals) >= 0.3:
            track_to_gt[tid] = name

    detection = {
        name: {
            "frames_visible": v["visible"],
            "recall_iou50": round(v["hit"] / v["visible"], 3) if v["visible"] else None,
            "mean_iou": round(float(np.mean(v["ious"])), 3) if v["ious"] else None,
            "track_ids": dict(v["track_ids"].most_common()),
        }
        for name, v in per_obj.items()
    }

    # ---------------- 2. Номерний знак ----------------
    truth = gt["plate_text"]
    analysed = det_ok = read_ok = valid = 0
    for rec, fg in zip(records[:n], frames_gt[:n]):
        if rec["plates"] is None:
            continue
        o = next((o for o in fg["objects"] if o["name"] == "plate"), None)
        if o is None or o["visibility"] < 0.9:  # знак перекритий або поза кадром
            continue
        analysed += 1
        hits = [p for p in rec["plates"] if iou(p.box, o["box"]) >= 0.3]
        det_ok += bool(hits)
        valid += any(p.plate for p in hits)
        read_ok += any(p.plate == truth for p in hits)
    plate = {
        "truth": truth,
        "frames_analysed_plate_visible": analysed,
        "detection_rate": round(det_ok / analysed, 3) if analysed else None,
        "valid_format_rate": round(valid / analysed, 3) if analysed else None,
        "exact_read_rate": round(read_ok / analysed, 3) if analysed else None,
        "vote_by_string": summary["plate"]["by_string"],
        "vote_by_char": summary["plate"]["by_char"],
        "final_correct": summary["plate_text"] == truth,
    }

    # ---------------- 3. Водій і рухомий об'єкт ----------------
    sims_by_gt = defaultdict(list)
    for t in summary["tracks"]:
        if t.similarities and t.track_id in track_to_gt:
            sims_by_gt[track_to_gt[t.track_id]].extend(t.similarities)
    driver = {
        "predicted_track": summary["driver_id"],
        "predicted_is": track_to_gt.get(summary["driver_id"]),
        "correct": track_to_gt.get(summary["driver_id"]) == gt["driver"],
        "similarity": {k: {"mean": round(float(np.mean(v)), 3), "min": round(float(np.min(v)), 3),
                           "max": round(float(np.max(v)), 3), "n": len(v)} for k, v in sims_by_gt.items()},
    }
    pred_moving = {track_to_gt.get(i) for i in summary["moving_ids"]}
    pred_naive = {track_to_gt.get(i) for i in summary["moving_ids_naive"]}
    pred_naive_max = {track_to_gt.get(t.track_id) for t in summary["tracks"] if t.move_score_naive_max > 0.25}
    moving = {
        "truth": gt["moving_object"],
        "predicted": sorted(map(str, pred_moving)),
        "correct": pred_moving == {gt["moving_object"]},
        "naive_predicted": sorted(map(str, pred_naive)),
        "naive_correct": pred_naive == {gt["moving_object"]},
        "naive_max_predicted": sorted(map(str, pred_naive_max)),
        "naive_max_correct": pred_naive_max == {gt["moving_object"]},
    }

    # ---------------- 4. Рух камери ----------------
    W, H = gt["width"], gt["height"]
    grid = np.array([[x, y] for x in np.linspace(0, W, 9) for y in np.linspace(0, H, 6)])
    M0 = np.vstack([np.array(frames_gt[0]["M"]), [0, 0, 1]])
    errs = []
    for rec, fg in zip(records[:n], frames_gt[:n]):
        Mf = np.vstack([np.array(fg["M"]), [0, 0, 1]])
        true_T = M0 @ np.linalg.inv(Mf)  # кадр f → світ → кадр 0
        e = np.linalg.norm(CameraMotionEstimator.apply(rec["T"], grid) -
                           CameraMotionEstimator.apply(true_T, grid), axis=1)
        errs.append(float(e.mean()))
    camera = {"mean_error_px": round(float(np.mean(errs)), 2), "max_error_px": round(float(np.max(errs)), 2),
              "final_error_px": round(errs[-1], 2), "errors": [round(e, 3) for e in errs]}

    return {"track_to_gt": {int(k): v for k, v in track_to_gt.items()}, "detection": detection,
            "plate": plate, "driver": driver, "moving": moving, "camera": camera}
