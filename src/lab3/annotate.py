"""Побудова анотованого відео з підсумковими рішеннями.

На кожен кадр наносяться: рамки об'єктів з номерами треків і класами,
позначки ``MOVING`` (рухомий об'єкт) і ``DRIVER`` (водій), рамка номерного
знака з прочитаним (після голосування) номером, рамки облич зі схожістю, а
також траєкторія рухомого об'єкта. Траєкторія зберігається в координатах
першого кадру і щоразу переноситься в поточний кадр оберненою матрицею
руху камери — тому вона «приклеєна» до сцени, а не до екрана.

Написи — англійською: стандартні шрифти OpenCV (Hershey) не мають кирилиці.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from .camera_motion import CameraMotionEstimator

COLORS = {  # BGR
    "person": (255, 160, 40), "car": (60, 200, 255), "truck": (60, 200, 255), "bus": (60, 200, 255),
    "dog": (200, 90, 255), "cat": (200, 90, 255),
}
GREEN, RED, YELLOW, WHITE, BLACK = (60, 220, 60), (50, 50, 240), (0, 230, 255), (255, 255, 255), (0, 0, 0)


def _label(img, text, x, y, color, scale=0.6):
    """Підпис на кольоровій плашці (читається на будь-якому фоні)."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    y = max(y, th + 6)
    cv2.rectangle(img, (int(x), int(y - th - 6)), (int(x + tw + 6), int(y + base - 2)), color, -1)
    cv2.putText(img, text, (int(x + 3), int(y - 4)), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 2, cv2.LINE_AA)


def draw_frame(frame: np.ndarray, rec: dict, summary: dict, trail_ref: list) -> np.ndarray:
    """Малює всі позначки на одному кадрі (повертає новий кадр)."""
    img = frame.copy()
    moving, driver_id = set(summary["moving_ids"]), summary["driver_id"]
    kept = {t.track_id for t in summary["tracks"]}

    for d in rec["dets"]:
        if d.track_id not in kept:
            continue
        x1, y1, x2, y2 = map(int, d.box)
        color = COLORS.get(d.cls, WHITE)
        thick = 2
        tags = []
        if d.track_id == driver_id:
            color, thick = GREEN, 4
            tags.append("DRIVER")
        if d.track_id in moving:
            color, thick = RED, 4
            tags.append("MOVING")
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        _label(img, f"#{d.track_id} {d.cls} {d.conf:.2f} " + " ".join(tags), x1, y1, color)

        if d.track_id in moving:  # точка траєкторії в координатах першого кадру
            trail_ref.append(CameraMotionEstimator.apply(rec["T"], [d.center])[0])

    # Траєкторія рухомого об'єкта: з координат першого кадру — у поточний кадр
    if len(trail_ref) > 1:
        pts = CameraMotionEstimator.apply(np.linalg.inv(rec["T"]), np.array(trail_ref))
        cv2.polylines(img, [pts.astype(np.int32)], False, RED, 3, cv2.LINE_AA)

    for pr in rec.get("plates_draw") or []:
        x1, y1, x2, y2 = map(int, pr.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), YELLOW, 3)
        _label(img, f"PLATE {summary['plate_text'] or '?'}", x1, y2 + 26, YELLOW, 0.75)

    for f in rec.get("faces_draw") or []:
        x1, y1, x2, y2 = map(int, f.box)
        c = GREEN if f.similarity > 0.363 else WHITE
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        _label(img, f"sim {f.similarity:.2f}", x1, y2 + 22, c, 0.5)

    # Інформаційна панель
    panel = [f"frame {rec['frame']}  t={rec['t']:.1f}s",
             f"plate: {summary['plate_text'] or '-'}",
             f"driver: #{driver_id} (sim {summary['driver_similarity']:.2f})" if driver_id else "driver: -",
             f"moving: {', '.join('#' + str(i) for i in summary['moving_ids']) or '-'}"]
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (330, 20 + 26 * len(panel)), BLACK, -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    for i, line in enumerate(panel):
        cv2.putText(img, line, (20, 36 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)
    return img


def write_annotated(video_in: Path, video_out: Path, records: list[dict], summary: dict,
                    snapshot_frames=(), snapshot_dir: Path | None = None) -> None:
    """Другий прохід по відео: малює позначки і записує H.264-файл.

    Номер і обличчя аналізуються лише в кожному ``PROCESS_EVERY``-му
    кадрі; щоб рамки не «блимали», в інших кадрах малюємо останні знайдені.
    """
    cap = cv2.VideoCapture(str(video_in))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = imageio_ffmpeg.write_frames(str(video_out), (W, H), fps=fps, codec="libx264",
                                         pix_fmt_out="yuv420p", quality=None, bitrate="6M",
                                         macro_block_size=1)
    writer.send(None)
    trail: list = []
    last_plates, last_faces = [], []
    for rec in records:
        ok, frame = cap.read()
        if not ok:
            break
        if rec["plates"] is not None:
            last_plates, last_faces = rec["plates"], rec["faces"]
        rec = dict(rec, plates_draw=last_plates, faces_draw=last_faces)
        out = draw_frame(frame, rec, summary, trail)
        writer.send(cv2.cvtColor(out, cv2.COLOR_BGR2RGB).tobytes())
        if snapshot_dir is not None and rec["frame"] in snapshot_frames:
            cv2.imwrite(str(snapshot_dir / f"annotated_{rec['frame']:04d}.jpg"), out)
    writer.close()
    cap.release()
