"""Точка входу лабораторної роботи №3.

Приклади запуску (з кореня проєкту):

    python run_lab3.py scene      # згенерувати синтетичне відео + еталонну розмітку
    python run_lab3.py process    # обробити відео: об'єкти, номер, водій
    python run_lab3.py all        # обидва кроки поспіль
    python run_lab3.py process --reuse  # лише перерахувати аналіз (нейромережі — з кешу)
    python run_lab3.py process --video my_video.mp4 --driver my_driver.jpg  # власне відео
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lab3 import config as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Лабораторна робота №3: обробка відео")
    ap.add_argument("step", choices=["scene", "process", "all"])
    ap.add_argument("--video", type=Path, default=C.SCENE_VIDEO, help="вхідне відео")
    ap.add_argument("--driver", type=Path, default=C.DRIVER_REFERENCE, help="еталонне фото водія")
    ap.add_argument("--reuse", action="store_true", help="взяти результати нейромереж з кешу")
    args = ap.parse_args()

    if args.step in ("scene", "all"):
        from lab3.scene import build_scene
        meta = build_scene(preview_dir=C.FIGURES_DIR)
        print(f"Сцену збережено: {C.SCENE_VIDEO} ({meta['frames']} кадрів, {meta['duration_s']} с)")

    if args.step in ("process", "all"):
        from lab3.pipeline import run_pipeline
        run_pipeline(args.video, args.driver, reuse=args.reuse)


if __name__ == "__main__":
    main()
