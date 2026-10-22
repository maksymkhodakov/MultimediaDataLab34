"""Конфігурація лабораторної роботи №3 «Обробка відео».

Усі шляхи, назви моделей і порогові значення зібрано в одному місці, щоб
модулі конвеєра не містили «магічних чисел», а звіт міг посилатися на
конкретні параметри.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Шляхи проєкту
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # корінь репозиторію
ASSETS_DIR = ROOT / "assets"                # вихідні фото для синтетичної сцени
MODELS_DIR = ROOT / "models"                # ваги нейромереж
DATA_DIR = ROOT / "data"                    # згенероване відео та результати
FIGURES_DIR = ROOT / "report" / "figures"   # ілюстрації для звіту

SCENE_VIDEO = DATA_DIR / "synthetic_scene.mp4"            # вхідне відео (30–60 с)
SCENE_GT = DATA_DIR / "synthetic_scene_ground_truth.json"  # еталонна розмітка сцени
ANNOTATED_VIDEO = DATA_DIR / "synthetic_scene_annotated.mp4"
RESULTS_JSON = DATA_DIR / "results.json"
FRAMES_CSV = DATA_DIR / "frame_log.csv"

# Еталонне фото водія — окреме фото (інший рік, одяг, освітлення), не кадр із відео.
DRIVER_REFERENCE = ASSETS_DIR / "driver_reference_2016.jpg"

# --------------------------------------------------------------------------
# Моделі (усі — відомі попередньо навчені мережі)
# --------------------------------------------------------------------------
COCO_MODEL = MODELS_DIR / "yolo11m.pt"                      # YOLO11m, COCO (80 класів)
SEG_MODEL = MODELS_DIR / "yolo11m-seg.pt"                   # YOLO11m-seg — лише для побудови сцени
PLATE_MODEL = MODELS_DIR / "license-plate-finetune-v1s.pt"  # YOLOv11s, донавчена на номерних знаках
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"   # детектор облич YuNet
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"  # дескриптор облич SFace

# Класи COCO, які нас цікавлять за умовою завдання (автомобілі, люди, тварини, рюкзаки…)
TARGET_CLASSES = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "dog", "cat", "backpack", "handbag", "suitcase",
}

# --------------------------------------------------------------------------
# Параметри обробки
# --------------------------------------------------------------------------
DET_CONF = 0.35          # мінімальна впевненість детектора об'єктів
TRACKER_CFG = "bytetrack.yaml"  # ByteTrack, вбудований в Ultralytics
PROCESS_EVERY = 2        # номер і обличчя аналізуємо в кожному 2-му кадрі

PLATE_CONF = 0.40        # мінімальна впевненість детектора номерів
PLATE_UPSCALE = 3        # у скільки разів збільшуємо вирізаний номер перед OCR
# Шаблон українського номера: 2 літери, 4 цифри, 2 літери (лише латинські
# «двійники» кириличних літер, дозволені ДСТУ 4278:2019: A B C E H I K M O P T X).
UA_PLATE_LETTERS = "ABCEHIKMOPTX"
UA_PLATE_REGEX = rf"[{UA_PLATE_LETTERS}]{{2}}\d{{4}}[{UA_PLATE_LETTERS}]{{2}}"

FACE_SCORE = 0.80        # поріг впевненості YuNet
SFACE_COSINE_THRESHOLD = 0.363  # поріг косинусної схожості SFace (рекомендація авторів моделі)
MIN_FACE_SAMPLES = 5     # мінімум облич на треку, щоб робити висновок про особу

MIN_TRACK_FRACTION = 0.10  # треки, коротші за 10 % відео, вважаємо шумом
MOVING_THRESHOLD = 0.25    # зміщення > 25 % розміру об'єкта (після компенсації руху камери)
BORDER_MARGIN = 4          # рамки, що торкаються краю кадру (обрізані), не беремо для оцінки руху

DEVICE = "mps"  # Apple Silicon GPU; на інших машинах — "cuda" або "cpu"
