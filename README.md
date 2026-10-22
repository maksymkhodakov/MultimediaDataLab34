# Лабораторна робота №3-4
## "Методи обробки мультимедійної інформації"
## Ходаков Максим Олегович ШІ-2

### Завдання 3. Обробка відео: об'єкти, номер автомобіля, ідентифікація водія

Конвеєр обробки 40-секундного відео (1280×720, 25 кадр/с), що:

* виділяє та розпізнає об'єкти й відстежує їх між кадрами — **YOLO11m (COCO) + ByteTrack**;
* визначає єдиний рухомий об'єкт з **компенсацією руху камери** (оптичний потік Лукаса–Канаде + RANSAC);
* знаходить і читає номер автомобіля — **YOLOv11-LP + EasyOCR** + шаблон українського номера + голосування по кадрах;
* ідентифікує водія за окремим еталонним фото — **YuNet + SFace**.

Відео синтетичне (дозволено викладачем) і зібране з вільних фото (див. `assets/ATTRIBUTION.md`); для нього
зберігається еталонна розмітка, тож якість оцінюється кількісно.

### Структура

```
assets/                     вихідні фото для сцени + ATTRIBUTION.md
models/                     ваги моделей (не в git, див. нижче)
src/lab3/
  config.py                 шляхи, моделі, пороги
  plate_render.py           рендер номера KA 7429 MB за ДСТУ 4278:2019 і перенесення гомографією
  scene.py                  генерація синтетичного відео + еталонна розмітка
  detection.py              YOLO11m + ByteTrack
  camera_motion.py          оцінка руху камери (кадр → перший кадр)
  plates.py                 детектор номерів, EasyOCR, нормалізація, голосування
  faces.py                  YuNet + SFace
  analysis.py               рішення по треках: рухомий об'єкт, водій, номер
  annotate.py               анотоване відео
  evaluation.py             метрики за еталонною розміткою
  report_figures.py         графіки для звіту
  pipeline.py               увесь конвеєр
run_lab3.py                 CLI: scene | process | all
notebooks/Lab3_video_processing.ipynb   покрокова демонстрація з результатами
report/                     звіт (.docx) та ілюстрації
data/                       згенероване відео та результати (не в git)
```

### Запуск

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# ваги моделей
mkdir -p models && cd models
curl -LO https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt
curl -LO https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m-seg.pt
curl -LO https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1s.pt
curl -LO https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -LO https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
cd ..

.venv/bin/python run_lab3.py all          # згенерувати відео й обробити його
.venv/bin/python run_lab3.py process --video my.mp4 --driver my_driver.jpg   # власне відео
```

На Apple M1 обробка 1000 кадрів займає ≈ 5 хв (≈ 3.3 кадр/с). На інших машинах змініть `DEVICE` у `config.py`.

### Результат

| Пункт | Результат |
|---|---|
| Об'єкти | автомобіль з номером, другий автомобіль, водій, пасажир, собака (recall@IoU0.5 0.86–1.00) |
| Рухомий об'єкт | пасажир (оцінка 1.42 при порозі 0.25; нерухомі ≤ 0.06) |
| Номер | **KA7429MB** — точно прочитаний у 100 % кадрів з відкритим знаком |
| Водій | трек водія, середня схожість SFace 0.707 (пасажир — 0.115, поріг 0.363) |
