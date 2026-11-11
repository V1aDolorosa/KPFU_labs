from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
import ultralytics

from ultralytics import YOLO


# -------------------- НАСТРОЙКИ --------------------

VIDEO_PATH = Path(__file__).resolve().parent / "video.mp4"
OUTPUT_FOLDER = Path(__file__).resolve().parent / "yolo_results"
WEIGHTS = "yolov8n.pt"
TARGET_NAMES = ["person", "car", "bus"]
CONFIDENCE = 0.5
THRESHOLDS = [0.2, 0.5, 0.8]
IMAGE_SIZE = 640
SAMPLE_SECONDS = [0, 3, 6, 9, 11]

OUTPUT_FOLDER.mkdir(exist_ok=True)


# -------------------- ЗАДАНИЕ 1. ПОДГОТОВКА СРЕДЫ --------------------

print("Ultralytics:", ultralytics.__version__)
print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)

if torch.cuda.is_available():
    DEVICE = 0
else:
    DEVICE = "cpu"

print("Устройство:", DEVICE)


# -------------------- ЗАДАНИЕ 2. ЗАГРУЗКА ВИДЕО --------------------

if not VIDEO_PATH.exists():
    raise FileNotFoundError("Видео video.mp4 не найдено.")

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Не удалось открыть видео.")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, first_frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Не удалось прочитать первый кадр.")

if fps <= 0:
    raise ValueError("Не удалось определить FPS.")

height, width = first_frame.shape[:2]

video_size = (width + width % 2, height + height % 2)

print("Размер видео:", width, "x", height)
print("FPS:", fps)
print("Количество кадров:", frame_count)


# Переводим секунды контрольных кадров в номера кадров

sample_frames = []

for second in SAMPLE_SECONDS:
    sample_frames.append(round(second * fps))


# -------------------- ЗАДАНИЕ 3. ЗАГРУЗКА МОДЕЛИ --------------------

model = YOLO(WEIGHTS)

# -------------------- ЗАДАНИЕ 4. ВЫБОР КЛАССОВ --------------------

name_to_id = {}

for class_id, class_name in model.names.items():
    name_to_id[class_name] = class_id

for name in TARGET_NAMES:
    if name not in name_to_id:
        raise ValueError("Такого класса нет в модели: " + name)

if len(TARGET_NAMES) < 3 or len(set(TARGET_NAMES)) != len(TARGET_NAMES):
    raise ValueError("Укажите минимум три разных класса.")

TARGET_IDS = []

for name in TARGET_NAMES:
    TARGET_IDS.append(name_to_id[name])


# -------------------- ЗАДАНИЕ 3. ОБНАРУЖЕНИЕ НА ПЕРВОМ КАДРЕ --------------------

result = model.predict(
    source=first_frame,
    conf=CONFIDENCE,
    imgsz=IMAGE_SIZE,
    device=DEVICE,
    verbose=False
)[0]

cv2.imwrite(
    str(OUTPUT_FOLDER / "first_frame_preview.jpg"),
    result.plot()
)


# -------------------- ЗАДАНИЕ 5. ОБРАБОТКА ВИДЕО --------------------

summary = []

main_detections = None
main_counts = None
main_folder = None

for threshold in THRESHOLDS:

    result_folder = OUTPUT_FOLDER / (
        "run_conf_" + str(int(threshold * 100)).zfill(3)
    )

    result_folder.mkdir(exist_ok=True)

    samples_folder = result_folder / "samples"
    samples_folder.mkdir(exist_ok=True)

    detections = []
    frame_counts = []

    writer = None

    # Размеченное видео сохраняем при основном пороге 0.5

    if threshold == CONFIDENCE:

        writer = cv2.VideoWriter(
            str(result_folder / "annotated.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            video_size
        )

        if not writer.isOpened():
            raise RuntimeError("Не удалось создать выходное видео.")

    results = model.predict(
        source=str(VIDEO_PATH),
        stream=True,
        conf=threshold,
        classes=TARGET_IDS,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False
    )

    for frame_number, result in enumerate(results):

        counts = {name: 0 for name in TARGET_NAMES}

        # Сохраняем информацию о каждом обнаруженном объекте

        for box in result.boxes:

            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            detections.append({
                "frame": frame_number,
                "class_name": class_name,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": center_x,
                "center_y": center_y
            })

            counts[class_name] += 1

        # Сохраняем количество объектов на текущем кадре

        frame_row = {"frame": frame_number}

        for name in TARGET_NAMES:
            frame_row[name] = counts[name]

        frame_counts.append(frame_row)

        # Сохраняем контрольные кадры и размеченное видео

        if frame_number in sample_frames or writer is not None:

            annotated = result.plot()

            if frame_number in sample_frames:

                sample_name = (
                    "frame_" + str(frame_number).zfill(4) + ".jpg"
                )

                cv2.imwrite(
                    str(samples_folder / sample_name),
                    annotated
                )

            if writer is not None:

                if width % 2 != 0 or height % 2 != 0:

                    annotated = cv2.copyMakeBorder(
                        annotated,
                        0,
                        height % 2,
                        0,
                        width % 2,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0)
                    )

                writer.write(annotated)

    results.close()

    if writer is not None:
        writer.release()

    # Создаём CSV-файлы

    detections_df = pd.DataFrame(
        detections,
        columns=[
            "frame", "class_name", "confidence",
            "x1", "y1", "x2", "y2",
            "center_x", "center_y"
        ]
    )

    counts_df = pd.DataFrame(frame_counts)

    detections_df.to_csv(
        result_folder / "detections.csv",
        index=False,
        encoding="utf-8-sig"
    )

    counts_df.to_csv(
        result_folder / "counts.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Запоминаем результаты основного порога

    if threshold == CONFIDENCE:

        main_detections = detections_df
        main_counts = counts_df
        main_folder = result_folder

print("\nПервые три обнаружения при conf = 0.5:")

print(
    main_detections.head(3).to_string(index=False)
)


# -------------------- ЗАДАНИЕ 6. ПОСТРОЕНИЕ ГРАФИКА --------------------

ax = main_counts.plot(
    x="frame",
    y=TARGET_NAMES,
    figsize=(10, 4),
    drawstyle="steps-post"
)

ax.set_xlabel("Номер кадра")
ax.set_ylabel("Количество объектов")
ax.set_title("Количество объектов при conf = 0.5")

ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    main_folder / "counts_over_time.png"
)

plt.show()


# -------------------- ЗАДАНИЕ 7. СРАВНЕНИЕ ПОРОГОВ --------------------

for threshold in THRESHOLDS:

    result_folder = OUTPUT_FOLDER / (
        "run_conf_" + str(int(threshold * 100)).zfill(3)
    )

    counts_df = pd.read_csv(result_folder / "counts.csv")

    for name in TARGET_NAMES:

        summary.append({
            "conf": threshold,
            "class_name": name,
            "mean_per_frame": round(
                float(counts_df[name].mean()), 3
            ),
            "max_per_frame": int(counts_df[name].max())
        })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUTPUT_FOLDER / "threshold_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nСравнение порогов:")
print(summary_df.to_string(index=False))


# -------------------- ДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ. BYTETRACK --------------------

# Отслеживаем автомобили и считаем пересечение вертикальной линии

line_x = width // 2

last_side = {}
crossed_ids = set()

tracking_video = OUTPUT_FOLDER / "tracking_line.mp4"

track_writer = cv2.VideoWriter(
    str(tracking_video),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    video_size
)

if not track_writer.isOpened():
    raise RuntimeError("Не удалось создать видео отслеживания.")

track_results = model.track(
    source=str(VIDEO_PATH),
    stream=True,
    persist=True,
    tracker="bytetrack.yaml",
    conf=CONFIDENCE,
    classes=[name_to_id["car"]],
    imgsz=IMAGE_SIZE,
    device=DEVICE,
    verbose=False
)

for result in track_results:

    boxes = result.boxes

    track_ids = []
    coordinates = []

    if boxes.id is not None:

        track_ids = boxes.id.int().cpu().tolist()
        centers_x = boxes.xywh[:, 0].cpu().tolist()
        coordinates = boxes.xyxy.cpu().tolist()

        # Проверяем, пересёк ли автомобиль линию

        for track_id, center_x in zip(track_ids, centers_x):

            if center_x < line_x:
                side = -1
            else:
                side = 1

            if track_id in last_side:

                if last_side[track_id] != side:
                    crossed_ids.add(track_id)

            last_side[track_id] = side

    # Рисуем рамки обнаруженных автомобилей

    annotated = result.plot(labels=False)

    # Рисуем вертикальную линию

    cv2.line(
        annotated,
        (line_x, 0),
        (line_x, height - 1),
        (0, 0, 255),
        2
    )

    cv2.putText(
        annotated,
        "COUNTING LINE",
        (line_x + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # Подписываем автомобили их ID

    for track_id, box in zip(track_ids, coordinates):

        x1, y1, x2, y2 = map(int, box)

        cv2.putText(
            annotated,
            "car ID: " + str(track_id),
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Отображаем счётчик пересечений

    cv2.putText(
        annotated,
        "Crossings: " + str(len(crossed_ids)),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # Подготавливаем кадр для сохранения

    if width % 2 != 0 or height % 2 != 0:

        annotated = cv2.copyMakeBorder(
            annotated,
            0,
            height % 2,
            0,
            width % 2,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

    track_writer.write(annotated)

track_results.close()
track_writer.release()

print("\nКоличество траекторий с пересечением:", len(crossed_ids))
print("ID траекторий:", crossed_ids)
print("Видео с линией:", tracking_video)



print("\nРезультаты сохранены в:", OUTPUT_FOLDER.resolve())
