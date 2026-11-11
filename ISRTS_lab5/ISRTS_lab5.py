from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
import ultralytics

from ultralytics import YOLO


# -------------------- НАСТРОЙКИ --------------------

VIDEO_PATH = Path(__file__).resolve().parent / "video.mp4"
WEIGHTS = "yolov8n.pt"
TARGET_NAMES = ["person", "car", "bus"]
CONFIDENCE = 0.5
THRESHOLDS = [0.2, 0.5, 0.8]
IMAGE_SIZE = 640
# Кадры для ручной проверки
SAMPLE_TIMES = [0, 3, 6, 9, 11]
# Дополнительное задание
RUN_TRACKING = True
TRACK_CLASS = "car"

OUTPUT_FOLDER = Path(__file__).resolve().parent / "yolo_results"



OUTPUT_FOLDER.mkdir(exist_ok=True)

print("Ultralytics:", ultralytics.__version__)
print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)

if torch.cuda.is_available():
    DEVICE = 0
else:
    DEVICE = "cpu"

print("Устройство:", DEVICE)


if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        "Видео не найдено. Положите его рядом со скриптом "
        "и назовите video.mp4."
    )

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
    raise ValueError("Не удалось определить FPS видео.")

height, width = first_frame.shape[:2]

print("Размер кадра:", width, "x", height)
print("FPS:", fps)
print("Количество кадров:", frame_count)
print("Длительность:", round(frame_count / fps, 2), "секунд")


# -------------------- 2. ЗАГРУЗКА МОДЕЛИ --------------------

model = YOLO(WEIGHTS)

print("Классы модели:")
print(model.names)

name_to_id = {}

for class_id, class_name in model.names.items():
    name_to_id[class_name] = class_id

for name in TARGET_NAMES:
    if name not in name_to_id:
        raise ValueError("Такого класса нет в модели: " + name)

if len(TARGET_NAMES) < 3:
    raise ValueError("Нужно выбрать минимум три класса.")

if len(set(TARGET_NAMES)) != len(TARGET_NAMES):
    raise ValueError("Названия классов не должны повторяться.")

TARGET_IDS = []

for name in TARGET_NAMES:
    TARGET_IDS.append(name_to_id[name])

print("Выбранные классы:", TARGET_NAMES)
print("ID классов:", TARGET_IDS)


# -------------------- 3. ПРОВЕРКА ПЕРВОГО КАДРА --------------------

result = model.predict(
    source=first_frame,
    conf=CONFIDENCE,
    imgsz=IMAGE_SIZE,
    device=DEVICE,
    verbose=False
)[0]

# Сохраняем изображение с найденными объектами

preview = result.plot()

cv2.imwrite(
    str(OUTPUT_FOLDER / "first_frame_preview.jpg"),
    preview
)

print("Обнаружения на первом кадре:")

for box in result.boxes:

    class_id = int(box.cls.item())
    class_name = model.names[class_id]
    confidence = float(box.conf.item())

    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    print("Класс:", class_name)
    print("Confidence:", round(confidence, 3))

    print(
        "Рамка:",
        round(x1, 1),
        round(y1, 1),
        round(x2, 1),
        round(y2, 1)
    )

    print(
        "Центр:",
        round(center_x, 1),
        round(center_y, 1)
    )

    print("--------------------")


# -------------------- 4. ОБРАБОТКА ВИДЕО --------------------

sample_frames = []

for second in SAMPLE_TIMES:
    sample_frames.append(round(second * fps))

summary = []


for threshold in THRESHOLDS:

    print("\nОбработка видео. Порог:", threshold)

    folder_name = "run_conf_" + str(int(threshold * 100)).zfill(3)

    result_folder = OUTPUT_FOLDER / folder_name
    result_folder.mkdir(exist_ok=True)

    samples_folder = result_folder / "samples"
    samples_folder.mkdir(exist_ok=True)

    detections = []
    frame_counts = []

    save_video = (threshold == CONFIDENCE)

    writer = None

    if save_video:

        output_video = result_folder / "annotated.mp4"

        output_width = width + width % 2
        output_height = height + height % 2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(output_video),
            fourcc,
            fps,
            (output_width, output_height)
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

        counts = {}

        for name in TARGET_NAMES:
            counts[name] = 0
        for box in result.boxes:

            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

            center_x
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

        # Сохраняем количество объектов на каждом кадре

        frame_row = {
            "frame": frame_number,
            "time_s": frame_number / fps
        }

        for name in TARGET_NAMES:
            frame_row[name] = counts[name]

        frame_counts.append(frame_row)

        # Сохраняем пять кадров для ручной проверки

        if frame_number in sample_frames:

            sample_image = result.plot()

            sample_name = (
                "frame_" + str(frame_number).zfill(4) + ".jpg"
            )

            cv2.imwrite(
                str(samples_folder / sample_name),
                sample_image
            )

        # Записываем видео с рамками

        if save_video:

            annotated = result.plot()

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

    # Создаём таблицу обнаружений

    detections_df = pd.DataFrame(
        detections,
        columns=[
            "frame", "class_name", "confidence",
            "x1", "y1", "x2", "y2", "center_x", "center_y"
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

    print("Обработано кадров:", len(counts_df))
    print("Всего обнаружений:", len(detections_df))

    for name in TARGET_NAMES:

        summary.append({
            "conf": threshold,
            "class_name": name,
            "frames": len(counts_df),
            "detections_sum": int(counts_df[name].sum()),
            "mean_per_frame": round(
                float(counts_df[name].mean()), 3
            ),
            "max_per_frame": int(counts_df[name].max())
        })

    if threshold == CONFIDENCE:

        main_detections = detections_df
        main_counts = counts_df
        main_folder = result_folder


# -------------------- 5. СРАВНЕНИЕ ПОРОГОВ --------------------

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUTPUT_FOLDER / "threshold_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nСравнение порогов:")
print(summary_df)


# -------------------- 6. ГРАФИК И РЕЗУЛЬТАТЫ --------------------

print("\nПервые три обнаружения при conf = 0.5:")
print(main_detections.head(3))

print("\nСреднее и максимальное количество объектов:")

print(
    main_counts[TARGET_NAMES].agg(["mean", "max"])
)

ax = main_counts.plot(
    x="time_s",
    y=TARGET_NAMES,
    figsize=(10, 4),
    drawstyle="steps-post"
)

ax.set_xlabel("Время видео, секунд")
ax.set_ylabel("Количество объектов в кадре")
ax.set_title("Количество объектов при conf = 0.5")

ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    main_folder / "counts_over_time.png"
)

plt.show()


# -------------------- 7. ДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ: BYTETRACK --------------------

if RUN_TRACKING:

    if TRACK_CLASS not in name_to_id:
        raise ValueError("Выбранного класса нет в модели.")

    track_model = YOLO(WEIGHTS)

    # Вертикальная линия посередине кадра

    line_x = width / 2

    last_side = {}
    crossed_ids = set()

    track_results = track_model.track(
        source=str(VIDEO_PATH),
        stream=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        classes=[name_to_id[TRACK_CLASS]],
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False
    )

    for result in track_results:

        boxes = result.boxes

        if boxes.id is not None:

            track_ids = boxes.id.int().cpu().tolist()
            centers_x = boxes.xywh[:, 0].cpu().tolist()

            for track_id, center_x in zip(track_ids, centers_x):

                if center_x < line_x:
                    side = -1
                else:
                    side = 1

                if track_id in last_side:

                    if last_side[track_id] != side:
                        crossed_ids.add(track_id)

                last_side[track_id] = side

    track_results.close()

    print("\nКласс для отслеживания:", TRACK_CLASS)

    print(
        "Количество траекторий с пересечением:",
        len(crossed_ids)
    )

    print("ID траекторий:", crossed_ids)


print("\nОбработка завершена.")

print(
    "Результаты находятся в папке:",
    OUTPUT_FOLDER.resolve()
)
