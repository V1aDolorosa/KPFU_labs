import os
import glob
import json
import math
import gc

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

DATA_DIR = "dataset"
TEST_IMAGES_DIR = "test_images"

IMG_SIZE = 128
VAL_SPLIT = 0.2

# None — использовать все изображения каждого класса.
# При большом датасете можно установить, например, 200.
MAX_PER_CLASS = None

EPOCHS = 12
BATCH_SIZE = 32
RANDOM_STATE = 42

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

tf.keras.utils.set_random_seed(RANDOM_STATE)


# ============================================================
# 2. ЗАГРУЗКА ИЗОБРАЖЕНИЙ
# ============================================================

def load_images(data_dir, classes, img_size=128, max_per_class=None):
    X, y, filepaths = [], [], []

    for label, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)

        files = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, name))
            and name.lower().endswith(IMAGE_EXTENSIONS)
        ]

        files = sorted(files)

        if max_per_class is not None:
            files = files[:max_per_class]

        for filepath in files:
            img = cv2.imread(filepath)

            if img is None:
                print("Не удалось прочитать:", filepath)
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))

            X.append(img)
            y.append(label)
            filepaths.append(filepath)

    if not X:
        raise ValueError(
            f"В папке {data_dir} не найдено читаемых изображений."
        )

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int64)

    return X, y, filepaths


# ============================================================
# 3. ПОИСК КЛАССОВ И ОПИСАНИЕ ДАТАСЕТА
# ============================================================

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"Папка {DATA_DIR} не найдена. "
        "Создай её и размести внутри папки с изображениями марок."
    )

classes = sorted([
    name
    for name in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, name))
])

print("=" * 60)
print("CNN — КЛАССИФИКАЦИЯ МАРОК АВТОМОБИЛЕЙ")
print("=" * 60)

print("\nНайдены классы:", classes)

if len(classes) < 3:
    raise ValueError("Для задания требуется минимум три класса (марки).")


# ============================================================
# 4. АУГМЕНТАЦИЯ ДАННЫХ
# ============================================================

def build_augmentation(strength="standard"):
    if strength == "strong":
        # Эксперимент 2: усиливаем аугментацию.
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
            layers.RandomTranslation(0.05, 0.05),
        ])

    # Базовая аугментация из исходного кода.
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])


# ============================================================
# 5. ПОСТРОЕНИЕ CNN
# ============================================================

def build_simple_cnn(img_size, num_classes, augmentation="standard"):
    data_augmentation = build_augmentation(augmentation)

    model = models.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),

        data_augmentation,

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 6. ДЕМОНСТРАЦИЯ ПРИМЕРОВ ИЗ ДАТАСЕТА
# ============================================================

def show_examples(X_train, y_train, classes):
    indices = []
    rng = np.random.default_rng(RANDOM_STATE)

    for cls_idx in range(len(classes)):
        class_indices = np.where(y_train == cls_idx)[0]

        if len(class_indices) > 0:
            indices.append(rng.choice(class_indices))

    if not indices:
        return

    grid = math.ceil(math.sqrt(len(indices)))

    plt.figure(figsize=(8, 8))

    for i, idx in enumerate(indices, start=1):
        plt.subplot(grid, grid, i)
        plt.imshow(X_train[idx])
        plt.title(classes[y_train[idx]])
        plt.axis("off")

    plt.suptitle("Примеры изображений обучающей выборки")
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. ОБУЧЕНИЕ И ДВА ЭКСПЕРИМЕНТА
# ============================================================

EXPERIMENTS = [
    {
        "name": "Базовая модель",
        "img_size": IMG_SIZE,
        "epochs": EPOCHS,
        "augmentation": "standard"
    },
    {
        "name": "Эксперимент 1: IMG_SIZE = 160",
        "img_size": 160,
        "epochs": EPOCHS,
        "augmentation": "standard"
    },
    {
        "name": "Эксперимент 2: усиленная аугментация",
        "img_size": IMG_SIZE,
        "epochs": 18,
        "augmentation": "strong"
    }
]

results = []
histories = []

best_model = None
best_accuracy = -1.0
best_img_size = IMG_SIZE
best_experiment = ""
best_y_val = None
best_y_pred = None

X = None
y = None
filepaths = None
current_img_size = None
class_counts = None

examples_shown = False


for experiment in EXPERIMENTS:
    exp_name = experiment["name"]
    exp_img_size = experiment["img_size"]
    exp_epochs = experiment["epochs"]
    exp_augmentation = experiment["augmentation"]

    print("\n" + "=" * 60)
    print(exp_name)
    print("=" * 60)

    # Если размер изображения изменился, загружаем датасет заново.
    # Это позволяет не хранить одновременно несколько версий
    # всего датасета в оперативной памяти.

    if current_img_size != exp_img_size:
        if X is not None:
            del X, y, filepaths
            gc.collect()

        X, y, filepaths = load_images(
            DATA_DIR,
            classes,
            img_size=exp_img_size,
            max_per_class=MAX_PER_CLASS
        )

        current_img_size = exp_img_size

        # Описываем датасет при первой загрузке.
        if class_counts is None:
            class_counts = np.bincount(
                y,
                minlength=len(classes)
            )

            print("\nКоличество изображений по классам:")

            for cls, count in zip(classes, class_counts):
                print(f"  {cls}: {count}")

            print("Всего изображений:", len(X))
            print("Размер изображений:", X.shape[1:])

            if np.any(class_counts < 2):
                raise ValueError(
                    "В каждом классе должно быть минимум "
                    "два читаемых изображения."
                )

            val_count = math.ceil(len(y) * VAL_SPLIT)

            if val_count < len(classes):
                raise ValueError(
                    "Недостаточно изображений для стратифицированного "
                    "разделения. Добавь изображения в каждый класс."
                )

            if len(y) - val_count < len(classes):
                raise ValueError(
                    "Слишком мало изображений для обучающей выборки."
                )

    # Разделение сохраняет одинаковое соотношение классов.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VAL_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)

    if not examples_shown:
        show_examples(X_train, y_train, classes)
        examples_shown = True

    # Одинаковое начальное состояние для сравнения экспериментов.
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    model = build_simple_cnn(
        img_size=exp_img_size,
        num_classes=len(classes),
        augmentation=exp_augmentation
    )

    if exp_name == "Базовая модель":
        model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=exp_epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Оценка модели.
    val_loss, val_acc = model.evaluate(
        X_val,
        y_val,
        verbose=0
    )

    y_pred = np.argmax(
        model.predict(X_val, verbose=0),
        axis=1
    )

    epochs_run = len(history.history["accuracy"])

    print(f"\nЭксперимент: {exp_name}")
    print(f"Val loss: {val_loss:.4f}")
    print(f"Val accuracy: {val_acc:.4f}")
    print(f"Выполнено эпох: {epochs_run}")

    results.append({
        "experiment": exp_name,
        "img_size": exp_img_size,
        "augmentation": exp_augmentation,
        "epochs_requested": exp_epochs,
        "epochs_run": epochs_run,
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss)
    })

    histories.append({
        "name": exp_name,
        "accuracy": list(history.history["accuracy"]),
        "val_accuracy": list(history.history["val_accuracy"])
    })

    # Сохраняем модель с лучшей валидационной точностью.
    if val_acc > best_accuracy:
        best_accuracy = float(val_acc)
        best_model = model
        best_img_size = exp_img_size
        best_experiment = exp_name
        best_y_val = y_val.copy()
        best_y_pred = y_pred.copy()

    del X_train, X_val, y_train, y_val
    gc.collect()


# ============================================================
# 8. СРАВНЕНИЕ ЭКСПЕРИМЕНТОВ
# ============================================================

print("\n" + "=" * 60)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 60)

results_df = pd.DataFrame(results)

print(
    results_df[
        [
            "experiment",
            "img_size",
            "epochs_run",
            "val_accuracy",
            "val_loss"
        ]
    ].to_string(index=False)
)

print("\nЛучшая модель:", best_experiment)
print(f"Лучшая Val accuracy: {best_accuracy:.4f}")
print("Размер входного изображения:", best_img_size)


# Кривые точности всех экспериментов.
plt.figure(figsize=(10, 6))

for item in histories:
    epochs_range = range(1, len(item["accuracy"]) + 1)

    plt.plot(
        epochs_range,
        item["accuracy"],
        label=f'{item["name"]} — train'
    )

    plt.plot(
        epochs_range,
        item["val_accuracy"],
        linestyle="--",
        label=f'{item["name"]} — validation'
    )

plt.xlabel("Эпоха")
plt.ylabel("Accuracy")
plt.title("Сравнение точности моделей")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 9. ОТЧЁТ КЛАССИФИКАЦИИ И МАТРИЦА ОШИБОК
# ============================================================

print("\n" + "=" * 60)
print("ОТЧЁТ ДЛЯ ЛУЧШЕЙ МОДЕЛИ")
print("=" * 60)

print(
    classification_report(
        best_y_val,
        best_y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        zero_division=0
    )
)

cm = confusion_matrix(
    best_y_val,
    best_y_pred,
    labels=np.arange(len(classes))
)

plt.figure(figsize=(7, 6))

im = plt.imshow(
    cm,
    interpolation="nearest",
    cmap="Blues"
)

plt.title("Матрица ошибок")
plt.colorbar(im, fraction=0.046, pad=0.04)

tick_marks = np.arange(len(classes))

plt.xticks(
    tick_marks,
    classes,
    rotation=45,
    ha="right"
)

plt.yticks(tick_marks, classes)

# Подписываем число примеров в каждой ячейке.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.xlabel("Предсказано")
plt.ylabel("Истина")
plt.tight_layout()
plt.show()


# ============================================================
# 10. СОХРАНЕНИЕ МОДЕЛИ И РЕЗУЛЬТАТОВ
# ============================================================

EXPORT_DIR = "export"
os.makedirs(EXPORT_DIR, exist_ok=True)

model_path = os.path.join(
    EXPORT_DIR,
    "cars_simple_cnn.keras"
)

best_model.save(model_path)

with open(
    os.path.join(EXPORT_DIR, "classes.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        classes,
        f,
        ensure_ascii=False,
        indent=2
    )

with open(
    os.path.join(EXPORT_DIR, "model_info.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "best_experiment": best_experiment,
            "img_size": best_img_size,
            "val_accuracy": best_accuracy,
            "classes": classes
        },
        f,
        ensure_ascii=False,
        indent=2
    )

results_df.to_json(
    os.path.join(EXPORT_DIR, "experiment_results.json"),
    orient="records",
    force_ascii=False,
    indent=2
)

print("\nМодель и результаты сохранены в папку export:")
print(os.listdir(EXPORT_DIR))


# ============================================================
# 11. ПРОВЕРКА НА ПРОИЗВОЛЬНЫХ ФОТОГРАФИЯХ
# ============================================================

os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

test_files = sorted([
    os.path.join(TEST_IMAGES_DIR, name)
    for name in os.listdir(TEST_IMAGES_DIR)
    if os.path.isfile(os.path.join(TEST_IMAGES_DIR, name))
    and name.lower().endswith(IMAGE_EXTENSIONS)
])

if not test_files:
    print(
        f"\nПапка {TEST_IMAGES_DIR} пуста. "
        "Положи туда три фотографии, отсутствующие в обучающей выборке, "
        "и запусти скрипт повторно."
    )

else:
    print("\n" + "=" * 60)
    print("ПРОВЕРКА НА ПРОИЗВОЛЬНЫХ ФОТО")
    print("=" * 60)

    tested = 0

    for fname in test_files:
        if tested >= 3:
            break

        img = cv2.imread(fname)

        if img is None:
            print("Не удалось прочитать:", fname)
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(
            img_rgb,
            (best_img_size, best_img_size)
        )

        img_normalized = (
            img_resized.astype(np.float32) / 255.0
        )

        probs = best_model.predict(
            np.expand_dims(img_normalized, axis=0),
            verbose=0
        )[0]

        pred_idx = int(np.argmax(probs))
        pred_cls = classes[pred_idx]
        conf = float(np.max(probs))

        print(f"\nФайл: {os.path.basename(fname)}")
        print(f"Предсказанная марка: {pred_cls}")
        print(f"Уверенность модели: {conf:.2%}")

        plt.figure(figsize=(6, 5))
        plt.imshow(img_rgb)
        plt.title(
            f"Предсказание: {pred_cls}\n"
            f"Уверенность: {conf:.2%}"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        tested += 1

    if tested < 3:
        print(
            f"\nУспешно проверено фотографий: {tested} из 3. "
            "Проверь содержимое папки test_images."
        )

print("\nРабота программы завершена.")
