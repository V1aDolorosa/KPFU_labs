
import os
import json
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

SEED = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
EXPORT_DIR = BASE_DIR / "export"

IMG_SIZE = 128
VAL_SPLIT = 0.20
EPOCHS = 20
BATCH_SIZE = 16

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

EXPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CNN — КЛАССИФИКАЦИЯ МАРОК АВТОМОБИЛЕЙ")
print("=" * 60)
print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
print("Папка датасета:", DATA_DIR)


# ============================================================
# 2. ЗАГРУЗКА И ПРОВЕРКА ДАТАСЕТА
# ============================================================

def collect_dataset():
    """
    Считывает пути к изображениям.
    Название папки считается названием класса.
    """

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Папка с датасетом не найдена: {DATA_DIR}"
        )

    class_dirs = sorted(
        folder
        for folder in DATA_DIR.iterdir()
        if folder.is_dir()
    )

    classes = [folder.name for folder in class_dirs]

    if len(classes) < 3:
        raise ValueError(
            f"Найдено только {len(classes)} классов. "
            "Необходимо минимум 3."
        )

    image_paths = []
    labels = []

    print("\nКоличество изображений:")

    for label, folder in enumerate(class_dirs):
        count = 0

        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # Проверяем, что изображение действительно читается.
            image = cv2.imread(str(path))

            if image is None:
                print(f"Пропущен повреждённый файл: {path}")
                continue

            image_paths.append(str(path))
            labels.append(label)
            count += 1

        print(f"  {folder.name}: {count} фото")

        if count < 2:
            raise ValueError(
                f"В классе {folder.name} недостаточно изображений."
            )

    labels = np.asarray(labels, dtype=np.int64)

    if len(image_paths) == 0:
        raise ValueError("В датасете нет изображений.")

    print(f"\nВсего изображений: {len(image_paths)}")
    print("Классы:", classes)

    return np.asarray(image_paths), labels, classes


def load_images(paths, img_size):
    """
    Загружает изображения, меняет размер и нормализует
    значения пикселей в диапазон [0, 1].
    """

    images = []

    for path in paths:
        image = cv2.imread(str(path))

        if image is None:
            raise RuntimeError(
                f"Не удалось прочитать изображение: {path}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (img_size, img_size),
            interpolation=cv2.INTER_AREA
        )

        images.append(image)

    X = np.asarray(images, dtype=np.float32) / 255.0

    return X


# ============================================================
# 3. ВИЗУАЛИЗАЦИЯ ПРИМЕРОВ ДАТАСЕТА
# ============================================================

def save_dataset_examples(train_paths, y_train, classes):
    fig, axes = plt.subplots(
        1, len(classes),
        figsize=(5 * len(classes), 5)
    )

    axes = np.atleast_1d(axes)

    for class_index, class_name in enumerate(classes):
        indices = np.where(y_train == class_index)[0]

        if len(indices) == 0:
            axes[class_index].axis("off")
            continue

        path = train_paths[indices[0]]
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        axes[class_index].imshow(image)
        axes[class_index].set_title(class_name)
        axes[class_index].axis("off")

    fig.suptitle("Примеры изображений обучающей выборки")
    fig.tight_layout()

    fig.savefig(
        EXPORT_DIR / "dataset_examples.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 4. АУГМЕНТАЦИЯ И АРХИТЕКТУРА CNN
# ============================================================

def build_cnn(img_size, num_classes, use_batch_norm=False):
    """
    Создаёт простую сверточную нейронную сеть.

    use_batch_norm=False — базовая сеть.
    use_batch_norm=True  — вариант с BatchNormalization.
    """

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation"
    )

    inputs = tf.keras.layers.Input(
        shape=(img_size, img_size, 3)
    )

    x = augmentation(inputs)

    for filters in (32, 64, 128):
        if use_batch_norm:
            x = tf.keras.layers.Conv2D(
                filters,
                kernel_size=(3, 3),
                padding="same",
                use_bias=False
            )(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)

        else:
            x = tf.keras.layers.Conv2D(
                filters,
                kernel_size=(3, 3),
                padding="same",
                activation="relu"
            )(x)

        x = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        )(x)

    # Уменьшает число параметров по сравнению с Flatten.
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 5. ОБУЧЕНИЕ И ГРАФИКИ
# ============================================================

def save_learning_curves(history, experiment_name):
    """Сохраняет графики accuracy и loss."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        history.history["accuracy"],
        label="Train accuracy"
    )

    axes[0].plot(
        history.history["val_accuracy"],
        label="Validation accuracy"
    )

    axes[0].set_title("Точность обучения")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        history.history["loss"],
        label="Train loss"
    )

    axes[1].plot(
        history.history["val_loss"],
        label="Validation loss"
    )

    axes[1].set_title("Функция потерь")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(f"Кривые обучения: {experiment_name}")
    fig.tight_layout()

    fig.savefig(
        EXPORT_DIR / f"learning_curve_{experiment_name}.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def train_experiments(train_paths, val_paths,
                      y_train, y_val, classes):
    """
    Проводит три эксперимента и выбирает модель
    с наибольшей точностью на валидационной выборке.
    """

    experiments = [
        {
            "key": "baseline_128",
            "title": "Базовая CNN, 128x128",
            "img_size": 128,
            "batch_norm": False,
        },
        {
            "key": "imgsize_160",
            "title": "CNN, 160x160",
            "img_size": 160,
            "batch_norm": False,
        },
        {
            "key": "batchnorm_128",
            "title": "CNN с BatchNormalization",
            "img_size": 128,
            "batch_norm": True,
        },
    ]

    results = []
    best_result = None

    for experiment in experiments:
        name = experiment["key"]
        img_size = experiment["img_size"]
        use_batch_norm = experiment["batch_norm"]

        print("\n" + "=" * 60)
        print("ЭКСПЕРИМЕНТ:", experiment["title"])
        print("=" * 60)

        # Единые данные для честного сравнения экспериментов.
        X_train = load_images(train_paths, img_size)
        X_val = load_images(val_paths, img_size)

        tf.keras.utils.set_random_seed(SEED)

        model = build_cnn(
            img_size=img_size,
            num_classes=len(classes),
            use_batch_norm=use_batch_norm
        )

        model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=4,
                restore_best_weights=True,
                mode="max",
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=0.00001,
                verbose=1
            ),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            shuffle=True,
            verbose=1
        )

        val_loss, val_accuracy = model.evaluate(
            X_val, y_val, verbose=0
        )

        y_pred = np.argmax(
            model.predict(X_val, verbose=0),
            axis=1
        )

        save_learning_curves(history, name)

        result = {
            "experiment": experiment["title"],
            "name": name,
            "img_size": img_size,
            "batch_norm": use_batch_norm,
            "val_accuracy": float(val_accuracy),
            "val_loss": float(val_loss),
            "epochs_trained": len(history.history["loss"]),
        }

        results.append(result)

        print(f"\nValidation accuracy: {val_accuracy:.4f}")
        print(f"Validation loss:     {val_loss:.4f}")

        if (
            best_result is None
            or val_accuracy > best_result["val_accuracy"]
        ):
            best_result = {
                **result,
                "model": model,
                "y_pred": y_pred,
            }

        del X_train, X_val

    return best_result, results


# ============================================================
# 6. МАТРИЦА ОШИБОК И ОТЧЁТ
# ============================================================

def evaluate_best_model(best_result, y_val, classes):
    y_pred = best_result["y_pred"]

    print("\n" + "=" * 60)
    print("ЛУЧШАЯ МОДЕЛЬ")
    print("=" * 60)

    print("Эксперимент:", best_result["experiment"])
    print(f"Validation accuracy: {best_result['val_accuracy']:.4f}")

    report = classification_report(
        y_val,
        y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        zero_division=0
    )

    print("\nClassification report:")
    print(report)

    with open(
        EXPORT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            f"Лучшая модель: {best_result['experiment']}\n"
        )
        file.write(
            f"Validation accuracy: "
            f"{best_result['val_accuracy']:.4f}\n\n"
        )
        file.write(report)

    cm = confusion_matrix(
        y_val,
        y_pred,
        labels=np.arange(len(classes))
    )

    cm_df = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    cm_df.index.name = "Истинный класс"
    cm_df.to_csv(
        EXPORT_DIR / "confusion_matrix.csv",
        encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_title("Матрица ошибок")
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))

    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="black"
            )

    fig.tight_layout()

    fig.savefig(
        EXPORT_DIR / "confusion_matrix.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_experiment_comparison(results):
    df = pd.DataFrame(results)

    df.to_csv(
        EXPORT_DIR / "experiment_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nСравнение экспериментов:")
    print(
        df[
            ["experiment", "val_accuracy", "val_loss", "epochs_trained"]
        ].to_string(index=False)
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        df["experiment"],
        df["val_accuracy"]
    )

    ax.set_title("Сравнение точности моделей")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        EXPORT_DIR / "experiment_comparison.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 7. СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ
# ============================================================

def save_best_model(best_result, classes):
    model_path = EXPORT_DIR / "cars_simple_cnn.keras"

    best_result["model"].save(str(model_path))

    with open(
        EXPORT_DIR / "classes.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            classes,
            file,
            ensure_ascii=False,
            indent=2
        )

    model_info = {
        "img_size": best_result["img_size"],
        "experiment": best_result["experiment"],
        "validation_accuracy": best_result["val_accuracy"],
        "seed": SEED,
    }

    with open(
        EXPORT_DIR / "model_info.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            model_info,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\nМодель сохранена:", model_path)
    print("Метки классов сохранены:", EXPORT_DIR / "classes.json")
    print("Настройки модели сохранены:", EXPORT_DIR / "model_info.json")


# ============================================================
# 8. ПРОВЕРКА НА ПРОИЗВОЛЬНЫХ ФОТОГРАФИЯХ
# ============================================================

def predict_images(image_paths, model, classes, img_size):
    """
    Предсказывает марки для изображений, которых нет
    в обучающей или валидационной выборке.
    """

    for number, path in enumerate(image_paths, start=1):
        path = Path(str(path).strip().strip("\"'")).expanduser()

        if not path.is_file():
            print(f"\nФайл не найден: {path}")
            continue

        image = cv2.imread(str(path))

        if image is None:
            print(f"\nНе удалось прочитать изображение: {path}")
            continue

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        resized = cv2.resize(
            image_rgb,
            (img_size, img_size),
            interpolation=cv2.INTER_AREA
        )

        X = np.asarray(
            resized,
            dtype=np.float32
        ) / 255.0

        X = np.expand_dims(X, axis=0)

        probabilities = model.predict(
            X,
            verbose=0
        )[0]

        pred_index = int(np.argmax(probabilities))
        predicted_class = classes[pred_index]
        confidence = float(probabilities[pred_index])

        print("\n" + "-" * 50)
        print("Фото:", path.name)
        print("Предсказанная марка:", predicted_class)
        print(f"Уверенность модели: {confidence:.2%}")
        print("Вероятности по классам:")

        for index in np.argsort(probabilities)[::-1]:
            print(
                f"  {classes[index]}: "
                f"{probabilities[index]:.2%}"
            )

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.imshow(image_rgb)
        ax.set_title(
            f"Предсказание: {predicted_class}\n"
            f"Уверенность: {confidence:.2%}"
        )
        ax.axis("off")

        fig.tight_layout()

        fig.savefig(
            EXPORT_DIR / f"prediction_{number}.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)


def load_saved_model():
    model_path = EXPORT_DIR / "cars_simple_cnn.keras"
    classes_path = EXPORT_DIR / "classes.json"
    info_path = EXPORT_DIR / "model_info.json"

    for path in (model_path, classes_path, info_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Не найден файл {path}. "
                "Сначала запусти обучение командой python main.py."
            )

    model = tf.keras.models.load_model(str(model_path))

    with open(classes_path, encoding="utf-8") as file:
        classes = json.load(file)

    with open(info_path, encoding="utf-8") as file:
        model_info = json.load(file)

    return model, classes, model_info


# ============================================================
# 9. ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Обучение CNN для распознавания марок автомобилей"
    )

    parser.add_argument(
        "--predict",
        nargs="+",
        metavar="IMAGE",
        help="Проверить сохранённую модель на указанных фотографиях"
    )

    args = parser.parse_args()

    # Режим инференса без повторного обучения.
    if args.predict:
        model, classes, info = load_saved_model()

        predict_images(
            args.predict,
            model,
            classes,
            info["img_size"]
        )
        return

    # Читаем датасет.
    paths, labels, classes = collect_dataset()

    # Единое разделение для всех экспериментов.
    train_paths, val_paths, y_train, y_val = train_test_split(
        paths,
        labels,
        test_size=VAL_SPLIT,
        random_state=SEED,
        stratify=labels
    )

    print("\nОбучающая выборка:", len(train_paths))
    print("Валидационная выборка:", len(val_paths))

    save_dataset_examples(
        train_paths,
        y_train,
        classes
    )

    # Три эксперимента.
    best_result, results = train_experiments(
        train_paths,
        val_paths,
        y_train,
        y_val,
        classes
    )

    # Оценка лучшей модели.
    evaluate_best_model(
        best_result,
        y_val,
        classes
    )

    # Сравнение экспериментов.
    save_experiment_comparison(results)

    # Сохранение модели.
    save_best_model(
        best_result,
        classes
    )

    print("\nВсе результаты сохранены в:", EXPORT_DIR)

    # Интерактивная проверка трёх новых фотографий.
    print("\n" + "=" * 60)
    print("ПРОВЕРКА НА НОВЫХ ФОТОГРАФИЯХ")
    print("Введи путь к фото. Enter — завершить ввод.")
    print("=" * 60)

    test_paths = []

    for i in range(1, 4):
        try:
            path = input(
                f"Путь к фотографии {i}/3: "
            ).strip().strip("\"'")
        except EOFError:
            break

        if not path:
            break

        test_paths.append(path)

    if test_paths:
        predict_images(
            test_paths,
            best_result["model"],
            classes,
            best_result["img_size"]
        )
    else:
        print(
            "\nПроверку фотографий можно выполнить отдельно "
            "командой:\n"
            "python main.py --predict /путь/к/фото1.jpg "
            "/путь/к/фото2.jpg /путь/к/фото3.jpg"
        )


if __name__ == "__main__":
    main()

