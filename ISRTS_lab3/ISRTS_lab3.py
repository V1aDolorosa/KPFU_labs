from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np


# ========== ЗАГРУЗКА МОДЕЛЕЙ ==========

model_n = YOLO("yolov8n.pt")
model_s = YOLO("yolov8s.pt")


# ========== ЗАГРУЗКА ПЕРВОГО ИЗОБРАЖЕНИЯ ==========

image_url = "https://i.pinimg.com/originals/bf/4c/c9/bf4cc984740f7588454f61fd33da26ba.jpg"

headers = {"User-Agent": "Mozilla/5.0"}

resp = requests.get(image_url, headers=headers, timeout=15)
resp.raise_for_status()

img = Image.open(BytesIO(resp.content)).convert("RGB")
img_first_array = np.array(img)

plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis("off")
plt.title("Исходное изображение")
plt.show()


# ========== ЗАДАНИЕ 1: ТОЛЬКО ЛЮДИ ==========

print("\n========== ЗАДАНИЕ 1: ТОЛЬКО ЛЮДИ ==========")

res_people_first = model_n.predict(
    source=img_first_array,
    conf=0.3,
    iou=0.45,
    classes=[0],
    verbose=False
)[0]

num_persons = len(res_people_first.boxes)

print(
    f"Людей найдено: {num_persons}"
    if num_persons
    else "Людей не найдено"
)

plt.figure(figsize=(10, 8))
plt.imshow(res_people_first.plot())
plt.axis("off")
plt.title("Детекция людей")
plt.show()


# ========== ЗАДАНИЕ 2: ЛЮДИ И КОШКИ ==========

print("\n========== ЗАДАНИЕ 2: ЛЮДИ И КОШКИ ==========")

res2 = model_n.predict(
    source=img_first_array,
    conf=0.3,
    iou=0.45,
    classes=[0, 15],
    verbose=False
)[0]

people_count = sum(
    int(cls) == 0 for cls in res2.boxes.cls
)

cats_count = sum(
    int(cls) == 15 for cls in res2.boxes.cls
)

print(f"Людей найдено: {people_count}")
print(f"Кошек найдено: {cats_count}")

plt.figure(figsize=(10, 8))
plt.imshow(res2.plot())
plt.axis("off")
plt.title("Детекция людей и кошек")
plt.show()


# ========== ЗАДАНИЕ 3: СВОЁ ИЗОБРАЖЕНИЕ ==========

print("\n========== ЗАДАНИЕ 3: СВОЁ ИЗОБРАЖЕНИЕ ==========")

image_url = "https://u.cubeupload.com/via/lab33.jpg"

resp = requests.get(image_url, headers=headers, timeout=15)
resp.raise_for_status()

img = Image.open(BytesIO(resp.content)).convert("RGB")
img_array = np.array(img)

plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis("off")
plt.title("Моё изображение")
plt.show()


# ========== ЗАДАНИЕ 4: СРАВНЕНИЕ CONF ==========

print("\n========== ЗАДАНИЕ 4: СРАВНЕНИЕ CONF ==========")

for conf in [0.25, 0.35, 0.5]:

    res_conf = model_n.predict(
        source=img_array,
        conf=conf,
        iou=0.45,
        classes=[0],
        verbose=False
    )[0]

    print(
        f"conf={conf}: "
        f"найдено людей — {len(res_conf.boxes)}"
    )


# ========== ЗАДАНИЕ 5: ПОДСЧЁТ ЛЮДЕЙ ==========

print("\n========== ЗАДАНИЕ 5: ПОДСЧЁТ ЛЮДЕЙ ==========")

res = model_n.predict(
    source=img_array,
    conf=0.3,
    iou=0.45,
    classes=[0],
    verbose=False
)[0]

print(len(res.boxes))


# ========== ЗАДАНИЕ 6: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ==========

print("\n========== ЗАДАНИЕ 6: СОХРАНЕНИЕ ==========")

Image.fromarray(res.plot()).save("people_result.jpg")
Image.fromarray(res2.plot()).save("people_cat_result.jpg")

print("Сохранены файлы:")
print("people_result.jpg — детекция людей на моём фото")
print("people_cat_result.jpg — детекция людей и кошек на первом фото")


# ========== ЗАДАНИЕ 7: СРАВНЕНИЕ МОДЕЛЕЙ ==========

print("\n========== ЗАДАНИЕ 7: YOLOv8n VS YOLOv8s ==========")

for name, model in [
    ("YOLOv8n", model_n),
    ("YOLOv8s", model_s)
]:

    res_model = model.predict(
        source=img_array,
        conf=0.25,
        iou=0.45,
        classes=[0, 15],
        verbose=False
    )[0]

    people = sum(
        int(cls) == 0 for cls in res_model.boxes.cls
    )

    cats = sum(
        int(cls) == 15 for cls in res_model.boxes.cls
    )

    print(f"{name}: людей — {people}, кошек — {cats}")

    plt.figure(figsize=(10, 8))
    plt.imshow(res_model.plot())
    plt.axis("off")
    plt.title(name)
    plt.show()


# ========== ЗАДАНИЕ 8: КРУПНЫЕ ЛЮДИ ==========

print("\n========== ЗАДАНИЕ 8: КРУПНЫЕ ЛЮДИ ==========")

h, w = img_first_array.shape[:2]

res_first = model_n.predict(
    source=img_first_array,
    conf=0.45,
    iou=0.45,
    classes=[0],
    verbose=False
)[0]

large = []

for b in res_first.boxes:

    x1, y1, x2, y2 = b.xyxy[0].tolist()

    area = (x2 - x1) * (y2 - y1)

    if area >= 0.03 * w * h:
        large.append([x1, y1, x2, y2])

print("Крупных людей:", len(large))

fig, ax = plt.subplots(figsize=(10, 8))

ax.imshow(img_first_array)

for box in large:

    x1, y1, x2, y2 = map(int, box)

    ax.add_patch(
        plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="red",
            linewidth=2
        )
    )

ax.axis("off")
ax.set_title("Крупные люди — площадь рамки не менее 3% кадра")

plt.show()
