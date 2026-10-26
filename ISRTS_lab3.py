from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

model_n = YOLO("yolov8n.pt")
model_s = YOLO("yolov8s.pt")

image_url = "https://i.pinimg.com/originals/bf/4c/c9/bf4cc984740f7588454f61fd33da26ba.jpg"

# Загрузка изображения с заголовком User-Agent
headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get(image_url, headers=headers, timeout=15)
resp.raise_for_status()
img = Image.open(BytesIO(resp.content)).convert("RGB")
img_first_array = np.array(img)

plt.figure(figsize=(8,6))
plt.imshow(img)
plt.axis('off')
plt.title("Исходное изображение")
plt.show()

# ------------ ONE ----------
#
# Детекция ТОЛЬКО класса 'person'
print('Only Person')

results = model_n.predict(
        source=np.array(img),
        conf=0.3,     # чуть строже
        iou=0.45,     # NMS (ниже — меньше дублей)
        classes=[0],  # <— фильтруем только людей
        verbose=False
    )

r = results[0]
# Подсчёт людей
num_persons = len(r.boxes)
print(f"✅ Люди найдены: {num_persons}" if num_persons else "❌ Людей не найдено")
plt.figure(figsize=(10,8))
annotated = r.plot()  # рамки + подписи 'person'
plt.imshow(annotated)
plt.axis('off')
plt.title("Результат детекции (только 'person')")
plt.show()


#-------------TWO-------------

print('Person + Cat')
results = model_n.predict(
        source=np.array(img),
        conf=0.3,     
        iou=0.45,     
        classes=[0,15],
        verbose=False
    )

r = results[0]


num_persons = len(r.boxes)
print(f"✅ Люди найдены: {num_persons}" if num_persons else "❌ Людей не найдено")

# Показ результата с рамками (YOLO рисует только то, что осталось после фильтра)
plt.figure(figsize=(10,8))
annotated = r.plot()  # рамки + подписи 'person'
plt.imshow(annotated)
plt.axis('off')
plt.title("Результат детекции (только 'person+Cat')")
plt.show()


# ------------- THREE & Four-------------
image_url = "https://u.cubeupload.com/via/lab33.jpg"

resp = requests.get(image_url, headers=headers, timeout=15)
resp.raise_for_status()
img = Image.open(BytesIO(resp.content)).convert("RGB")

img_array = np.array(img)

plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis("off")
plt.title("Новое изображение")
plt.show()

for conf in [0.25, 0.35, 0.5]:
    res_conf = model_n.predict(
        source=img_array,
        conf=conf,
        iou=0.45,
        classes=[0],
        verbose=False
    )[0]

    print(f"conf={conf}: детекций людей — {len(res_conf.boxes)}")

#--------- FIVE & SIX -----------------

# Только люди на новом изображении
res = model_n.predict(
    source=img_array,
    conf=0.3,
    iou=0.2,
    classes=[0],
    verbose=False
)[0]

# Задание 5: количество людей
print(len(res.boxes))

# Задание 6: сохранить результаты
Image.fromarray(res.plot()).save("people_result.jpg")

print("Результаты сохранены!")

#-------- SEVEN -------------
for name, model in [("YOLOv8n", model_n), ("YOLOv8s", model_s)]:
    res_model = model.predict(
        source=img_array,
        conf=0.25,
        iou=0.45,
        classes=[0, 15],
        verbose=False
    )[0]

    print(f"{name}: {len(res_model.boxes)} объектов")

    plt.figure(figsize=(10, 8))
    plt.imshow(res_model.plot())
    plt.axis("off")
    plt.title(name)
    plt.show()



#----------- EIGHT -----------

h, w = img_first_array.shape[:2]

res_first = model_n.predict(
    source=img_first_array,
    conf=0.45,
    classes=[0],
    verbose=False
)[0]

large = [
    b for b in res_first.boxes
    if (b.xyxy[0][2] - b.xyxy[0][0]) *
       (b.xyxy[0][3] - b.xyxy[0][1]) >= 0.03 * w * h
]

print("Крупных людей:", len(large))

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img_first_array)

for b in large:
    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
    ax.add_patch(plt.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        fill=False, edgecolor="red", linewidth=2
    ))

ax.axis("off")
plt.show()
