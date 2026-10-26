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


# ------------- THREE -------------
image_url = "https://cubeupload.com/im/via/lab3.jpg"

resp = requests.get(image_url, headers=headers, timeout=15)
resp.raise_for_status()
img = Image.open(BytesIO(resp.content)).convert("RGB")

img_array = np.array(img)

plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis("off")
plt.title("Новое изображение")
plt.show()

