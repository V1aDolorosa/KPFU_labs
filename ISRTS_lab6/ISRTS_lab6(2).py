from pathlib import Path
import json
import zipfile
import cv2
import numpy as np
import pandas as pd

def main():
  img = cv.imread('5_figure_in_1_frame.png')
  H,W = img.shape[:2]
  print(f"Размер: {W} x {H}")

  cv2.namedWindow("5 Фигур на одном кадре", cv2.WINDOW_NORMAL)
  cv2.resizeWindow("5 Фигур на одном кадре", 1920, 720)
  cv2.imshow("5 Фигур на одном кадре", img)
  cv2.waitKey(0)

def make_mask(img, dark_object=True, kernel_size = 3):
  if kernel_size < 1 or kernel_size % 2 == 0:
    raise ValueError('Размер ядра должен быть нечетным')
  gray = cv2.cvtColor(img,cv2, COLOR_BGRWGRAY)
  blur = cv2.GaussianBlur(gray,(5,5),0)
  mode = cv2.THRESH_BINARY_INV if dark_objects else cv2.THRESH_BINARY
  _, raw = cv.adaptiveTreshold(
    blur, 0, 255, mode | cv2.THRESH_OTSU
    )
  kernel = np.ones((kernel_sizem kernel_size), np.uint8)
  clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
  clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    print("Маска после очистки:")

    cv2.namedWindow("MASK", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MASK", 1920, 720)
    cv2.imshow("MASK", mask)
    cv2.waitKey(0)

  return gray, raw, clean


def measure_parts(binary, min_area = 500, min_aspect = 1.3):
  height, width = binary.shape
  contours, _ = cv2.findContours(
    bonary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  objects = []
  
  for contour in contours:
    area = float(cv2.contourArea(contour))
    if area < min_area:
      continue
    
    x,y, w,h = cv2.boundingRect(contour)
    if x == 0 or y == 0 or x + w >= width or y + h >= height
      continue

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        continue
    #Цент круга
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"] 
    #Рамка
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)

    longest = int(np.argmax(lengths))
    length = float(lengths.max())
    short = float(lengths.min())

    
    if short <= 0:
        continue

    aspect = length / short
    valid = bool(aspect >= min_aspect)
    vector = edges[longest]

    angle = float(np.degrees(
        np.arctan2(vector[1], vector[0])
    ) % 180) if valid else float("nan")

    ix = int(np.clip(round(cx), 0, width - 1))
    iy = int(np.clip(round(cy), 0, height - 1))

     
    objects.append({
        "cx_px": cx,             # Координата x центра, в пикселях.
        "cy_px": cy,             # Координата y центра, в пикселях.
        "area_px2": area,         # Площадь контура, в квадратных пикселях.

        "length_px": length,     # Длинная сторона охватывающего прямоугольника.
        "width_px": short,       # Короткая сторона охватывающего прямоугольника.
        "aspect": aspect,        # Отношение длинной стороны к короткой.
        "angle_deg": angle,      # Угол наклона длинной стороны, в градусах.
        "angle_valid": valid,    # Достаточно ли объект вытянут для оценки угла.

        "center_inside": bool(binary[iy, ix] > 0),

        "contour": contour,      # точки контура объекта.
        "box": box              # четыре вершины его прямоугольника.
    })

    for number, obj in enumerate(objects, start=1):
        obj["id"] = number

    rows = [{key: obj[key] for key in COLUMNS} for obj in objects]
    return pd.DataFrame(rows, columns=COLUMNS), objects

def tracking_object(Directory, quality = 5):
    p = Path(Directory)
    fariko = list([x for x in p.iterdir() if x.is_file() and x.suffix in (".png",".jpg",".jpeg"))[:quality]
    for i in fariko:
        cv2.imread(i)

if __init__ == __main__:
    main()
    make_mask('5_figure_in_1_frame.png')

