from pathlib import Path
import json
import zipfile
import cv2
import numpy as np
import pandas as pd

img = cv2.imread('5_figure_in_1_frame.png')
H, W = img.shape[:2]
print("Размер:", W, "x", H)

cv2.namedWindow("5 Фигур на одном кадре", cv2.WINDOW_NORMAL)
cv2.resizeWindow("5 Фигур на одном кадре", 1920, 720)
cv2.imshow("5 Фигур на одном кадре", img)
cv2.waitKey(0)


MIN_AREA_PX2 = 500
MIN_ASPECT = 1.3


def make_mask(img, dark_objects=True, kernel_size=3):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Размер ядра должен быть нечётным.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mode = cv2.THRESH_BINARY_INV if dark_objects else cv2.THRESH_BINARY
    _, raw = cv2.threshold(
        blur, 0, 255, mode | cv2.THRESH_OTSU
    )
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    return gray, raw, clean


gray, raw_mask, mask = make_mask(img)
print("Маска до очистки:")

cv2.namedWindow("Бинарная маска", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Бинарная маска", 1920, 720)
cv2.imshow("Бинарная маска", raw_mask)
cv2.waitKey(0)

print("Маска после очистки:")

cv2.namedWindow("MASK", cv2.WINDOW_NORMAL)
cv2.resizeWindow("MASK", 1920, 720)
cv2.imshow("MASK", mask)
cv2.waitKey(0)


COLUMNS = [
    "id", "cx_px", "cy_px", "area_px2", "length_px",
    "width_px", "aspect", "angle_deg", "angle_valid",
    "center_inside"
]


def measure_parts(binary, min_area=500, min_aspect=1.3):
    height, width = binary.shape
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    objects = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

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
            "cx_px": cx, "cy_px": cy, "area_px2": area,
            "length_px": length, "width_px": short,
            "aspect": aspect, "angle_deg": angle,
            "angle_valid": valid,
            "center_inside": bool(binary[iy, ix] > 0),
            "contour": contour, "box": box
        })

    objects.sort(key=lambda obj: (obj["cy_px"], obj["cx_px"]))

    for number, obj in enumerate(objects, start=1):
        obj["id"] = number

    rows = [{key: obj[key] for key in COLUMNS} for obj in objects]
    return pd.DataFrame(rows, columns=COLUMNS), objects


measurements, objects = measure_parts(
    mask, MIN_AREA_PX2, MIN_ASPECT
)
print(measurements.round(2))


REFERENCE = (W / 2, H / 2)

measurements["distance_px"] = np.hypot(
    measurements["cx_px"] - REFERENCE[0],
    measurements["cy_px"] - REFERENCE[1]
)

candidates = measurements[
    measurements["angle_valid"] & measurements["center_inside"]
].sort_values(["distance_px", "id"])

if candidates.empty:
    selected_id = None
    target = {"status": "NO_TARGET"}
else:
    chosen = candidates.iloc[0]
    selected_id = int(chosen["id"])

    target = {
        "status": "TARGET_FOUND",
        "object_id": selected_id,
        "coordinate_frame": "image_px",
        "x_px": float(chosen["cx_px"]),
        "y_px": float(chosen["cy_px"]),
        "axis_angle_deg": float(chosen["angle_deg"]),
        "orientation_period_deg": 180
    }

print(json.dumps(target, ensure_ascii=False, indent=2))


annotated = img.copy()
ref = tuple(np.round(REFERENCE).astype(int))

cv2.drawMarker(
    annotated, ref, (255, 0, 0),
    cv2.MARKER_CROSS, 24, 2
)

for obj in objects:
    center = (round(obj["cx_px"]), round(obj["cy_px"]))
    color = (0, 0, 255) if obj["id"] == selected_id else (0, 140, 0)

    cv2.drawContours(annotated, [obj["contour"]], -1, color, 2)

    box = np.round(obj["box"]).astype(np.int32)
    cv2.polylines(annotated, [box], True, color, 2)

    cv2.drawMarker(
        annotated, center, color,
        cv2.MARKER_CROSS, 18, 2
    )

    if obj["angle_valid"]:
        theta = np.radians(obj["angle_deg"])
        delta = 0.4 * obj["length_px"] * np.array([
            np.cos(theta), np.sin(theta)
        ])

        p1 = tuple(np.round(np.array(center) - delta).astype(int))
        p2 = tuple(np.round(np.array(center) + delta).astype(int))

        cv2.line(annotated, p1, p2, color, 3)
        label = f'ID {obj["id"]}: {obj["angle_deg"]:.1f} deg'
    else:
        label = f'ID {obj["id"]}: angle undefined'

    cv2.putText(
        annotated, label,
        (max(0, center[0] - 95), max(18, center[1] - 18)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        cv2.LINE_AA
    )


# Отображение итогового изображения
cv2.namedWindow("Annotated", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Annotated", 1920, 720)
cv2.imshow("Annotated", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()


out = Path("pose_results")
out.mkdir(exist_ok=True)

measurements.to_csv(
    out / "measurements.csv",
    index=False,
    encoding="utf-8-sig"
)

(out / "target.json").write_text(
    json.dumps(target, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

for name, image in [
    ("source.png", img),
    ("mask.png", mask),
    ("annotated.png", annotated)
]:
    if not cv2.imwrite(str(out / name), image):
        raise RuntimeError(f"Не удалось сохранить {name}.")


with zipfile.ZipFile(
    "pose_results.zip", "w", zipfile.ZIP_DEFLATED
) as archive:
    for path in sorted(out.iterdir()):
        if path.is_file():
            archive.write(path, arcname=path.name)

print("Результаты сохранены в pose_results.zip")


def axis_error_deg(measured, reference):
    return abs((measured - reference + 90) % 180 - 90)
