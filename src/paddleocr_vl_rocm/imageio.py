from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .layout import LayoutBox


def _crop(rgb_image: Image.Image, box: LayoutBox) -> Image.Image:
    cropped = rgb_image.crop(tuple(float(v) for v in box.coordinate))
    polygon_points = box.polygon_points
    if not polygon_points:
        return cropped
    try:
        import cv2
        import numpy as np
    except Exception:
        return cropped
    x1, y1, _, _ = box.coordinate
    polygon = np.asarray(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
    polygon = polygon - np.asarray([x1, y1], dtype=np.int32)
    array = np.asarray(cropped).copy()
    mask = np.zeros(array.shape[:2], dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    array[~mask.astype(bool)] = 255
    return Image.fromarray(array)


def _crop_from_bgr(bgr_image: Any, box: LayoutBox) -> Image.Image:
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise RuntimeError("OpenCV and NumPy are required for polygon-masked crop.") from exc
    xmin, ymin, xmax, ymax = [int(i) for i in box.coordinate]
    img_crop = bgr_image[ymin:ymax, xmin:xmax].copy()
    polygon_points = box.polygon_points
    if polygon_points:
        mask = np.zeros(img_crop.shape[:2], dtype=np.int32)
        polygon = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        if polygon is not None and len(polygon) > 0:
            polygon = polygon - np.array([xmin, ymin])
        cv2.fillPoly(mask, [polygon], 1)
        img_crop[~mask.astype(bool)] = 255
    rgb_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_crop)


def _open_crop_source(path: Path) -> Image.Image:
    try:
        import cv2
        import numpy as np
    except Exception:
        with Image.open(path) as image:
            return image.convert("RGB")
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        with Image.open(path) as image:
            return image.convert("RGB")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _open_crop_source_bgr(path: Path) -> Any | None:
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return bgr


def _crop_margin(image: Image.Image) -> Image.Image:
    try:
        import cv2
        import numpy as np
    except Exception:
        return image
    array = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    max_val = gray.max()
    min_val = gray.min()
    if max_val == min_val:
        return image
    data = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    _, binary = cv2.threshold(data, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    if w <= 2 or h <= 2:
        return image
    return Image.fromarray(array[y : y + h, x : x + w])


def _merge_images(images: list[Image.Image], aligns: list[str]) -> Image.Image:
    if len(images) == 1:
        return images[0]
    merged = images[0]
    for idx, image in enumerate(images[1:]):
        align = aligns[idx] if idx < len(aligns) else "center"
        width = max(merged.width, image.width)
        height = merged.height + image.height
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        if align == "right":
            x1 = width - merged.width
            x2 = width - image.width
        elif align == "left":
            x1 = x2 = 0
        else:
            x1 = (width - merged.width) // 2
            x2 = (width - image.width) // 2
        canvas.paste(merged, (x1, 0))
        canvas.paste(image, (x2, merged.height))
        merged = canvas
    return merged
