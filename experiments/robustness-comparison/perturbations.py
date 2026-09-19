"""Image perturbation definitions and variant construction."""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _rgb_to_hsv(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    with np.errstate(divide="ignore", invalid="ignore"):
        hue = np.where(
            delta == 0,
            0.0,
            np.where(
                maxc == r,
                ((g - b) / delta) % 6.0,
                np.where(maxc == g, (b - r) / delta + 2.0, (r - g) / delta + 4.0),
            ),
        )
        hue = hue * 60.0
        saturation = np.where(maxc == 0, 0.0, delta / maxc)
        value = maxc
    return np.stack((hue, saturation, value), axis=-1)


def _hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0] / 60.0, hsv[..., 1], hsv[..., 2]
    i = np.floor(h).astype(np.int64) % 6
    f = h - np.floor(h)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    return np.stack(
        (
            np.where(i == 0, v, np.where(i == 1, q, np.where(i == 2, p, np.where(i == 3, p, np.where(i == 4, t, v))))),
            np.where(i == 0, t, np.where(i == 1, v, np.where(i == 2, v, np.where(i == 3, q, np.where(i == 4, p, p))))),
            np.where(i == 0, p, np.where(i == 1, p, np.where(i == 2, t, np.where(i == 3, v, np.where(i == 4, v, q))))),
        ),
        axis=-1,
    )


def _adjust_hue(image, factor):
    """Shift hue by factor * 360 degrees."""
    if factor == 0.0:
        return image
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    hsv = _rgb_to_hsv(array)
    hsv[..., 0] = (hsv[..., 0] + factor * 360.0) % 360.0
    array = np.clip(np.rint(_hsv_to_rgb(hsv) * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def perturb(image, transform, value):
    if transform == "rotation":
        return image.rotate(value, expand=False, fillcolor=(0, 0, 0))
    if transform == "crop":
        width, height = image.size
        crop_width = max(int(round(width * value)), 1)
        crop_height = max(int(round(height * value)), 1)
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        center = image.crop((left, top, left + crop_width, top + crop_height))
        return center.resize((width, height), Image.BILINEAR)
    if transform == "brightness":
        return ImageEnhance.Brightness(image).enhance(value)
    if transform == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=value))
    if transform == "saturation":
        return ImageEnhance.Color(image).enhance(value)
    if transform == "hue":
        return _adjust_hue(image, value)
    raise ValueError(f"Unknown transform: {transform}")


TRANSFORMS = [
    {"name": "rotation", "values": [5, 10, 20, 45, 90, 180], "unit": "deg"},
    {"name": "crop", "values": [0.9, 0.8, 0.7, 0.5, 0.3], "unit": "x"},
    {"name": "brightness", "values": [0.2, 0.4, 0.6, 0.8, 1.2, 1.5, 2.0, 3.0], "unit": "x"},
    {"name": "blur", "values": [0.5, 1.0, 2.0, 4.0], "unit": "px"},
    {"name": "saturation", "values": [0.2, 0.4, 0.7, 1.3, 1.7, 2.5], "unit": "x"},
    {"name": "hue", "values": [-0.3, -0.15, 0.15, 0.3], "unit": None},
]


def level_label(spec, value):
    number = str(int(value)) if value == int(value) else f"{value:g}"
    return f"{spec['name']}_{number}{spec['unit'] or ''}"


def build_variants(images, image_ids, selected):
    variants = []
    for index, (image, image_id) in enumerate(zip(images, image_ids)):
        variants.append({
            "query": index,
            "image_id": image_id,
            "transform": "original",
            "value": None,
            "label": "-",
            "image": image,
        })
        for spec in selected:
            for value in spec["values"]:
                variants.append({
                    "query": index,
                    "image_id": image_id,
                    "transform": spec["name"],
                    "value": value,
                    "label": level_label(spec, value),
                    "image": perturb(image, spec["name"], value),
                })
    return variants
