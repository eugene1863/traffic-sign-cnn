"""Run inference with a trained traffic-sign CNN.

Single image, with an annotated copy saved next to the output path:
    python -m src.predict --image path/to/sign.jpg

Every image in a folder, results printed as a table:
    python -m src.predict --image-dir path/to/folder

Export the trained model to TFLite (for running on a phone / embedded
board in a vehicle, where a full TF install isn't available):
    python -m src.predict --export-tflite
"""

import argparse
import glob
import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow import keras

from . import config
from .class_names import CLASS_NAMES

# Below this confidence, predictions are flagged as unreliable rather than
# acted on -- important for a driver-assistance context where a wrong but
# confident-looking label is worse than admitting uncertainty.
LOW_CONFIDENCE_THRESHOLD = 0.60


def load_image(path: str, img_size: int) -> np.ndarray:
    img = keras.utils.load_img(path, target_size=(img_size, img_size))
    return keras.utils.img_to_array(img)


def predict_batch(model: keras.Model, images: np.ndarray, top_k: int = 3):
    """Returns, per image, a list of (class_name, probability) sorted by
    probability descending, truncated to top_k.
    """
    probs = model.predict(images, verbose=0)
    results = []
    for row in probs:
        order = np.argsort(row)[::-1][:top_k]
        results.append([(CLASS_NAMES[i], float(row[i])) for i in order])
    return results


def annotate_image(path: str, top_prediction, out_path: str) -> None:
    name, conf = top_prediction
    label = f"{name} ({conf:.0%})"
    if conf < LOW_CONFIDENCE_THRESHOLD:
        label = f"LOW CONFIDENCE: {label}"

    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, img.width // 20))
    except OSError:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    pad = 6
    draw.rectangle([0, 0, text_w + 2 * pad, text_h + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 0), font=font)

    img.save(out_path)


def export_tflite(model_path: str, out_path: str) -> None:
    model = keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model ({len(tflite_model) / 1024:.0f} KB) to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=config.BEST_MODEL_PATH)
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--image", help="path to a single image")
    p.add_argument("--image-dir", help="path to a folder of images")
    p.add_argument("--out-dir", default=config.OUTPUTS_DIR, help="where annotated images are written")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--export-tflite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.export_tflite:
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        export_tflite(args.model, config.TFLITE_MODEL_PATH)
        return

    if not args.image and not args.image_dir:
        raise SystemExit("Pass --image, --image-dir, or --export-tflite.")

    paths = [args.image] if args.image else sorted(
        p for ext in ("*.png", "*.jpg", "*.jpeg", "*.ppm") for p in glob.glob(os.path.join(args.image_dir, ext))
    )
    if not paths:
        raise SystemExit(f"No images found at {args.image or args.image_dir}")

    model = keras.models.load_model(args.model)
    images = np.stack([load_image(p, args.img_size) for p in paths])
    all_results = predict_batch(model, images, top_k=args.top_k)

    os.makedirs(args.out_dir, exist_ok=True)
    for path, results in zip(paths, all_results):
        top_name, top_conf = results[0]
        flag = " [LOW CONFIDENCE]" if top_conf < LOW_CONFIDENCE_THRESHOLD else ""
        print(f"\n{path}")
        print(f"  -> {top_name} ({top_conf:.1%}){flag}")
        for name, conf in results[1:]:
            print(f"     also considered: {name} ({conf:.1%})")

        out_path = os.path.join(args.out_dir, f"pred_{os.path.basename(path)}")
        annotate_image(path, results[0], out_path)
        print(f"  annotated image saved to {out_path}")


if __name__ == "__main__":
    main()
