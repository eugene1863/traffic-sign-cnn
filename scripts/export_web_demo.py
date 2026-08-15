"""Builds a single self-contained HTML file that runs the trained traffic-
sign CNN entirely in the browser (no server, no Python) -- for sharing as
a link.

Why not TensorFlow.js: the `tensorflowjs` converter package is unmaintained
against current TF/numpy (it throws on a removed numpy API), so instead
this extracts the raw trained weights and pairs them with a hand-written
JS forward pass that mirrors src/model.py's architecture exactly (same
conv/batchnorm/pool/dense sequence, same padding/epsilon). The model is
~1.5M params (~6MB as float32), well within a single HTML file.

Usage:
    python -m scripts.export_web_demo
"""

import base64
import json
import os

import numpy as np
from PIL import Image
from tensorflow import keras

from src import config
from src.class_names import CLASS_NAMES

TEMPLATE_PATH = os.path.join(config.PROJECT_ROOT, "scripts", "web_demo_template.html")
OUT_DIR = os.path.join(config.PROJECT_ROOT, "web")
OUT_PATH = os.path.join(OUT_DIR, "traffic_sign_demo.html")
FONTS_DIR = os.path.join(config.PROJECT_ROOT, "scripts", "fonts")

# (layer_name, kind) in forward-pass order. "conv"/"dense" contribute
# [kernel, bias]; "bn" contributes [gamma, beta, moving_mean, moving_variance].
LAYER_SEQUENCE = [
    ("block1_conv1", "conv"), ("block1_bn1", "bn"),
    ("block1_conv2", "conv"), ("block1_bn2", "bn"),
    ("block2_conv1", "conv"), ("block2_bn1", "bn"),
    ("block2_conv2", "conv"), ("block2_bn2", "bn"),
    ("block3_conv1", "conv"), ("block3_bn1", "bn"),
    ("block3_conv2", "conv"), ("block3_bn2", "bn"),
    ("dense", "dense"),
    ("batch_normalization", "bn"),
    ("predictions", "dense"),
]

SAMPLES = [
    (14, "Stop", "Test/00093.png"),
    (2, "Speed limit (50km/h)", "Test/00034.png"),
    (13, "Yield", "Test/00026.png"),
    (11, "Right-of-way at next intersection", "Test/00004.png"),
    (40, "Roundabout mandatory", "Test/00137.png"),
    (27, "Pedestrians", "Test/00018.png"),
]


def build_manifest_and_blob(model: keras.Model):
    manifest = []
    chunks = []
    offset = 0  # float32 element offset, not bytes

    for name, kind in LAYER_SEQUENCE:
        layer = model.get_layer(name)
        weights = layer.get_weights()

        if kind in ("conv", "dense"):
            kernel, bias = weights
            entry = {
                "name": name,
                "kind": kind,
                "kernelShape": list(kernel.shape),
                "kernelOffset": offset,
            }
            flat = kernel.astype(np.float32).ravel()
            chunks.append(flat)
            offset += flat.size

            entry["biasShape"] = list(bias.shape)
            entry["biasOffset"] = offset
            flat = bias.astype(np.float32).ravel()
            chunks.append(flat)
            offset += flat.size

        elif kind == "bn":
            gamma, beta, mean, var = weights
            entry = {
                "name": name,
                "kind": "bn",
                "channels": int(gamma.shape[0]),
                "epsilon": float(layer.epsilon),
            }
            for key, arr in (("gammaOffset", gamma), ("betaOffset", beta), ("meanOffset", mean), ("varOffset", var)):
                entry[key] = offset
                flat = arr.astype(np.float32).ravel()
                chunks.append(flat)
                offset += flat.size
        else:
            raise ValueError(kind)

        manifest.append(entry)

    blob = np.concatenate(chunks).astype(np.float32)
    assert blob.size == offset
    return manifest, blob


def encode_sample_images():
    """Small JPEG thumbnails of a few real GTSRB test images, embedded so
    the demo is usable with one click and not just a blank upload box.
    """
    samples = []
    for class_id, label, rel_path in SAMPLES:
        img = Image.open(os.path.join(config.DATA_DIR, rel_path)).convert("RGB")
        img = img.resize((96, 96), Image.NEAREST)
        buf = __import__("io").BytesIO()
        img.save(buf, format="JPEG", quality=88)
        data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        samples.append({"classId": class_id, "label": label, "dataUri": data_uri})
    return samples


def encode_font(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading model...")
    model = keras.models.load_model(config.BEST_MODEL_PATH)

    print("Extracting weights...")
    manifest, blob = build_manifest_and_blob(model)
    weights_b64 = base64.b64encode(blob.tobytes()).decode("ascii")
    print(f"  {blob.size:,} float32 params -> {len(weights_b64) / 1e6:.1f} MB base64")

    print("Encoding sample images...")
    samples = encode_sample_images()

    print("Encoding fonts...")
    oswald_b64 = encode_font(os.path.join(FONTS_DIR, "oswald-latin.woff2"))
    plexsans_b64 = encode_font(os.path.join(FONTS_DIR, "plexsans-latin.woff2"))

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    replacements = {
        "__MANIFEST_JSON__": json.dumps(manifest),
        "__WEIGHTS_B64__": weights_b64,
        "__CLASS_NAMES_JSON__": json.dumps(CLASS_NAMES),
        "__SAMPLES_JSON__": json.dumps(samples),
        "__IMG_SIZE__": str(config.IMG_SIZE),
        "__FONT_OSWALD_B64__": oswald_b64,
        "__FONT_PLEXSANS_B64__": plexsans_b64,
    }
    for token, value in replacements.items():
        if token not in html:
            raise ValueError(f"template missing placeholder {token}")
        html = html.replace(token, value)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"Wrote {OUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
