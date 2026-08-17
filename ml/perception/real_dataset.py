"""Loader for a real public driving-perception dataset: FTSC (French
Traffic Sign Classification), https://github.com/andrewcaunes/FTSC.

FTSC is 10,959 real photographs of French traffic signs, cropped from 2D RGB
cameras mounted on a vehicle in Antony, France, in varying weather and
lighting — genuine driving-perception imagery, not synthetic. Released under
CC BY-NC 4.0 (Attribution-NonCommercial): usable here for non-commercial,
educational/portfolio purposes with attribution, which is exactly what this
repo is. It is NOT redistributed in this repo (233MB of licensed images
don't belong in git); this module clones it on demand.

This replaces `ml/perception/synthetic_frames.py` as the perception task's
data source. The synthetic generator is kept (and still used by
`ml/training/train_perception.py`) as the original rapid-prototyping path —
see docs/architecture.md#dataset-methodology for why both exist.

Label taxonomy: FTSC's filenames encode a coarse category as the prefix
before the first "-" (e.g. "regulatory-prohibitory-B1_...jpg" -> "regulatory").
There are 6 such categories: regulatory, informative, danger, NIC (not in
catalogue), temporary, others. That's the classification target used here,
rather than FTSC's full 91 fine-grained sign classes, to keep this a
reasonable-sized problem for the same small CNN architecture
(`ml/perception/model.py`) used throughout this project.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np

REPO_URL = "https://github.com/andrewcaunes/FTSC.git"
LICENSE_NOTE = (
    "FTSC dataset (c) andrewcaunes, licensed CC BY-NC 4.0 "
    "(https://creativecommons.org/licenses/by-nc/4.0/). "
    "Non-commercial use with attribution only."
)
CLASS_NAMES = ["regulatory", "informative", "NIC", "danger", "temporary", "others"]
IMAGE_SIZE = 64


def ensure_dataset(data_dir: pathlib.Path) -> pathlib.Path:
    """Shallow-clone FTSC into data_dir if it isn't already there."""
    images_dir = data_dir / "images"
    if images_dir.is_dir() and any(images_dir.iterdir()):
        return images_dir

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning FTSC ({REPO_URL}) into {data_dir} ...", file=sys.stderr)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(data_dir)], check=True)
    print(LICENSE_NOTE, file=sys.stderr)
    return images_dir


def _label_for(filename: str) -> str | None:
    prefix = filename.split("-", 1)[0]
    return prefix if prefix in CLASS_NAMES else None


def load_ftsc(
    data_dir: pathlib.Path,
    limit: int | None = None,
    image_size: int = IMAGE_SIZE,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load FTSC images as (N, 3, image_size, image_size) float32 in [0,1]
    plus integer labels indexing CLASS_NAMES. `limit` caps the number of
    images loaded (uniformly sampled across classes) to keep this tractable
    on CPU — the full 10,959-image set takes several minutes to decode and
    resize."""
    from PIL import Image

    images_dir = ensure_dataset(data_dir)
    files = sorted(images_dir.iterdir())
    labeled = [(f, _label_for(f.name)) for f in files]
    labeled = [(f, label) for f, label in labeled if label is not None]

    if limit is not None and limit < len(labeled):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(labeled), size=limit, replace=False)
        labeled = [labeled[i] for i in idx]

    label_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    images = np.zeros((len(labeled), 3, image_size, image_size), dtype=np.float32)
    labels = np.zeros(len(labeled), dtype=np.int64)

    for i, (path, label) in enumerate(labeled):
        with Image.open(path) as im:
            im = im.convert("RGB").resize((image_size, image_size))
            arr = np.asarray(im, dtype=np.float32) / 255.0  # (H, W, 3)
        images[i] = arr.transpose(2, 0, 1)  # -> (3, H, W)
        labels[i] = label_to_idx[label]

    return images, labels
