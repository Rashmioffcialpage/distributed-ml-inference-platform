#!/usr/bin/env python3
"""Optional alternative to real_dataset.py's FTSC loader: GTSRB (German
Traffic Sign Recognition Benchmark) via Kaggle
(https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign),
the field's most widely-cited traffic-sign dataset (Stallkamp et al., 2012).

Not run automatically by this repo: the sandbox this project was developed
in blocks Kaggle's API and CDN outright (403 on kaggle.com, exactly like the
Docker Hub / PyPI-wheel-mirror blocks noted elsewhere in this repo), so this
script is provided for you to run on a machine that isn't behind that
restriction, with your own Kaggle API credentials.

Setup (once): https://www.kaggle.com/docs/api -> download kaggle.json to
~/.kaggle/kaggle.json

Usage:
    pip install kaggle
    python3 ml/perception/download_gtsrb_kaggle.py --data-dir data/gtsrb

After downloading, GTSRB's 43 fine-grained classes and directory-per-class
layout are different from FTSC's filename-prefix scheme in real_dataset.py —
write a small loader analogous to `load_ftsc()` reading
`data-dir/Train/<class_id>/*.ppm` if you swap this in; the model
architecture (ml/perception/model.py's PerceptionCNN, parameterized by
in_channels/image_size/num_classes) already supports it unchanged.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/gtsrb")
    p.add_argument("--dataset", default="meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
    args = p.parse_args()

    out = pathlib.Path(args.data_dir)
    out.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(out), "--unzip"],
        check=True,
    )
    print(f"GTSRB downloaded to {out}")


if __name__ == "__main__":
    main()
