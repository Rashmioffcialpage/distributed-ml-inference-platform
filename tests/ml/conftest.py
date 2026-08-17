import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for sub in ["common", "training", "trajectory", "perception", "pipelines"]:
    sys.path.insert(0, str(ROOT / "ml" / sub))
