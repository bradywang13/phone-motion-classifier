"""
import_motionsense.py - download the MotionSense dataset and convert it into
recordings that watch_motion.py and motion_cnn.py can train on.

MotionSense: 24 people with an iPhone 6s in their front trouser pocket, recorded
with the same iOS motion API Sensor Logger uses (gravity already removed).
https://github.com/mmalekzadeh/motion-sense

Run once:  python import_motionsense.py
Result:    motionsense/<Movement>/<trial>_sub<N>.csv   (same labels as your data/ folder)

The trainers then use these as EXTRA training data (see EXTRA_FOLDERS in each
script). Accuracy is still measured only on your own recordings.
"""

import io
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip"
OUT = Path("motionsense")
FS = 50          # MotionSense sample rate
G = 9.81         # MotionSense acceleration is in g, Sensor Logger's is in m/s^2

# MotionSense activity code -> your folder name. Stairs (ups/dws) are left out.
LABELS = {"wlk": "Walking", "jog": "Running", "sit": "Still", "std": "Still"}


def main():
    os.chdir(Path(__file__).resolve().parent)
    print(f"Downloading {URL} (~70 MB) ...")
    with urllib.request.urlopen(URL) as r:
        archive = zipfile.ZipFile(io.BytesIO(r.read()))

    n = 0
    for name in archive.namelist():
        # e.g. A_DeviceMotion_data/wlk_7/sub_3.csv
        parts = Path(name).parts
        if len(parts) != 3 or not name.endswith(".csv"):
            continue
        trial, sub = parts[1], Path(parts[2]).stem
        label = LABELS.get(trial.split("_")[0])
        if label is None:
            continue
        df = pd.read_csv(archive.open(name))
        out = pd.DataFrame({"seconds_elapsed": df.index / FS})
        for axis in "xyz":
            out[f"acc_{axis}"] = df[f"userAcceleration.{axis}"] * G
        for axis in "xyz":
            out[f"gyro_{axis}"] = df[f"rotationRate.{axis}"]
        path = OUT / label / f"{trial}_{sub}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        n += 1

    print(f"Wrote {n} recordings to {OUT}/")
    for d in sorted(OUT.iterdir()):
        print(f"  {d.name:<10} {len(list(d.glob('*.csv')))} recordings")


if __name__ == "__main__":
    main()
