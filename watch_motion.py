"""
watch_motion.py - classify movements from iPhone accelerometer + gyroscope data
(recorded with the Sensor Logger app).

In VS Code: set MODE (and RECORDING) just below the imports, save,
then click the Run button (triangle, top right of the editor).

From a terminal instead:
    python watch_motion.py demo                   # fake data -> train -> predict (tests your setup)
    python watch_motion.py plot  <recording>      # look at one recording
    python watch_motion.py train [data_folder]    # train on your recordings (default: data)
    python watch_motion.py predict <recording>    # label every second of a new recording

A <recording> is one Sensor Logger export: the .zip file, the unzipped folder,
or a single CSV that has both accelerometer and gyroscope columns.

Training folder layout (one subfolder per movement; the folder name is the label):
    data/
      walking/   2026-10-03_14-02-11.zip   2026-10-03_14-05-40.zip  ...
      running/   ...
      still/     ...
Zips are unzipped automatically.

Install once:  pip install -r requirements.txt
"""

import os
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

# ================= WHAT TO RUN (edit these, save, then press Run) =================
MODE = "predict"          # "demo", "plot", "train" or "predict"
DATA_FOLDER = "data"   # folder with one subfolder per movement (used by "train")
RECORDING = "test/2026-10-03_22-09-37"   # one recording (used by "plot" and "predict")
# ==================================================================================

# ---------------- Settings you might tweak ----------------
WINDOW_SECONDS = 2.0   # length of each chunk the model looks at
STEP_SECONDS = 1.0     # how far the window slides each time (50% overlap with a 2 s window)
TRIM_SECONDS = 3.0     # cut off start/end of each TRAINING recording (pocketing/unpocketing the phone)
MODEL_PATH = Path("motion_model.joblib")      # your real model
DEMO_MODEL_PATH = Path("demo_model.joblib")  # the demo saves here, so it never overwrites yours


# ---------------- 1. Loading ----------------
def to_seconds(df):
    """Get a time column in seconds. Sensor Logger has 'seconds_elapsed'."""
    if "seconds_elapsed" in df.columns:
        return df["seconds_elapsed"].to_numpy(dtype=float)
    time_col = next((c for c in df.columns if "time" in c.lower()), None)
    if time_col is None:
        raise ValueError(f"No time column. Columns are: {list(df.columns)}")
    t = df[time_col].to_numpy(dtype=float)
    step = np.median(np.diff(t))
    if step > 1e5:      # nanoseconds
        return t / 1e9
    if step > 1:        # milliseconds
        return t / 1e3
    return t


def find_xyz(columns, must_contain=()):
    """Find the x, y, z columns (optionally only ones whose name contains a keyword)."""
    cols = [c for c in columns
            if not must_contain or any(k in c.lower() for k in must_contain)]
    found = []
    for axis in "xyz":
        exact = [c for c in cols if c.lower() == axis]
        ending = [c for c in cols if c.lower().rstrip("_ ").endswith(axis)
                  and "time" not in c.lower()]
        match = exact or ending
        if not match:
            return None
        found.append(match[0])
    return found


def as_folder(path):
    """If the recording is a .zip, unzip it next to itself and return the folder."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        folder = path.with_suffix("")
        if not folder.exists():
            zipfile.ZipFile(path).extractall(folder)
        return folder
    return path


def find_csv(folder, name):
    """Shortest CSV whose name starts with `name` (so Accelerometer.csv beats AccelerometerUncalibrated.csv)."""
    matches = [p for p in folder.rglob("*.csv") if p.name.lower().startswith(name)]
    return min(matches, key=lambda p: len(p.name)) if matches else None


def load_recording(path):
    """Return (time in seconds from start, DataFrame with acc_x..z and gyro_x..z)."""
    if not Path(path).exists():
        sys.exit(f"Recording '{path}' not found. Check RECORDING at the top of the script.")
    path = as_folder(path)

    if path.is_dir():                                   # Sensor Logger export folder
        acc_file, gyro_file = find_csv(path, "accelerometer"), find_csv(path, "gyroscope")
        if acc_file is None or gyro_file is None:
            raise ValueError(f"{path}: need Accelerometer.csv and Gyroscope.csv inside")
        parts = []
        for f, prefix in [(acc_file, "acc"), (gyro_file, "gyro")]:
            df = pd.read_csv(f)
            xyz = find_xyz(df.columns)
            if xyz is None:
                raise ValueError(f"{f}: no x/y/z columns. Columns are: {list(df.columns)}")
            part = pd.DataFrame({"t": to_seconds(df)})
            for axis, col in zip("xyz", xyz):
                part[f"{prefix}_{axis}"] = df[col].astype(float).to_numpy()
            parts.append(part.dropna().sort_values("t"))
        # The two sensors tick at slightly different moments: pair each accel
        # sample with the nearest gyro sample.
        data = pd.merge_asof(parts[0], parts[1], on="t",
                             direction="nearest", tolerance=0.05)
    else:                                               # one CSV with both sensors
        df = pd.read_csv(path)
        acc = find_xyz(df.columns, ("acc",))
        gyro = find_xyz(df.columns, ("gyro", "rotation"))
        if acc is None or gyro is None:
            raise ValueError(f"{path}: couldn't find accel + gyro x/y/z columns. "
                             f"Columns are: {list(df.columns)}")
        data = pd.DataFrame({"t": to_seconds(df)})
        for prefix, cols in [("acc", acc), ("gyro", gyro)]:
            for axis, col in zip("xyz", cols):
                data[f"{prefix}_{axis}"] = df[col].astype(float).to_numpy()

    data = data.dropna().sort_values("t").reset_index(drop=True)
    t = data.pop("t").to_numpy()
    # Overall size of the motion: doesn't depend on how the phone is turned
    data["acc_mag"] = np.sqrt(data.acc_x**2 + data.acc_y**2 + data.acc_z**2)
    data["gyro_mag"] = np.sqrt(data.gyro_x**2 + data.gyro_y**2 + data.gyro_z**2)
    return t - t[0], data


# ---------------- 2. Windows and features ----------------
def window_features(win, fs):
    """Turn one window (rows = samples, cols = channels) into a dict of numbers."""
    feats = {}
    for col in win.columns:
        x = win[col].to_numpy()
        feats[f"{col}_mean"] = x.mean()
        feats[f"{col}_std"] = x.std()
        feats[f"{col}_min"] = x.min()
        feats[f"{col}_max"] = x.max()
        feats[f"{col}_range"] = x.max() - x.min()
        feats[f"{col}_energy"] = np.mean(x ** 2)
        # Dominant frequency: how many times per second the motion repeats (e.g. steps/sec)
        spectrum = np.abs(np.fft.rfft(x - x.mean()))
        freqs = np.fft.rfftfreq(len(x), d=1 / fs)
        feats[f"{col}_domfreq"] = freqs[np.argmax(spectrum[1:]) + 1]
    return feats


def recording_to_windows(path, trim=TRIM_SECONDS):
    """One recording -> DataFrame with one row of features per window."""
    t, signals = load_recording(path)
    keep = (t >= trim) & (t <= t[-1] - trim)
    t, signals = t[keep], signals[keep].reset_index(drop=True)
    if len(t) < 10:
        return pd.DataFrame()

    fs = 1 / np.median(np.diff(t))              # samples per second
    win_n = int(round(WINDOW_SECONDS * fs))
    step_n = max(1, int(round(STEP_SECONDS * fs)))

    rows = []
    for start in range(0, len(signals) - win_n + 1, step_n):
        feats = window_features(signals.iloc[start:start + win_n], fs)
        feats["t_start"] = t[start]
        rows.append(feats)
    return pd.DataFrame(rows)


# ---------------- 3. Train ----------------
def find_recordings(label_dir):
    for z in label_dir.glob("*.zip"):           # unzip any zips first
        as_folder(z)
    return [p for p in sorted(label_dir.iterdir())
            if (p.is_dir() and not p.name.startswith(("_", ".")))
            or p.suffix.lower() == ".csv"]


def build_dataset(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        sys.exit(f"Folder '{data_dir}' not found. Run this from your project folder.")
    frames = []
    for label_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for rec in find_recordings(label_dir):
            try:
                w = recording_to_windows(rec)
            except ValueError as e:
                print(f"  skipped {rec.name}: {e}")
                continue
            if w.empty:
                print(f"  skipped {rec.name}: too short")
                continue
            w["label"] = label_dir.name
            w["recording"] = str(rec)
            frames.append(w)
            print(f"  {label_dir.name:<12} {rec.name:<28} {len(w):>4} windows")
    if not frames:
        sys.exit(f"No recordings found. Expected {data_dir}/<movement>/<recording>")
    return pd.concat(frames, ignore_index=True)


def train(data_dir, model_path=MODEL_PATH):
    print(f"Loading {data_dir}/ ...")
    data = build_dataset(data_dir)
    X = data.drop(columns=["label", "recording", "t_start"])
    y = data["label"]
    groups = data["recording"]

    model = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)

    # Honest accuracy: test on whole recordings the model never trained on.
    # (Random windows would leak, because neighbouring windows overlap.)
    n_splits = int(min(5, data.groupby("label")["recording"].nunique().min()))
    if y.nunique() < 2:
        sys.exit("Need at least 2 movements (2 subfolders) to train.")
    if n_splits >= 2:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
        preds = cross_val_predict(model, X, y, groups=groups, cv=cv)
        labels = sorted(y.unique())
        print("\n=== Accuracy on recordings the model never saw ===")
        print(classification_report(y, preds, zero_division=0))
        print("Confusion matrix (rows = what you did, columns = what the model said):")
        print(pd.DataFrame(confusion_matrix(y, preds, labels=labels),
                           index=labels, columns=labels))
    else:
        print("\nNeed 2+ recordings per movement to measure accuracy. Skipping that step.")

    model.fit(X, y)
    joblib.dump({"model": model, "columns": list(X.columns),
                 "labels": sorted(y.unique()), "trained_on": str(data_dir)}, model_path)
    print(f"\nSaved model to {model_path}")

    top = pd.Series(model.feature_importances_, index=X.columns).nlargest(10)
    print("\nMost useful features:")
    print(top.round(3).to_string())


# ---------------- 4. Predict ----------------
def predict(path, model_path=MODEL_PATH):
    if not model_path.exists():
        sys.exit(f"No {model_path} here. Run 'train' first.")
    bundle = joblib.load(model_path)
    if "trained_on" in bundle:
        print(f"Using model trained on '{bundle['trained_on']}' "
              f"with movements: {', '.join(bundle['labels'])}")
    w = recording_to_windows(path, trim=0)
    if w.empty:
        sys.exit("Recording too short.")
    X = w[bundle["columns"]]
    model = bundle["model"]
    preds = model.predict(X)
    conf = model.predict_proba(X).max(axis=1)

    print(f"\nPredictions for {path}:")
    for t0, p, c in zip(w["t_start"], preds, conf):
        print(f"  {t0:6.1f}s  {p:<12} ({c:.0%} sure)")

    # Merge runs of the same label into segments
    segments = []
    for t0, p in zip(w["t_start"], preds):
        if segments and segments[-1][2] == p:
            segments[-1][1] = t0 + WINDOW_SECONDS
        else:
            if segments:
                segments[-1][1] = t0        # previous activity ends where this one starts
            segments.append([t0, t0 + WINDOW_SECONDS, p])
    print("\nSummary:")
    for start, end, p in segments:
        print(f"  {start:6.1f}s - {end:6.1f}s  {p}")


# ---------------- Plot ----------------
def plot(path):
    import matplotlib.pyplot as plt
    t, s = load_recording(path)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    for ax, prefix, unit in [(axes[0], "acc", "accelerometer"), (axes[1], "gyro", "gyroscope")]:
        for axis in "xyz":
            ax.plot(t, s[f"{prefix}_{axis}"], label=axis, lw=0.8)
        ax.set_ylabel(unit)
        ax.legend(loc="upper right")
    axes[1].set_xlabel("seconds")
    plt.tight_layout()
    plt.show()


# ---------------- Fake data to test your setup ----------------
def _fake_motion(kind, t, rng):
    n = len(t)
    def noise(s):
        return rng.normal(0, s, n)
    if kind == "walking":          # ~1.8 steps/sec, moderate bounce
        f = rng.uniform(1.6, 2.0)
        acc = [0.5*np.sin(2*np.pi*f*t) + noise(0.1),
               0.3*np.sin(2*np.pi*f*t + 1) + noise(0.1),
               1.5*np.abs(np.sin(np.pi*f*t)) + noise(0.1)]
        gyro = [0.8*np.sin(np.pi*f*t) + noise(0.1), noise(0.1), 0.3*np.sin(np.pi*f*t) + noise(0.1)]
    elif kind == "running":        # ~2.8 steps/sec, big bounce
        f = rng.uniform(2.5, 3.1)
        acc = [1.5*np.sin(2*np.pi*f*t) + noise(0.3),
               1.0*np.sin(2*np.pi*f*t + 1) + noise(0.3),
               5.0*np.abs(np.sin(np.pi*f*t)) + noise(0.3)]
        gyro = [2.5*np.sin(np.pi*f*t) + noise(0.3), noise(0.3), 1.0*np.sin(np.pi*f*t) + noise(0.3)]
    else:                          # still
        acc = [noise(0.02) for _ in range(3)]
        gyro = [noise(0.01) for _ in range(3)]
    return acc, gyro


def _fake_recording(kinds, folder, rng, fs=100, seconds_each=30):
    acc_parts, gyro_parts = [], []
    for i, kind in enumerate(kinds):
        t = np.arange(0, seconds_each, 1 / fs) + i * seconds_each
        acc, gyro = _fake_motion(kind, t, rng)
        acc_parts.append(pd.DataFrame({"seconds_elapsed": t, "z": acc[2], "y": acc[1], "x": acc[0]}))
        # gyro ticks a few ms later than accel, like a real phone
        gyro_parts.append(pd.DataFrame({"seconds_elapsed": t + 0.003,
                                        "z": gyro[2], "y": gyro[1], "x": gyro[0]}))
    folder.mkdir(parents=True, exist_ok=True)
    for name, parts in [("Accelerometer.csv", acc_parts), ("Gyroscope.csv", gyro_parts)]:
        df = pd.concat(parts, ignore_index=True)
        df.insert(0, "time", (1.7e18 + df["seconds_elapsed"] * 1e9).astype(np.int64))
        df.to_csv(folder / name, index=False)


def make_demo_data(folder="demo_data"):
    rng = np.random.default_rng(0)
    for kind in ["walking", "running", "still"]:
        for i in range(4):
            _fake_recording([kind], Path(folder) / kind / f"rec{i+1}", rng)
    _fake_recording(["still", "walking", "running"], Path(folder + "_test") / "mixed", rng,
                    seconds_each=8)
    print(f"Wrote fake recordings to {folder}/ and {folder}_test/mixed")


def main():
    # Work from the folder this script is in, so paths like "data" always resolve
    os.chdir(Path(__file__).resolve().parent)

    # Command-line arguments win; otherwise use the settings at the top (VS Code Run button)
    cmd = sys.argv[1] if len(sys.argv) > 1 else MODE
    target = sys.argv[2] if len(sys.argv) > 2 else None

    if cmd == "demo":
        make_demo_data()
        train("demo_data", DEMO_MODEL_PATH)
        predict("demo_data_test/mixed", DEMO_MODEL_PATH)
    elif cmd == "train":
        train(target or DATA_FOLDER)
    elif cmd == "predict":
        predict(target or RECORDING)
    elif cmd == "plot":
        plot(target or RECORDING)
    else:
        print(f'Unknown MODE "{cmd}". Use "demo", "plot", "train" or "predict".')
        print(__doc__)


if __name__ == "__main__":
    main()