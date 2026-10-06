"""
motion_cnn.py - step 2: a neural network that learns straight from the raw
accelerometer + gyroscope signal.

watch_motion.py (random forest) needs YOU to decide which numbers describe a
window (mean, std, dominant frequency...). This script gives a 1D convolutional
neural network (CNN) the raw 2-second signal and lets it learn its own patterns.

Put this file in the same folder as watch_motion.py (it reuses its loading code).
In VS Code: set MODE below, save, press Run.

Install once:  pip install -r requirements.txt
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import watch_motion as wm   # reuse the loading code (zip/folder/CSV, merging accel + gyro)

# ================= WHAT TO RUN (edit these, save, then press Run) =================
MODE = "predict"          # "demo", "train" or "predict"
DATA_FOLDER = "data"   # same folder watch_motion.py trains on
RECORDING = "test/2026-10-03_22-09-37"   # used by "predict"
# ==================================================================================

# ---------------- Settings you might tweak ----------------
FS = 50                # every recording is resampled to 50 samples per second
WINDOW_SECONDS = 2.0   # length of each chunk the network looks at
STEP_SECONDS = 0.5     # windows overlap more than in watch_motion -> more examples
EPOCHS = 15            # how many passes over the training data
BATCH_SIZE = 64        # windows per training step
LEARNING_RATE = 1e-3   # how big each learning step is
TEST_FRACTION = 0.25   # share of each movement's recordings held back for testing
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "acc_mag", "gyro_mag"]
MODEL_PATH = Path("motion_cnn.pt")
DEMO_MODEL_PATH = Path("demo_cnn.pt")
EXTRA_FOLDERS = wm.EXTRA_FOLDERS   # extra TRAINING data, set in watch_motion.py ([] = your data only)
EXTRA_MAX_WINDOWS = 60             # per extra recording: plenty of variety, much faster training


# ---------------- 1. Raw windows ----------------
def recording_to_raw_windows(path, trim):
    """One recording -> array (windows, channels, samples) and each window's start time."""
    t, s = wm.load_recording(path)
    keep = (t >= trim) & (t <= t[-1] - trim)
    t, s = t[keep], s[keep]
    win_n = int(WINDOW_SECONDS * FS)
    step_n = int(STEP_SECONDS * FS)
    if len(t) < 2:
        return np.empty((0, len(CHANNELS), win_n), np.float32), np.empty(0)

    # Resample to exactly FS samples/sec so every window is the same size
    t_even = np.arange(t[0], t[-1], 1 / FS)
    signal = np.stack([np.interp(t_even, t, s[c].to_numpy()) for c in CHANNELS])

    starts = list(range(0, signal.shape[1] - win_n + 1, step_n))
    if not starts:
        return np.empty((0, len(CHANNELS), win_n), np.float32), np.empty(0)
    windows = np.stack([signal[:, i:i + win_n] for i in starts]).astype(np.float32)
    return windows, t_even[starts]


def build_dataset(data_dir, quiet=False, max_windows=None):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        sys.exit(f"Folder '{data_dir}' not found.")
    X, y, groups = [], [], []
    for label_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for rec in wm.find_recordings(label_dir):
            try:
                w, _ = recording_to_raw_windows(rec, wm.TRIM_SECONDS)
                if max_windows and len(w) > max_windows:   # evenly spaced subset
                    w = w[np.linspace(0, len(w) - 1, max_windows).astype(int)]
            except ValueError as e:
                print(f"  skipped {rec.name}: {e}")
                continue
            if len(w) == 0:
                print(f"  skipped {rec.name}: too short")
                continue
            X.append(w)
            y += [label_dir.name] * len(w)
            groups += [str(rec)] * len(w)
            if not quiet:
                print(f"  {label_dir.name:<12} {rec.name:<28} {len(w):>4} windows")
    if not X:
        sys.exit(f"No recordings found in {data_dir}/<movement>/")
    return np.concatenate(X), np.array(y), np.array(groups)


# ---------------- 2. The network ----------------
class MotionCNN(nn.Module):
    """Three layers of 1D convolutions: each slides small pattern detectors along time."""

    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),      # average each pattern detector over the window
            nn.Flatten(),
            nn.Dropout(0.3),              # randomly ignore some detectors -> less overfitting
            nn.Linear(64, n_classes),     # one score per movement
        )

    def forward(self, x):
        return self.net(x)


def augment(xb):
    """Small random changes so the network doesn't memorise exact recordings."""
    scale = 1 + 0.1 * torch.randn(xb.shape[0], xb.shape[1], 1)
    return xb * scale + 0.05 * torch.randn_like(xb)


def fit(X, y_idx, n_classes, X_test=None, y_test=None):
    """Train a fresh network. Prints progress each epoch."""
    torch.manual_seed(0)
    model = MotionCNN(X.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    counts = np.bincount(y_idx, minlength=n_classes)
    weights = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)      # balances movements with less data

    X_t, y_t = torch.tensor(X), torch.tensor(y_idx)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(X_t))
        total = 0.0
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(augment(X_t[idx])), y_t[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        msg = f"  epoch {epoch:3d}/{EPOCHS}   training loss {total / len(order):.3f}"
        if X_test is not None and (epoch % 5 == 0 or epoch == EPOCHS):
            acc = (predict_idx(model, X_test) == y_test).mean()
            msg += f"   test accuracy {acc:.0%}"
        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(msg)
    return model


def predict_idx(model, X, return_conf=False):
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(X)), dim=1).numpy()
    return (probs.argmax(1), probs.max(1)) if return_conf else probs.argmax(1)


# ---------------- 3. Train ----------------
def train(data_dir, model_path=MODEL_PATH, extra_dirs=EXTRA_FOLDERS):
    print(f"Loading {data_dir}/ ...")
    X, y, groups = build_dataset(data_dir)
    labels = sorted(str(v) for v in set(y))   # plain strings so the file loads safely
    if len(labels) < 2:
        sys.exit("Need at least 2 movements to train.")
    own_recs = set(groups)

    # Extra recordings only ever go into training, never into the test
    used_extra = []
    for d in extra_dirs:
        if not Path(d).is_dir():
            print(f"\n(No '{d}' folder - run import_motionsense.py to add it. Continuing without.)")
            continue
        print(f"\nLoading extra training data from {d}/ ...")
        Xe, ye, ge = build_dataset(d, quiet=True, max_windows=EXTRA_MAX_WINDOWS)
        keep = np.isin(ye, labels)
        X, y, groups = np.concatenate([X, Xe[keep]]), np.concatenate([y, ye[keep]]), np.concatenate([groups, ge[keep]])
        print(f"  {len(set(ge[keep]))} recordings, {keep.sum()} windows")
        used_extra.append(str(d))
    y_idx = np.array([labels.index(v) for v in y])

    # Hold back whole recordings of YOURS for an honest test (same idea as watch_motion.py).
    # With extra data, a movement you recorded only once can still be tested.
    rng = np.random.default_rng(0)
    test_recs = []
    for label in labels:
        recs = sorted(r for r in set(groups[y == label]) if r in own_recs)
        if len(recs) >= 2 or (used_extra and recs):
            n_test = max(1, round(TEST_FRACTION * len(recs)))
            test_recs += list(rng.choice(recs, n_test, replace=False))
    is_test = np.isin(groups, test_recs)
    can_test = all(is_test[y == l].any() and (~is_test[y == l]).any() for l in labels)

    # Scale each channel to mean 0, std 1 (networks learn much better this way)
    train_part = X[~is_test] if can_test else X
    mean = train_part.mean(axis=(0, 2))
    std = train_part.std(axis=(0, 2)) + 1e-6
    Xn = ((X - mean[None, :, None]) / std[None, :, None]).astype(np.float32)

    if can_test:
        print(f"\nTraining on {(~is_test).sum()} windows, testing on {is_test.sum()} windows "
              f"from {len(test_recs)} held-back recordings ...")
        model = fit(Xn[~is_test], y_idx[~is_test], len(labels), Xn[is_test], y_idx[is_test])
        preds = predict_idx(model, Xn[is_test])
        y_true = [labels[i] for i in y_idx[is_test]]
        y_pred = [labels[i] for i in preds]
        print("\n=== Accuracy on recordings the network never saw ===")
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
        print("Confusion matrix (rows = what you did, columns = what the network said):")
        print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                           index=labels, columns=labels))
    else:
        print("\nNeed 2+ recordings per movement to test. Skipping the test.")

    print("\nTraining the final network on all recordings ...")
    model = fit(Xn, y_idx, len(labels))
    torch.save({"state": model.state_dict(), "labels": labels,
                "mean": mean.tolist(), "std": std.tolist(),
                "trained_on": " + ".join([str(data_dir)] + used_extra)}, model_path)
    print(f"Saved network to {model_path}")


# ---------------- 4. Predict ----------------
def predict(path, model_path=MODEL_PATH):
    if not model_path.exists():
        sys.exit(f"No {model_path} here. Run with MODE = \"train\" first.")
    if not Path(path).exists():
        sys.exit(f"Recording '{path}' not found. Check RECORDING at the top of the script.")
    ckpt = torch.load(model_path)
    labels = ckpt["labels"]
    print(f"Using network trained on '{ckpt['trained_on']}' with movements: {', '.join(labels)}")
    model = MotionCNN(len(CHANNELS), len(labels))
    model.load_state_dict(ckpt["state"])

    windows, starts = recording_to_raw_windows(path, trim=0)
    if len(windows) == 0:
        sys.exit("Recording too short.")
    mean, std = np.array(ckpt["mean"]), np.array(ckpt["std"])
    Xn = ((windows - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
    idx, conf = predict_idx(model, Xn, return_conf=True)
    preds = [labels[i] for i in idx]

    print(f"\nPredictions for {path}:")
    for t0, p, c in zip(starts, preds, conf):
        print(f"  {t0:6.1f}s  {p:<12} ({c:.0%} sure)")

    segments = []
    for t0, p in zip(starts, preds):
        if segments and segments[-1][2] == p:
            segments[-1][1] = t0 + WINDOW_SECONDS
        else:
            if segments:
                segments[-1][1] = t0
            segments.append([t0, t0 + WINDOW_SECONDS, p])
    print("\nSummary:")
    for start, end, p in segments:
        print(f"  {start:6.1f}s - {end:6.1f}s  {p}")


def main():
    os.chdir(Path(__file__).resolve().parent)
    cmd = sys.argv[1] if len(sys.argv) > 1 else MODE
    target = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == "demo":
        wm.make_demo_data()
        train("demo_data", DEMO_MODEL_PATH, extra_dirs=[])
        predict("demo_data_test/mixed", DEMO_MODEL_PATH)
    elif cmd == "train":
        train(target or DATA_FOLDER)
    elif cmd == "predict":
        predict(target or RECORDING)
    else:
        print(f'Unknown MODE "{cmd}". Use "demo", "train" or "predict".')


if __name__ == "__main__":
    main()
