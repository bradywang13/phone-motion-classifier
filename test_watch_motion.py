"""Run with: python test_watch_motion.py"""

import numpy as np
import pandas as pd

import watch_motion as wm


def test_domfreq_finds_a_2hz_sine():
    fs = 100
    t = np.arange(0, 2, 1 / fs)
    win = pd.DataFrame({"acc_mag": 5 + np.sin(2 * np.pi * 2 * t)})  # offset checks the mean is removed
    assert wm.window_features(win, fs)["acc_mag_domfreq"] == 2.0


def test_load_recording_pairs_accel_with_gyro(tmp_path=None):
    import tempfile
    from pathlib import Path
    folder = Path(tmp_path or tempfile.mkdtemp())
    t = np.arange(0, 1, 0.01)
    pd.DataFrame({"seconds_elapsed": t, "x": 1.0, "y": 2.0, "z": 2.0}).to_csv(folder / "Accelerometer.csv", index=False)
    pd.DataFrame({"seconds_elapsed": t + 0.003, "x": 0.0, "y": 0.0, "z": 0.0}).to_csv(folder / "Gyroscope.csv", index=False)
    times, data = wm.load_recording(folder)
    assert len(data) == len(t) and times[0] == 0
    assert np.allclose(data["acc_mag"], 3.0)  # sqrt(1 + 4 + 4)


if __name__ == "__main__":
    test_domfreq_finds_a_2hz_sine()
    test_load_recording_pairs_accel_with_gyro()
    print("all tests passed")
