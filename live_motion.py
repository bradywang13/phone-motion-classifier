"""
live_motion.py - classify your movement in real time while you move.

Sensor Logger streams accelerometer + gyroscope data over WiFi to this script
("HTTP Push"). Every second, the script takes the last 2 seconds of data and
runs your trained model on it.

Put this file next to watch_motion.py. Train a model first (watch_motion.py,
MODE = "train"). For the neural network, also train motion_cnn.py.

In VS Code: set the settings below, save, press Run. Stop with Ctrl+C in the terminal.
"""

import json
import os
import socket
import threading
import time
import urllib.request
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import watch_motion as wm

# ================= SETTINGS (edit these, save, then press Run) =================
MODEL = "cnn"   # "forest" = motion_model.joblib, "cnn" = motion_cnn.pt
REPLAY = ""        # "" = listen to your phone. Or a recording path, e.g. "test/2026-10-03_22-09-37",
                   #      to replay it as if it were streaming live (test without the phone)
PORT = 8000        # must match the port in Sensor Logger's Push URL
UPDATE_SECONDS = 1.0   # how often to make a prediction
SMOOTH = 3         # show the most common of the last N predictions (1 = no smoothing)
MIN_CONFIDENCE = 0.5   # below this, show "unsure"
# ================================================================================

WINDOW_SECONDS = wm.WINDOW_SECONDS
acc_buffer = deque(maxlen=5000)    # (time in seconds, x, y, z)
gyro_buffer = deque(maxlen=5000)
lock = threading.Lock()
status = {"label": "waiting for data", "conf": 0.0, "packets": 0, "names": set()}


# ---------------- 1. Receive data from the phone ----------------
class Receiver(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            message = json.loads(self.rfile.read(length) or b"{}")
            readings = message.get("payload", [])
        except (ValueError, AttributeError):
            readings = []
        with lock:
            for r in readings:
                name = str(r.get("name", "")).lower()
                status["names"].add(name)
                v = r.get("values", {})
                if name in ("accelerometer", "gyroscope") and {"x", "y", "z"} <= v.keys():
                    row = (r["time"] / 1e9, float(v["x"]), float(v["y"]), float(v["z"]))
                    (acc_buffer if name == "accelerometer" else gyro_buffer).append(row)
            status["packets"] += 1
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_GET(self):
        """Open http://<laptop-ip>:8000/ in any browser to see the live label in big text."""
        page = (f"<html><head><meta http-equiv='refresh' content='1'>"
                f"<meta name='viewport' content='width=device-width'></head>"
                f"<body style='font-family:sans-serif;text-align:center;margin-top:30vh'>"
                f"<div style='font-size:14vw;font-weight:bold'>{status['label']}</div>"
                f"<div style='font-size:5vw;color:gray'>{status['conf']:.0%} sure</div>"
                f"</body></html>")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(page.encode())

    def log_message(self, *args):   # keep the terminal quiet
        pass


def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))     # no data is sent; this just finds the WiFi address
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


# ---------------- 2. Build the latest window ----------------
def latest_window():
    """Last WINDOW_SECONDS of data as (times, DataFrame) in the same format as training."""
    with lock:
        acc, gyro = list(acc_buffer), list(gyro_buffer)
    if len(acc) < 10 or len(gyro) < 10:
        return None
    acc = pd.DataFrame(acc, columns=["t", "acc_x", "acc_y", "acc_z"]).sort_values("t")
    gyro = pd.DataFrame(gyro, columns=["t", "gyro_x", "gyro_y", "gyro_z"]).sort_values("t")
    end = acc["t"].iloc[-1]
    acc = acc[acc["t"] > end - WINDOW_SECONDS]
    if acc["t"].iloc[-1] - acc["t"].iloc[0] < 0.9 * WINDOW_SECONDS:
        return None                                  # not enough data yet
    data = pd.merge_asof(acc, gyro, on="t", direction="nearest", tolerance=0.05).dropna()
    if len(data) < 10:
        return None
    t = data.pop("t").to_numpy()
    data["acc_mag"] = np.sqrt(data.acc_x**2 + data.acc_y**2 + data.acc_z**2)
    data["gyro_mag"] = np.sqrt(data.gyro_x**2 + data.gyro_y**2 + data.gyro_z**2)
    return t, data.reset_index(drop=True)


# ---------------- 3. Load a model and make predictions ----------------
def load_predictor():
    if MODEL == "cnn":
        import torch
        import motion_cnn as mc
        ckpt = torch.load(mc.MODEL_PATH)
        net = mc.MotionCNN(len(mc.CHANNELS), len(ckpt["labels"]))
        net.load_state_dict(ckpt["state"])
        net.eval()
        mean, std = np.array(ckpt["mean"]), np.array(ckpt["std"])

        def predict(t, data):
            # Same spacing as training: win_n samples exactly 1/FS apart, ending now
            win_n = int(mc.WINDOW_SECONDS * mc.FS)
            t_even = t[-1] - np.arange(win_n)[::-1] / mc.FS
            x = np.stack([np.interp(t_even, t, data[c].to_numpy()) for c in mc.CHANNELS])
            x = ((x - mean[:, None]) / std[:, None]).astype(np.float32)[None]
            with torch.no_grad():
                probs = torch.softmax(net(torch.tensor(x)), dim=1).numpy()[0]
            return ckpt["labels"][probs.argmax()], probs.max()
        return predict, ckpt["labels"], mc.MODEL_PATH

    bundle = joblib.load(wm.MODEL_PATH)
    model = bundle["model"]

    def predict(t, data):
        fs = 1 / np.median(np.diff(t))
        X = pd.DataFrame([wm.window_features(data, fs)])[bundle["columns"]]
        probs = model.predict_proba(X)[0]
        return model.classes_[probs.argmax()], probs.max()
    return predict, list(model.classes_), wm.MODEL_PATH


# ---------------- Replay a recording as if it were live ----------------
def replay(path, url):
    folder = wm.as_folder(path)
    files = {"accelerometer": wm.find_csv(folder, "accelerometer"),
             "gyroscope": wm.find_csv(folder, "gyroscope")}
    rows = []
    for name, f in files.items():
        df = pd.read_csv(f)
        for tm, x, y, z in zip(df["time"], df["x"], df["y"], df["z"]):
            rows.append({"name": name, "time": int(tm), "values": {"x": x, "y": y, "z": z}})
    rows.sort(key=lambda r: r["time"])
    start_data, start_clock = rows[0]["time"] / 1e9, time.time()
    i, batch = 0, 0.2
    print(f"Replaying {path} ({(rows[-1]['time'] - rows[0]['time']) / 1e9:.0f} s) ...")
    while i < len(rows):
        now_data = start_data + (time.time() - start_clock)
        chunk = []
        while i < len(rows) and rows[i]["time"] / 1e9 <= now_data:
            chunk.append(rows[i])
            i += 1
        if chunk:
            body = json.dumps({"messageId": 0, "payload": chunk}).encode()
            req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
            urllib.request.urlopen(req).read()
        time.sleep(batch)
    print("Replay finished.")


def main():
    os.chdir(Path(__file__).resolve().parent)
    predict, labels, path = load_predictor()
    print(f"Loaded {path} with movements: {', '.join(labels)}")

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    if REPLAY:
        threading.Thread(target=replay, args=(REPLAY, f"http://127.0.0.1:{PORT}/data"),
                         daemon=True).start()
    else:
        ip = local_ip()
        print("\nIn Sensor Logger: Settings > Data Streaming > Enable HTTP Push")
        print(f"  Push URL:  http://{ip}:{PORT}/data")
        print(f"Live label in a browser:  http://{ip}:{PORT}/")
        print("Then start a recording in Sensor Logger. Stop this script with Ctrl+C.\n")

    recent = deque(maxlen=max(1, SMOOTH))
    last_data_time = time.time()
    last_packets = 0
    started = time.time()
    last_note = 0.0
    connected = False
    try:
        while True:
            time.sleep(UPDATE_SECONDS)

            # ---- status messages, so you can see where things are stuck ----
            if status["packets"] == 0:
                if time.time() - last_note > 5:
                    print(f"  waiting for the phone... (nothing received after "
                          f"{time.time() - started:.0f} s)")
                    last_note = time.time()
                continue
            if not connected:
                print("  connected: receiving data from the phone")
                connected = True
            if not acc_buffer or not gyro_buffer:
                if time.time() - last_note > 5:
                    seen = ", ".join(sorted(n for n in status["names"] if n)) or "nothing yet"
                    print(f"  receiving messages, but no accelerometer + gyroscope data. "
                          f"Sensors seen: {seen}")
                    last_note = time.time()
            # -----------------------------------------------------------------

            if status["packets"] != last_packets:
                last_packets, last_data_time = status["packets"], time.time()
            elif time.time() - last_data_time > 3:
                if status["label"] != "waiting for data":
                    print("  ... no data from the phone (is it still recording?)")
                status.update(label="waiting for data", conf=0.0)
                recent.clear()
                with lock:                       # start fresh for the next recording
                    acc_buffer.clear()
                    gyro_buffer.clear()
                continue

            win = latest_window()
            if win is None:
                continue
            label, conf = predict(*win)
            recent.append(label if conf >= MIN_CONFIDENCE else "unsure")
            shown = Counter(recent).most_common(1)[0][0]
            status.update(label=shown, conf=float(conf))
            bar = "#" * int(conf * 20)
            print(f"{time.strftime('%H:%M:%S')}  {shown:<14} {conf:4.0%} {bar}")
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
