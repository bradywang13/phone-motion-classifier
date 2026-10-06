# Motion classifier

Recognise what you're doing (walking, running, standing still, ...) from an
iPhone's accelerometer and gyroscope, recorded with the
[Sensor Logger](https://www.tszheichoi.com/sensorlogger) app.

Four scripts:

| Script | What it does |
| --- | --- |
| `watch_motion.py` | Random forest on hand-picked features (mean, std, dominant frequency, ...) of 2-second windows. Also plots recordings. |
| `motion_cnn.py` | 1D convolutional neural network that learns straight from the raw 2-second signal. |
| `live_motion.py` | Streams data from the phone over WiFi and classifies it in real time with either model. |
| `import_motionsense.py` | Downloads the MotionSense dataset and converts it into extra training recordings. |

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Check everything works with fake data (no phone needed):

```
python watch_motion.py demo
python motion_cnn.py demo
```

## Record your own data

1. In Sensor Logger, enable **Accelerometer** and **Gyroscope** and record each
   movement for a minute or so. Make several separate recordings per movement:
   accuracy is measured on whole recordings the model never trained on.
2. Export each recording and put it in a folder named after the movement
   (a `.zip` export is unzipped automatically):

```
data/
  walking/   2026-10-03_14-02-11/   2026-10-03_14-05-40/  ...
  running/   ...
  still/     ...
```

The first and last 3 seconds of each training recording are trimmed off
(putting the phone in and out of your pocket).

## Add the MotionSense dataset (recommended)

[MotionSense](https://github.com/mmalekzadeh/motion-sense) is 24 people with an
iPhone in their front pocket, recorded with the same iOS motion API Sensor Logger
uses. Download and convert it once:

```
python import_motionsense.py     # writes motionsense/Walking, Running, Still
```

Its walking, jogging, sitting and standing trials become Walking, Running and
Still; stairs are left out. Acceleration is converted from g to m/s² to match
Sensor Logger.

Both trainers then add it as **extra training data** (`EXTRA_FOLDERS` in
`watch_motion.py`; set it to `[]` to train on your recordings alone). Accuracy is
still measured only on your own recordings, so you see how well it works on
*your* phone. A movement you recorded only once can now be tested too.

## Train and predict

```
python watch_motion.py train                 # trains on data/, saves motion_model.joblib
python watch_motion.py predict <recording>   # label every second of a new recording
python watch_motion.py plot <recording>      # look at the raw signal

python motion_cnn.py train                   # saves motion_cnn.pt
python motion_cnn.py predict <recording>
```

A `<recording>` is a Sensor Logger export folder or `.zip`.
Training prints the accuracy and a confusion matrix on held-out recordings.

In VS Code you can instead edit `MODE` / `RECORDING` at the top of a script and
press Run.

## Live classification

1. Set `MODEL = "forest"` or `"cnn"` at the top of `live_motion.py` and run it.
2. In Sensor Logger: **Settings > Data Streaming > Enable HTTP Push**, and set
   the Push URL to the one the script prints (`http://<laptop-ip>:8000/data`).
   Phone and laptop must be on the same WiFi.
3. Start a recording. The label appears in the terminal, and in big text at
   `http://<laptop-ip>:8000/` in any browser.

To test without the phone, set `REPLAY` to a recording path and it is streamed
to the script as if it were live.
