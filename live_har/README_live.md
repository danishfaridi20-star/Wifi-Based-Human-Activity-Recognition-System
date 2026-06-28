# Live Demo — Real-time Walking Detection

Binary classifier running on live RSSI from ESP32, with camera overlay.

## Folder contents

```
live_demo/
├── train_binary.py     ← run ONCE to train model.pkl from session1-9.csv
├── live_har.py         ← your existing live demo (unchanged)
├── model.pkl           ← produced by train_binary.py
├── label_encoder.pkl   ← produced by train_binary.py
└── feature_names.pkl   ← produced by train_binary.py
```

## One-time setup

```
pip install opencv-python numpy pandas scipy joblib pyserial scikit-learn
python train_binary.py
```

Expect to see "MEAN LOSO ACCURACY: 81.26%". This is the honest cross-session accuracy — what you can quote to the panel.

## Running the demo

Plug in ESP32, find its COM port in Device Manager, then:

```
python live_har.py --port COM5
```

(Replace COM5 with whatever port your ESP32 is on — likely COM3-COM8 on Windows, `/dev/ttyUSB0` or `/dev/ttyACM0` on Linux.)

No ESP32 available? Use `--demo` for a simulated signal:

```
python live_har.py --demo
```

Press **Q** or **ESC** to quit.

## What the demo shows

- **Camera feed** — just for display, plays no role in classification
- **RSSI waveform** (bottom-left) — live plot of the signal the model sees
- **Activity label** (bottom-right) — `WALKING` or `NOT_WALKING` with confidence %
- **Buffering bar** — takes ~3 seconds to fill the 40-sample window before first prediction

## If you want to retrain with different data

Add more session CSVs to `../Data/`, re-run `python train_binary.py`. Takes about 10 seconds.

## For the panel

Quote this:

> *"Our live demo runs binary walking-detection on streaming RSSI from a single ESP32. The binary classifier achieves **81% accuracy under honest Leave-One-Session-Out cross-validation** — deliberately tested on sessions the model has never seen. The full five-class problem drops to 43% under the same evaluation (Part 2), which is why we collapsed to binary for the live demonstration — it stays reliable across environmental changes."*

Honest, specific, and pre-empts the "isn't it easier with binary?" question by framing it as a deliberate engineering choice.

## Why binary, not 5-class, for the live demo?

The 5-class RSSI model (Part 1/2) works in matched conditions but degrades under domain shift — standing vs sitting vs lying is a narrow discrimination that depends on absolute RSSI levels, which drift between sessions. Binary walking-vs-static sidesteps this entirely because it hinges on signal periodicity (walking has a stride rhythm ~1-2 Hz, static activities don't), which is invariant to baseline drift. Top model features confirm it: `ac1`, `ac2` (autocorrelation) and `fft_low_pow` dominate.

## CSI on the same ESP32? (anticipated question)

Possible but requires flashing ESP32-CSI-Tool firmware and re-collecting training data. Not feasible overnight, but a natural future-work slide: *"same hardware, CSI-quality signal, zero additional cost."*
