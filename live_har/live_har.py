#Real Time Human Action Detection is done

import argparse
import collections
import os
import random
import sys
import threading
import time
import warnings

import cv2
import joblib
import numpy  as np
import pandas as pd
from scipy import signal

warnings.filterwarnings("ignore")


BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "model.pkl")
LE_PATH    = os.path.join(BASE_PATH, "label_encoder.pkl")
FEAT_PATH  = os.path.join(BASE_PATH, "feature_names.pkl")


WINDOW_SIZE = 40      # exact match to main.py
BAUD_RATE   = 115200  # exact match to rssi_logger_2.py

#coloring the activities:

ACTIVITY_COLORS = {
    "walking":       (0,   210,  80),
    "sitting":       (255, 170,   0),
    "standing":      (0,   200, 255),
    "lying":         (180,   0, 240),
    "falling":       (0,    40, 255),
    "getting up":    (0,   255, 200),
    "standing up":   (0,   255, 200),
    "sitting down":  (0,   160, 255),
    "transitioning": (160, 160, 160),
}
DEFAULT_COLOR = (200, 200, 200)


#extracting features


def extract_features(x_raw: np.ndarray) -> list:
    """
    32 features: 18 time-domain + 9 spectral + 5 autocorrelation.
    Identical to main.py - per-window z-score normalisation applied first.
    """
    x_raw = x_raw.astype(float)

    # Per-window z-score normalisation
    mu, sigma = np.mean(x_raw), np.std(x_raw)
    x = (x_raw - mu) / sigma if sigma > 1e-6 else x_raw - mu

    diff1 = np.diff(x)
    n     = len(x)

    #time-domain (18)
    td = [
        np.mean(x),
        np.std(x),
        np.min(x),
        np.max(x),
        np.max(x) - np.min(x),
        np.sqrt(np.mean(x ** 2)),
        np.median(x),
        float(pd.Series(x).skew()),
        float(pd.Series(x).kurtosis()),
        np.percentile(x, 75) - np.percentile(x, 25),
        np.sum(np.abs(diff1)),
        np.sum(x ** 2),
        int(np.sum(np.diff(np.sign(x - np.mean(x))) != 0)),
        np.max(diff1) - np.min(diff1),
        np.mean(np.abs(diff1)),
        np.std(diff1),
        np.percentile(x, 90) - np.percentile(x, 10),
        np.sum(x > np.mean(x)) / n,
    ]

    #Spectral (9)
    freqs       = np.fft.rfftfreq(n)
    fft         = np.abs(np.fft.rfft(x))
    fft_sq      = fft ** 2
    total_power = np.sum(fft_sq) + 1e-10

    dominant_freq = freqs[np.argmax(fft[1:]) + 1]
    spectral_mean = np.sum(freqs * fft_sq) / total_power
    spectral_std  = np.sqrt(
        np.sum(((freqs - spectral_mean) ** 2) * fft_sq) / total_power
    )
    psd_norm     = fft_sq / total_power
    spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    low_pow      = np.sum(fft_sq[(freqs >= 0.0) & (freqs < 0.1)]) / total_power
    mid_pow      = np.sum(fft_sq[(freqs >= 0.1) & (freqs < 0.3)]) / total_power
    high_pow     = np.sum(fft_sq[(freqs >= 0.3)])                  / total_power
    cumpower     = np.cumsum(fft_sq)
    rolloff_idx  = np.searchsorted(cumpower, 0.85 * total_power)
    rolloff      = freqs[min(rolloff_idx, len(freqs) - 1)]
    peak_ratio   = np.max(fft_sq) / (np.mean(fft_sq) + 1e-10)

    sp = [dominant_freq, spectral_mean, spectral_std, spec_entropy,
          low_pow, mid_pow, high_pow, rolloff, peak_ratio]

    #Autocorrelation (5)
    acf_full = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
    acf      = acf_full[n - 1:]
    acf_norm = acf / (acf[0] + 1e-10)
    ac1 = acf_norm[1] if n > 1 else 0.0
    ac2 = acf_norm[2] if n > 2 else 0.0
    ac4 = acf_norm[4] if n > 4 else 0.0
    ac8 = acf_norm[8] if n > 8 else 0.0
    peaks, _ = signal.find_peaks(acf_norm[1:], height=0.1)
    dominant_period = peaks[0] + 1 if len(peaks) > 0 else 0

    ac = [ac1, ac2, ac4, ac8, float(dominant_period)]

    return td + sp + ac   # 18 + 9 + 5 = 32 features

#reading the rssi signals from the ESP32 connected via Arduino IDE

class RSSIReader(threading.Thread):
    """
    Reads raw RSSI lines from the ESP32 over serial.
    rssi_logger_2.py strips the line and checks: raw.lstrip("-").isdigit()
    So each line from the ESP32 is simply a plain dBm integer, e.g. "-65\n"
    can also be seen in the Serial Monitor in the Arduino IDE after uploading the code
    """

    def __init__(self, port=None, demo=False, baud=BAUD_RATE):
        super().__init__(daemon=True)
        self.port   = port
        self.demo   = demo
        self.baud   = baud
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self.latest = None   # most recent RSSI float
        self.error  = None   # set if serial connection fails

    def get_latest(self):
        with self._lock:
            return self.latest

    def stop(self):
        self._stop.set()

    # ── thread body ──────────────────────────────────────────────────────────
    def run(self):
        if self.demo:
            self._run_demo()
        else:
            self._run_serial()

    def _run_demo(self):
        """Simulate plausible RSSI fluctuation - fills WINDOW=40 in ~3 s."""
        base = -65.0
        t    = 0.0
        while not self._stop.is_set():
            noise = random.gauss(0, 1.5)
            drift = 4.0 * np.sin(2 * np.pi * t / 80)
            with self._lock:
                self.latest = base + drift + noise
            t    += 1
            time.sleep(0.08)   # ~12 Hz

    def _run_serial(self):
        try:
            import serial
        except ImportError:
            self.error = "pyserial not installed.  Run:  pip install pyserial"
            return

        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
            print(f"[RSSI] Connected → {self.port}  @  {self.baud} baud") #baud rate set to: 115200 in the IDE
        except Exception as e:
            self.error = str(e)
            return

        while not self._stop.is_set():
            try:
                raw = ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue
                # Mirror rssi_logger_2.py validation exactly
                if raw.lstrip("-").isdigit():
                    with self._lock:
                        self.latest = float(raw)
                # Non-numeric lines (boot messages, etc.) are silently skipped
            except Exception as e:
                print(f"[RSSI] Read error: {e}")
                time.sleep(0.1)

        ser.close()

#overlay display:

def _semi_rect(img, pt1, pt2, bgr, alpha):
    #Drawing  a transparent rectangle.
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, bgr, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_overlay(frame, activity, confidence, latest_rssi,
                 rssi_history, fps, samples_collected):
    h, w  = frame.shape[:2]
    color = ACTIVITY_COLORS.get(activity.lower(), DEFAULT_COLOR)

    PANEL_H = 130
    # Bottom panel background
    _semi_rect(frame, (0, h - PANEL_H), (w, h), (12, 12, 12), 0.75)

    #RSSI waveform
    gx, gy = 15, h - PANEL_H + 10
    gw, gh = max(200, w // 4 - 20), PANEL_H - 22

    if len(rssi_history) >= 2:
        arr  = np.array(rssi_history, dtype=float)
        lo, hi = arr.min() - 1, arr.max() + 1
        span = max(hi - lo, 1.0)
        pts  = []
        n_pts = len(arr)
        for i, v in enumerate(arr):
            px = gx + int(i / max(n_pts - 1, 1) * gw)
            py = gy + gh - int((v - lo) / span * gh)
            pts.append((px, py))
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], color, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (55, 55, 55), 1)
    cv2.putText(frame, "RSSI signal",
                (gx, gy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110, 110, 110), 1)
    if latest_rssi is not None:
        cv2.putText(frame, f"{latest_rssi:.0f} dBm",
                    (gx, gy + gh + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 210, 255), 1)

    #Right side of the display: activity + confidence
    rx  = gx + gw + 25
    ry0 = h - PANEL_H + 14

    # Confidence bar
    bar_w = w - rx - 18
    bar_h = 11
    fill  = int(bar_w * confidence)
    cv2.rectangle(frame, (rx, ry0), (rx + bar_w, ry0 + bar_h), (40, 40, 40), -1)
    if fill > 0:
        cv2.rectangle(frame, (rx, ry0), (rx + fill, ry0 + bar_h), color, -1)
    cv2.putText(frame, f"Confidence:  {confidence * 100:.0f}%",
                (rx, ry0 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    # Activity label  (bottom-right, large)
    label = f"Activity:  {activity.upper()}"
    scale = min(1.1, max(0.7, w / 960))
    thick = 2
    font  = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    tx = w - tw - 18
    ty = h - PANEL_H + 65 + th
    cv2.putText(frame, label, (tx + 2, ty + 2), font, scale, (0, 0, 0),    thick + 1, cv2.LINE_AA)
    cv2.putText(frame, label, (tx,     ty),     font, scale, color,         thick,     cv2.LINE_AA)

    #Window-fill progress bar (bottom strip) 
    pct   = min(samples_collected / WINDOW_SIZE, 1.0)
    bpx   = rx
    bpy   = h - 16
    bpw   = w - rx - 18
    bph   = 6
    cv2.rectangle(frame, (bpx, bpy), (bpx + bpw, bpy + bph), (35, 35, 35), -1)
    cv2.rectangle(frame,
                  (bpx, bpy),
                  (bpx + int(bpw * pct), bpy + bph),
                  (70, 70, 70) if pct < 1.0 else color, -1)
    status_txt = f"Buffering… {samples_collected}/{WINDOW_SIZE}" if pct < 1.0 else "Live"
    cv2.putText(frame, status_txt, (bpx, bpy - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                (90, 90, 90) if pct < 1.0 else (0, 200, 90), 1)

    # FPS  (top-left)
    cv2.putText(frame, f"FPS {fps:.1f}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

    #Badge  (top-right)
    badge = "RSSI = Brain  |  Camera = Display only"
    (bw_b, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(frame, badge, (w - bw_b - 10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 185, 120), 1, cv2.LINE_AA)

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Live HAR - WiFi RSSI → model.pkl, Camera = display only"
    )
    parser.add_argument(
        "--port", default=None,
        help="ESP32 serial port (e.g. COM5  or  /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Simulated RSSI - no ESP32 needed"
    )
    parser.add_argument(
        "--cam", default=0, type=int,
        help="Camera device index (default 0)"
    )
    args = parser.parse_args()

    #Loading artefacts
    for p, name in [(MODEL_PATH, "model.pkl"),
                    (LE_PATH,    "label_encoder.pkl"),
                    (FEAT_PATH,  "feature_names.pkl")]:
        if not os.path.exists(p):
            print(f"[ERROR] {name} not found at:  {p}")
            print("        Run  main.py  first.")
            sys.exit(1)

    print("[MODEL] Loading artefacts …")
    model = joblib.load(MODEL_PATH)
    le    = joblib.load(LE_PATH)
    feats = joblib.load(FEAT_PATH)   # used only for verification
    print(f"[MODEL] {type(model).__name__}  "
          f"|  {len(feats)} features  "
          f"|  classes: {list(le.classes_)}")

    # Sanity-check feature count
    if len(feats) != 32:
        print(f"[WARN]  Expected 32 features, got {len(feats)}. "
              "Make sure this is the model trained by the current main.py.")

    #Serial / demo
    if not args.demo and args.port is None:
        print("\n[WARN] No --port given → switching to --demo mode.")
        print("       Use:  python live_har.py --port COM5\n")
        args.demo = True

    reader = RSSIReader(port=args.port, demo=args.demo)
    reader.start()
    time.sleep(0.4)

    if reader.error:
        print(f"[ERROR] Serial failed: {reader.error}")
        sys.exit(1)

    mode = "DEMO (simulated)" if args.demo else f"LIVE  port={args.port}"
    print(f"[RSSI] {mode}")

    #Camera (display only)
    #NOTE: the camera is totally seperate from feature extraction
    cap = cv2.VideoCapture(args.cam)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        print(f"[CAM]  Camera {args.cam} opened  -  display only, zero prediction role")
    else:
        cap = None
        print(f"[CAM]  Camera {args.cam} unavailable - using blank frame")

    
    window       = collections.deque(maxlen=WINDOW_SIZE)  # RSSI rolling buffer
    rssi_history = collections.deque(maxlen=200)           # graph history

    activity   = "Waiting…"
    confidence = 0.0
    prev_time  = time.time()
    fps        = 0.0

    print(f"\n[LIVE] System running.  Window = {WINDOW_SIZE} samples.  "
          "Press  Q  or  ESC  to quit.\n")

    while True:
        # 1. Camera frame  (display only - no prediction happens here)
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        else:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 2. Collect RSSI from background thread
        rssi_val = reader.get_latest()
        if rssi_val is not None:
            window.append(rssi_val)
            rssi_history.append(rssi_val)

        # 3. Predict - RSSI only, identical pipeline to tester.py
        if len(window) == WINDOW_SIZE:
            feat_vec = np.array(
                extract_features(np.array(window, dtype=float))
            ).reshape(1, -1)
            try:
                enc_pred   = model.predict(feat_vec)[0]
                activity   = le.inverse_transform([enc_pred])[0]
                if hasattr(model, "predict_proba"):
                    confidence = float(model.predict_proba(feat_vec)[0].max())
                else:
                    confidence = 1.0
            except Exception as e:
                activity, confidence = "Error", 0.0
                print(f"[PREDICT] {e}")

        # 4. FPS
        now       = time.time()
        fps       = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
        prev_time = now

        # 5. Render overlay
        frame = draw_overlay(
            frame,
            activity          = activity,
            confidence        = confidence,
            latest_rssi       = rssi_val,
            rssi_history      = rssi_history,
            fps               = fps,
            samples_collected = len(window),
        )

        cv2.imshow("Live HAR  |  RSSI Brain  +  Camera Display       (Q / ESC to quit)",
                   frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    # Cleanup
    print("[DONE] Stopping…")
    reader.stop()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Bye.")


if __name__ == "__main__":
    main()