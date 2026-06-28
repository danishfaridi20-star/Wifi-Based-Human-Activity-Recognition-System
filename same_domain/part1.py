import os, warnings
import pandas as pd, numpy as np, joblib
from scipy import stats, signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

BASE_PATH   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_PATH, "Data")
WINDOW_SIZE = 40
STEP_SIZE   = 10
TRAIN_SIDS  = [1, 3]
TEST_SID    = 2

# Extracting the features:

def extract_features(x_raw, session_baseline):
    x_raw = x_raw.astype(float); n = len(x_raw)
    x_rel = x_raw - session_baseline
    mu, sigma = np.mean(x_raw), np.std(x_raw)
    x_z = (x_raw - mu) / sigma if sigma > 1e-6 else x_raw - mu
    diff_raw = np.diff(x_raw)
    raw_feats = [
        float(np.std(x_raw)), float(np.max(x_raw) - np.min(x_raw)),
        float(np.percentile(x_raw,75) - np.percentile(x_raw,25)),
        float(np.median(np.abs(x_raw - np.median(x_raw)))),
        float(np.mean(np.abs(diff_raw))), float(np.std(diff_raw)),
        float(np.max(np.abs(diff_raw))), float(np.sum(np.abs(diff_raw))),
        int(np.sum(np.diff(np.sign(x_raw - np.mean(x_raw))) != 0))]
    rel_feats = [float(np.mean(x_rel)), float(np.median(x_rel)),
                 float(np.percentile(x_rel,25)), float(np.percentile(x_rel,75))]
    shape_feats = [float(pd.Series(x_z).skew()), float(pd.Series(x_z).kurtosis()),
                   float(np.percentile(x_z,90)-np.percentile(x_z,10)),
                   float(np.sum(x_z > 0) / n)]
    freqs = np.fft.rfftfreq(n); fft = np.abs(np.fft.rfft(x_z)); fft_sq = fft**2
    total = np.sum(fft_sq) + 1e-10
    dom_f = float(freqs[np.argmax(fft[1:]) + 1])
    s_mean = float(np.sum(freqs * fft_sq) / total)
    s_std = float(np.sqrt(np.sum(((freqs - s_mean)**2) * fft_sq) / total))
    pnorm = fft_sq / total
    s_ent = float(-np.sum(pnorm * np.log2(pnorm + 1e-10)))
    low_p = float(np.sum(fft_sq[(freqs>=0.0)&(freqs<0.1)])/total)
    mid_p = float(np.sum(fft_sq[(freqs>=0.1)&(freqs<0.3)])/total)
    hig_p = float(np.sum(fft_sq[(freqs>=0.3)])/total)
    cum = np.cumsum(fft_sq)
    rolloff = float(freqs[min(np.searchsorted(cum, 0.85*total), len(freqs)-1)])
    p_ratio = float(np.max(fft_sq)/(np.mean(fft_sq)+1e-10))
    spec_feats = [dom_f, s_mean, s_std, s_ent, low_p, mid_p, hig_p, rolloff, p_ratio]
    acf_full = np.correlate(x_z - np.mean(x_z), x_z - np.mean(x_z), mode='full')
    acf = acf_full[n-1:]; acn = acf / (acf[0] + 1e-10)
    ac1 = float(acn[1]) if n > 1 else 0.0
    ac2 = float(acn[2]) if n > 2 else 0.0
    ac4 = float(acn[4]) if n > 4 else 0.0
    ac8 = float(acn[8]) if n > 8 else 0.0
    peaks, _ = signal.find_peaks(acn[1:], height=0.1)
    period = float(peaks[0]+1) if len(peaks) > 0 else 0.0
    return raw_feats + rel_feats + shape_feats + spec_feats + [ac1, ac2, ac4, ac8, period]

FEATURE_NAMES = ["raw_std","raw_range","raw_iqr","raw_mad","raw_mean_abs_diff",
    "raw_std_diff","raw_max_abs_diff","raw_total_var","raw_zero_cross",
    "rel_mean","rel_median","rel_q25","rel_q75",
    "shape_skew","shape_kurt","shape_90_10","shape_frac_pos",
    "fft_dom","fft_sm","fft_ss","fft_ent","fft_low","fft_mid","fft_high",
    "fft_rolloff","fft_peak_ratio","ac1","ac2","ac4","ac8","ac_period"]

#fixing timestamps

def fix_ts(s):
    return pd.to_datetime(
        s.astype(str).str.replace(r"(\d{2}/\d{2})(\d{4})", r"\1/\2", regex=True),
        format="%d/%m/%Y %H:%M:%S.%f")

#loading the session till n terms from the folder "Data"

def load_session(sid):
    f = os.path.join(DATA_DIR, f"session{sid}.csv")
    df = pd.read_csv(f, encoding="utf-8-sig"); df.columns = df.columns.str.strip()
    df.rename(columns={df.columns[0]:"timestamp", df.columns[1]:"rssi",
                        df.columns[2]:"label"}, inplace=True)
    df["timestamp"] = fix_ts(df["timestamp"])
    df["rssi"] = pd.to_numeric(df["rssi"], errors="coerce")
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["rssi","label"]).reset_index(drop=True)
    zs = np.abs(stats.zscore(df["rssi"])); df = df[zs < 3].reset_index(drop=True)
    return df

#Preparing windows:

def build_windows(df, baseline, include_labels=True):
    rows = []
    r = df["rssi"].values
    l = df["label"].values if include_labels else [None] * len(df)
    for start in range(0, len(r)-WINDOW_SIZE, STEP_SIZE):
        w = r[start:start+WINDOW_SIZE]
        feats = extract_features(w, baseline)
        maj = pd.Series(l[start:start+WINDOW_SIZE]).mode()[0] if include_labels else None
        rows.append({"feats": feats, "label": maj, "start": start})
    return rows

#training the model on these sessions

print("=" * 70)
print(f"PART 1 - MATCHED-CONDITIONS RSSI ACTIVITY RECOGNITION")
print(f"    Train: sessions {TRAIN_SIDS}  |  Test: session {TEST_SID}")
print("=" * 70)

print("\n--- Loading training sessions ---")
train_rows = []
for sid in TRAIN_SIDS:
    df = load_session(sid)
    baseline = float(np.median(df["rssi"].values))
    windows = build_windows(df, baseline)
    train_rows.extend(windows)
    print(f"  session{sid}: {len(df)} rows, {len(windows)} windows, "
          f"mean RSSI={df['rssi'].mean():.1f}dBm")

X_tr = np.array([r["feats"] for r in train_rows])
y_tr = np.array([r["label"] for r in train_rows])
print(f"  Training matrix: X_tr={X_tr.shape}  |  labels: {sorted(set(y_tr))}")

#training Random Forest

print("\n--- Training Random Forest ---")
le = LabelEncoder().fit(y_tr)
model = RandomForestClassifier(
    n_estimators=500, class_weight="balanced_subsample",
    random_state=42, n_jobs=-1)
model.fit(X_tr, le.transform(y_tr))
print(f"  Trained RF on {len(X_tr)} windows, {len(FEATURE_NAMES)} features")
print(f"  Classes: {list(le.classes_)}")

#Evaluating on sessions that were held-out 

print(f"\n--- Evaluating on held-out session{TEST_SID} ---")
df_test = load_session(TEST_SID)
test_baseline = float(np.median(df_test["rssi"].values))
test_windows = build_windows(df_test, test_baseline, include_labels=False)

# Row-level prediction via majority vote over overlapping windows

n = len(df_test); row_votes = [[] for _ in range(n)]
for w in test_windows:
    pred = le.inverse_transform(model.predict(np.array([w["feats"]])))[0]
    for i in range(w["start"], w["start"] + WINDOW_SIZE):
        row_votes[i].append(pred)

last = le.classes_[0]; pred_labels = []
for votes in row_votes:
    if votes:
        lbl = pd.Series(votes).mode()[0]; last = lbl
    else:
        lbl = last
    pred_labels.append(lbl)

y_true = df_test["label"].values
y_pred = np.array(pred_labels)

#Results:

acc = accuracy_score(y_true, y_pred)
print("\n" + "=" * 70)
print(f"  ROW-LEVEL ACCURACY ON SESSION {TEST_SID}: {acc*100:.2f}%")
print("=" * 70)

print("\nPer-class performance:")
print(classification_report(y_true, y_pred, zero_division=0))

print("Confusion matrix:")
classes = sorted(set(y_true) | set(y_pred))
cm = confusion_matrix(y_true, y_pred, labels=classes)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
cm_df.index.name = "true \\ pred"
print(cm_df.to_string())

# Feature importance
fi = pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
print("\nTop 8 most informative features:")
print(fi.head(8).round(4).to_string())

#saving model and the result file in pkl and csv formats respectively

out_df = df_test.copy()
out_df["pred_label"] = y_pred
out_df["match"] = (out_df["label"] == out_df["pred_label"])
out_path = os.path.join(BASE_PATH, "part1_results.csv")
out_df[["timestamp","rssi","label","pred_label","match"]].to_csv(out_path, index=False)

joblib.dump(model, os.path.join(BASE_PATH, "part1_model.pkl"))
joblib.dump(le,    os.path.join(BASE_PATH, "part1_encoder.pkl"))
print(f"\nSaved: part1_model.pkl, part1_encoder.pkl, part1_results.csv")
print("\n>>> Part 1 complete. Headline number: {:.1f}% <<<".format(acc*100))