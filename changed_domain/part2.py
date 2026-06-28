#this program is for changed domain accuracy finding

import os, warnings
import pandas as pd, numpy as np, joblib
from scipy import stats, signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

BASE_PATH   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_PATH, "..", "Data")
WINDOW_SIZE = 40
STEP_SIZE   = 10

TRAIN_SIDS  = [1, 3]
# Reference: session 2 is same-day
# Cross-domain test sessions: sessions 4-9
REFERENCE_SID = 2
TEST_SIDS     = [4, 5, 6, 7, 8, 9]

DAY_OF_SESSION = {1: "15/04", 2: "15/04", 3: "15/04",
                  4: "17/04", 5: "17/04", 6: "17/04",
                  7: "19/04", 8: "19/04", 9: "19/04"}

#extracting features
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

#loading sessions from the "Data" folder
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

#building windows
def build_windows(df, baseline, include_labels=True):
    rows = []; r = df["rssi"].values
    l = df["label"].values if include_labels else [None] * len(df)
    for start in range(0, len(r)-WINDOW_SIZE, STEP_SIZE):
        feats = extract_features(r[start:start+WINDOW_SIZE], baseline)
        maj = pd.Series(l[start:start+WINDOW_SIZE]).mode()[0] if include_labels else None
        rows.append({"feats": feats, "label": maj, "start": start})
    return rows

#this function predicts in row-level via majority vote over overlapping windows.
def predict_rows(model, le, df, baseline):
    windows = build_windows(df, baseline, include_labels=False)
    n = len(df); votes = [[] for _ in range(n)]
    for w in windows:
        pred = le.inverse_transform(model.predict(np.array([w["feats"]])))[0]
        for i in range(w["start"], w["start"] + WINDOW_SIZE):
            votes[i].append(pred)
    last = le.classes_[0]; out = []
    for v in votes:
        if v: lbl = pd.Series(v).mode()[0]; last = lbl
        else: lbl = last
        out.append(lbl)
    return np.array(out)

#training the model

print("=" * 72)
print("PART 2 — CHANGED-DOMAIN EVALUATION (Domain Shift Demonstration)")
print(f"    Train: sessions {TRAIN_SIDS}  |  Test: sessions {[REFERENCE_SID] + TEST_SIDS}")
print("=" * 72)

print("\n--- Training (same as Part 1) ---")
train_rows = []
for sid in TRAIN_SIDS:
    df = load_session(sid)
    baseline = float(np.median(df["rssi"].values))
    train_rows.extend(build_windows(df, baseline))
    print(f"  session{sid} (day {DAY_OF_SESSION[sid]}): "
          f"{len(df)} rows, mean RSSI={df['rssi'].mean():.1f}dBm")

X_tr = np.array([r["feats"] for r in train_rows])
y_tr = np.array([r["label"] for r in train_rows])
le   = LabelEncoder().fit(y_tr)
model = RandomForestClassifier(
    n_estimators=500, class_weight="balanced_subsample",
    random_state=42, n_jobs=-1).fit(X_tr, le.transform(y_tr))
print(f"  Trained RF on {len(X_tr)} windows, {len(FEATURE_NAMES)} features")
print(f"  Training classes: {list(le.classes_)}")

#evaluation is done on every test session
print("\n--- Evaluating across sessions ---")

all_results = []
detailed = {}

#for the reference of part 1
df_ref = load_session(REFERENCE_SID)
pred_ref = predict_rows(model, le, df_ref, float(np.median(df_ref["rssi"].values)))

#this restrict true labels to classes the model knows
mask_ref = np.isin(df_ref["label"].values, le.classes_)
acc_ref = accuracy_score(df_ref["label"].values[mask_ref], pred_ref[mask_ref])
all_results.append({"session": REFERENCE_SID, "day": DAY_OF_SESSION[REFERENCE_SID],
                    "kind": "SAME-day (reference)", "n_rows": mask_ref.sum(),
                    "accuracy": acc_ref})
detailed[REFERENCE_SID] = (df_ref["label"].values, pred_ref)

#finally cross-domain test sessions
for sid in TEST_SIDS:
    df = load_session(sid)
    pred = predict_rows(model, le, df, float(np.median(df["rssi"].values)))
    mask = np.isin(df["label"].values, le.classes_)
    acc = accuracy_score(df["label"].values[mask], pred[mask])
    all_results.append({"session": sid, "day": DAY_OF_SESSION[sid],
                        "kind": "CROSS-day", "n_rows": mask.sum(),
                        "accuracy": acc})
    detailed[sid] = (df["label"].values, pred)

#Summary table:
print("\n" + "=" * 72)
print(" RESULTS — same model, same training, only test session changed")
print("=" * 72)
print(f" {'Test session':<14}{'Recording day':<16}{'Condition':<25}{'Rows':>7}{'Accuracy':>11}")
print(" " + "-" * 71)
for r in all_results:
    marker = "  ← Part 1 reference" if r["kind"].startswith("SAME") else ""
    print(f"   session{r['session']}     {r['day']:<16}{r['kind']:<25}"
          f"{r['n_rows']:>7}{r['accuracy']*100:>10.2f}%{marker}")

cross_accs = [r["accuracy"] for r in all_results if r["kind"] == "CROSS-day"]
same_acc   = [r["accuracy"] for r in all_results if r["kind"].startswith("SAME")][0]

print(" " + "-" * 71)
print(f" Same-day reference accuracy (session 2):            {same_acc*100:6.2f}%")
print(f" Cross-day mean accuracy (sessions {TEST_SIDS}): {np.mean(cross_accs)*100:6.2f}%")
print(f" Accuracy drop under domain shift:                   {(same_acc - np.mean(cross_accs))*100:6.2f} pp")
print("=" * 72)

#per-class breakdown on the worst cross-day session (for confusion matrix slide)
worst_sid = min(TEST_SIDS, key=lambda s: [r["accuracy"] for r in all_results if r["session"]==s][0])
print(f"\n--- Confusion matrix for worst cross-day session (session{worst_sid}, "
      f"{DAY_OF_SESSION[worst_sid]}) ---")
y_true, y_pred = detailed[worst_sid]
mask = np.isin(y_true, le.classes_)
classes = sorted(set(y_true[mask]) | set(y_pred[mask]))
cm = confusion_matrix(y_true[mask], y_pred[mask], labels=classes)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
cm_df.index.name = "true \\ pred"
print(cm_df.to_string())

#Save
pd.DataFrame(all_results).to_csv(
    os.path.join(BASE_PATH, "part2_summary.csv"), index=False)

# Save per-session row-level predictions for detailed inspection
all_rows = []
for sid, (yt, yp) in detailed.items():
    df_s = load_session(sid)
    for ts, r, t, p in zip(df_s["timestamp"], df_s["rssi"], yt, yp):
        all_rows.append({"session": sid, "timestamp": ts, "rssi": r,
                         "true": t, "pred": p, "match": t == p})
pd.DataFrame(all_rows).to_csv(
    os.path.join(BASE_PATH, "part2_row_predictions.csv"), index=False)

joblib.dump(model, os.path.join(BASE_PATH, "part2_model.pkl"))
joblib.dump(le,    os.path.join(BASE_PATH, "part2_encoder.pkl"))
print(f"\nSaved: part2_summary.csv, part2_row_predictions.csv, part2_model.pkl, part2_encoder.pkl")

print("\n>>> Part 2 complete. <<<")
print(f"    Headline: Same model, same training — accuracy drops "
      f"{same_acc*100:.0f}% → {np.mean(cross_accs)*100:.0f}% when tested cross-day.")