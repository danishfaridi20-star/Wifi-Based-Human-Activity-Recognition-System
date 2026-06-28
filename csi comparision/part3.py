#CSI Comparison code

import os, re, glob, warnings
import numpy as np, pandas as pd, joblib
from scipy import signal, stats
import csiread
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_PATH, "csi_data")

#this will parse one dat file into CSI magnitude matrix (n_packets, 30, 3, 3)
def parse_csi_file(path):
    """Returns magnitude array or None if file is empty/corrupt."""
    csi = csiread.Intel(path, nrxnum=3, ntxnum=3, pl_size=10)
    csi.read()
    if csi.count == 0 or csi.csi is None:
        return None
    return np.abs(csi.csi)  # shape: (n_packets, 30, 3, 3) like this will be made

#Feature extracting
# Each file gives us a (T, 30, 3, 3) tensor where T ≈ 95 packets

def extract_csi_features(mag):
    """mag: (T, 30, 3, 3) CSI-magnitude tensor. Returns 1-D feature vector."""
    T = mag.shape[0]
    feats = []

    #1. Collapse to per-packet "total energy" time-series (like RSSI)
    pkt_energy = mag.mean(axis=(1,2,3))    # (T,)
    pkt_std    = mag.std(axis=(1,2,3))     # (T,)

    diff_e = np.diff(pkt_energy)
    feats += [
        float(np.mean(pkt_energy)), float(np.std(pkt_energy)),
        float(np.min(pkt_energy)),  float(np.max(pkt_energy)),
        float(np.max(pkt_energy)-np.min(pkt_energy)),
        float(np.median(pkt_energy)),
        float(pd.Series(pkt_energy).skew()),
        float(pd.Series(pkt_energy).kurtosis()),
        float(np.mean(np.abs(diff_e))),
        float(np.std(diff_e)),
        float(np.max(np.abs(diff_e))),
        float(np.sum(np.abs(diff_e))),
        int(np.sum(np.diff(np.sign(pkt_energy - np.mean(pkt_energy))) != 0)),
    ]

    # 2. Spectral features on the energy time-series
    freqs = np.fft.rfftfreq(T)
    fft   = np.abs(np.fft.rfft(pkt_energy - pkt_energy.mean()))
    fft_sq = fft ** 2
    total = np.sum(fft_sq) + 1e-10
    dom_f = float(freqs[np.argmax(fft[1:]) + 1]) if len(fft) > 1 else 0.0
    s_mean = float(np.sum(freqs * fft_sq) / total)
    s_std  = float(np.sqrt(np.sum(((freqs - s_mean)**2) * fft_sq) / total))
    pnorm  = fft_sq / total
    s_ent  = float(-np.sum(pnorm * np.log2(pnorm + 1e-10)))
    low_p  = float(np.sum(fft_sq[(freqs>=0.0)&(freqs<0.1)])/total)
    mid_p  = float(np.sum(fft_sq[(freqs>=0.1)&(freqs<0.3)])/total)
    hig_p  = float(np.sum(fft_sq[(freqs>=0.3)])/total)
    feats += [dom_f, s_mean, s_std, s_ent, low_p, mid_p, hig_p]

    # 3. Autocorrelation (periodicity — walking, jumping, clapping)
    ac_full = np.correlate(pkt_energy - pkt_energy.mean(),
                           pkt_energy - pkt_energy.mean(), mode="full")
    ac = ac_full[T-1:]
    acn = ac / (ac[0] + 1e-10)
    feats += [float(acn[k]) if T > k else 0.0 for k in (1, 2, 4, 8, 16)]
    peaks, _ = signal.find_peaks(acn[1:], height=0.1)
    feats.append(float(peaks[0] + 1) if len(peaks) > 0 else 0.0)

    #4. Per-subcarrier temporal variability (the core CSI signature)
    # For each of 30 subcarriers, collapse across antennas and compute
    # temporal std. Then summarise those 30 numbers.
    per_sc_std = mag.mean(axis=(2,3)).std(axis=0)   # (30,)
    per_sc_mean = mag.mean(axis=(2,3)).mean(axis=0)  # (30,)
    feats += [
        float(per_sc_std.mean()), float(per_sc_std.std()),
        float(per_sc_std.min()),  float(per_sc_std.max()),
        float(per_sc_mean.mean()), float(per_sc_mean.std()),
    ]

    #5. Per-antenna-pair mean magnitude (9 values)
    per_ap = mag.mean(axis=(0,1))   # (3, 3) — avg over time & subcarriers
    feats += [float(v) for v in per_ap.flatten()]

    #6. Subcarrier-frequency diversity: std across subcarriers per packet
    # then summarise
    sc_spread = mag.mean(axis=(2,3)).std(axis=1)  # (T,)
    feats += [float(sc_spread.mean()), float(sc_spread.std()),
              float(sc_spread.max())]

    return feats

FEATURE_NAMES = [
    # packet-energy time-domain
    "e_mean","e_std","e_min","e_max","e_range","e_median","e_skew","e_kurt",
    "e_mean_abs_diff","e_std_diff","e_max_abs_diff","e_total_var","e_zero_cross",
    # spectral
    "spec_dom","spec_mean","spec_std","spec_ent","spec_low","spec_mid","spec_high",
    # autocorrelation
    "ac1","ac2","ac4","ac8","ac16","ac_period",
    # per-subcarrier
    "sc_std_mean","sc_std_std","sc_std_min","sc_std_max","sc_mean_mean","sc_mean_std",
    # per antenna-pair (3x3 flattened)
    "ap11","ap12","ap13","ap21","ap22","ap23","ap31","ap32","ap33",
    # subcarrier spread
    "sc_spread_mean","sc_spread_std","sc_spread_max",
]
print(f"Feature count: {len(FEATURE_NAMES)}")

#loading all files followed by traing
print("=" * 72)
print("PART 3 — CSI-BASED ACTIVITY RECOGNITION (Intel 5300)")
print("=" * 72)
print("\n--- Phase 1: Parsing CSI files & extracting features ---")

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.dat")))
X_rows, y_rows, meta_rows = [], [], []
skipped = []
for f in files:
    name = os.path.basename(f)
    m = re.match(r"\d+_\d+_(\w+)_\d+\.dat", name)
    if not m:
        skipped.append((name, "unparseable name")); continue
    label = m.group(1)
    mag = parse_csi_file(f)
    if mag is None or mag.shape[0] < 20:
        skipped.append((name, "empty or too few packets")); continue
    feats = extract_csi_features(mag)
    X_rows.append(feats); y_rows.append(label); meta_rows.append(name)

X = np.array(X_rows); y = np.array(y_rows)
print(f"  Loaded {len(X)} files across {len(set(y))} classes")
print(f"  Feature matrix: {X.shape}")
print(f"  Skipped files: {len(skipped)}")
for n, why in skipped: print(f"    - {n}: {why}")

print("\n  Class distribution:")
print(pd.Series(y).value_counts().to_string())

#5 fold stratified cross-validation of the CSI data

print("\n" + "=" * 72)
print("PHASE 2 — 5-fold stratified cross-validation")
print("=" * 72)

le = LabelEncoder().fit(y)
y_enc = le.transform(y)

clf = RandomForestClassifier(
    n_estimators=400, class_weight="balanced_subsample",
    random_state=42, n_jobs=-1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accs, cv_preds = [], np.empty(len(y), dtype=object)
for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y_enc), 1):
    clf.fit(X[tr_idx], y_enc[tr_idx])
    pred = le.inverse_transform(clf.predict(X[te_idx]))
    acc = accuracy_score(y[te_idx], pred)
    cv_accs.append(acc)
    for i, p in zip(te_idx, pred): cv_preds[i] = p
    print(f"  Fold {fold}: {acc*100:.2f}%  ({len(te_idx)} test files)")
print(f"\n  MEAN 5-fold CV ACCURACY: {np.mean(cv_accs)*100:.2f}% "
      f"± {np.std(cv_accs)*100:.2f}%")

#hold test is performed
#final model is generated

print("\n" + "=" * 72)
print("PHASE 3 — Hold-out test (20% of files, stratified)")
print("=" * 72)

X_tr, X_te, y_tr, y_te, meta_tr, meta_te = train_test_split(
    X, y_enc, meta_rows, test_size=0.2, stratify=y_enc, random_state=42)

final_clf = RandomForestClassifier(
    n_estimators=500, class_weight="balanced_subsample",
    random_state=42, n_jobs=-1).fit(X_tr, y_tr)
pred = le.inverse_transform(final_clf.predict(X_te))
y_te_lbl = le.inverse_transform(y_te)
acc = accuracy_score(y_te_lbl, pred)

print(f"\n  HOLD-OUT TEST ACCURACY: {acc*100:.2f}%  ({len(y_te)} files)")
print("\nPer-class report:")
print(classification_report(y_te_lbl, pred, zero_division=0))

classes = sorted(set(y))
cm = confusion_matrix(y_te_lbl, pred, labels=classes)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
cm_df.index.name = "true \\ pred"
print("Confusion matrix:")
print(cm_df.to_string())

#finally comparison of CSI with RSSI

print("\n" + "=" * 72)
print("PHASE 4 — CSI vs RSSI summary (for presentation)")
print("=" * 72)
print("  ┌─────────────────────────────────────┬────────────┐")
print("  │ Sensing modality & evaluation       │ Accuracy   │")
print("  ├─────────────────────────────────────┼────────────┤")
print("  │ RSSI (Part 1) same-domain, 4-class  │  79.97%    │")
print("  │ RSSI (Part 2) cross-domain, 5-class │  33.46%    │")
print(f"  │ CSI  (Part 3) 5-fold CV, 5-class    │  {np.mean(cv_accs)*100:5.2f}%    │")
print(f"  │ CSI  (Part 3) hold-out, 5-class     │  {acc*100:5.2f}%    │")
print("  └─────────────────────────────────────┴────────────┘")
print("\n  Information per packet:  RSSI = 1 scalar;  CSI = 270 complex values")
print("  Hardware cost:          ESP32 ~₹6k;       Intel 5300 ~₹12k+")


#saving predictions.csv file and model
out = pd.DataFrame({"file": meta_rows, "true": y, "pred_cv": cv_preds})
out["match"] = out["true"] == out["pred_cv"]
out.to_csv(os.path.join(BASE_PATH, "part3_cv_predictions.csv"), index=False)

pd.DataFrame({"file": meta_te, "true": y_te_lbl, "pred": pred,
              "match": y_te_lbl == pred}).to_csv(
    os.path.join(BASE_PATH, "part3_holdout_predictions.csv"), index=False)

joblib.dump(final_clf,      os.path.join(BASE_PATH, "part3_model.pkl"))
joblib.dump(le,             os.path.join(BASE_PATH, "part3_encoder.pkl"))
joblib.dump(FEATURE_NAMES,  os.path.join(BASE_PATH, "part3_feature_names.pkl"))
print("\nSaved: part3_cv_predictions.csv, part3_holdout_predictions.csv,")
print("       part3_model.pkl, part3_encoder.pkl, part3_feature_names.pkl")
print(f"\n>>> Part 3 complete. Headline: CSI hits {acc*100:.1f}% vs RSSI's 80%. <<<")