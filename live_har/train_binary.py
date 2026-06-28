#for making Walking and Not-Walking BINARY for easy classification

import os, glob, re, warnings
import numpy as np, pandas as pd, joblib
from scipy import stats, signal
from sklearn.preprocessing   import LabelEncoder
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics         import classification_report, accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

BASE_PATH   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_PATH, "..", "Data")
WINDOW_SIZE = 40
STEP_SIZE   = 10

#feature extraction:

def extract_features(x_raw):
    x_raw = x_raw.astype(float)
    mu, sigma = np.mean(x_raw), np.std(x_raw)
    x = (x_raw - mu) / sigma if sigma > 1e-6 else x_raw - mu
    diff1 = np.diff(x); n = len(x)
    td = [np.mean(x), np.std(x), np.min(x), np.max(x), np.max(x)-np.min(x),
          np.sqrt(np.mean(x**2)), np.median(x),
          float(pd.Series(x).skew()), float(pd.Series(x).kurtosis()),
          np.percentile(x,75)-np.percentile(x,25),
          np.sum(np.abs(diff1)), np.sum(x**2),
          int(np.sum(np.diff(np.sign(x-np.mean(x)))!=0)),
          np.max(diff1)-np.min(diff1), np.mean(np.abs(diff1)), np.std(diff1),
          np.percentile(x,90)-np.percentile(x,10),
          np.sum(x > np.mean(x)) / n]
    freqs = np.fft.rfftfreq(n); fft = np.abs(np.fft.rfft(x)); fft_sq = fft**2
    total = np.sum(fft_sq) + 1e-10
    dom_f = freqs[np.argmax(fft[1:])+1]
    s_mean = np.sum(freqs*fft_sq)/total
    s_std = np.sqrt(np.sum(((freqs-s_mean)**2)*fft_sq)/total)
    pnorm = fft_sq/total
    s_ent = -np.sum(pnorm*np.log2(pnorm+1e-10))
    low_p = np.sum(fft_sq[(freqs>=0.0)&(freqs<0.1)])/total
    mid_p = np.sum(fft_sq[(freqs>=0.1)&(freqs<0.3)])/total
    hig_p = np.sum(fft_sq[(freqs>=0.3)])/total
    cum = np.cumsum(fft_sq)
    r_idx = np.searchsorted(cum, 0.85*total)
    rolloff = freqs[min(r_idx, len(freqs)-1)]
    p_ratio = np.max(fft_sq)/(np.mean(fft_sq)+1e-10)
    sp = [dom_f, s_mean, s_std, s_ent, low_p, mid_p, hig_p, rolloff, p_ratio]
    acf_full = np.correlate(x-np.mean(x), x-np.mean(x), mode='full')
    acf = acf_full[n-1:]; acn = acf/(acf[0]+1e-10)
    ac1 = acn[1] if n>1 else 0.0; ac2 = acn[2] if n>2 else 0.0
    ac4 = acn[4] if n>4 else 0.0; ac8 = acn[8] if n>8 else 0.0
    peaks, _ = signal.find_peaks(acn[1:], height=0.1)
    period = float(peaks[0]+1) if len(peaks)>0 else 0.0
    return td + sp + [ac1, ac2, ac4, ac8, period]

FEATURE_NAMES = (
    ["rssi_mean","rssi_std","rssi_min","rssi_max","rssi_range",
     "rssi_rms","rssi_median","rssi_skew","rssi_kurtosis","rssi_iqr",
     "rssi_total_var","rssi_energy","rssi_zero_cross","rssi_pp_diff",
     "rssi_mad","rssi_std_diff","rssi_90_10","rssi_frac_above"] +
    ["fft_dom_freq","fft_spec_mean","fft_spec_std","fft_spec_entropy",
     "fft_low_pow","fft_mid_pow","fft_high_pow","fft_rolloff","fft_peak_ratio"] +
    ["ac1","ac2","ac4","ac8","ac_period"]
)
assert len(FEATURE_NAMES) == 32, "Feature name count must be 32"

def fix_ts(s):
    return pd.to_datetime(
        s.astype(str).str.replace(r"(\d{2}/\d{2})(\d{4})", r"\1/\2", regex=True),
        format="%d/%m/%Y %H:%M:%S.%f")

#loading all sessions
#labelling walking and not walking

print("=" * 68)
print("TRAIN BINARY - Walking vs Not-Walking")
print("=" * 68)
session_files = sorted(glob.glob(os.path.join(DATA_DIR, "session*.csv")))
if not session_files:
    raise FileNotFoundError(f"No session*.csv in {DATA_DIR}")

print(f"\nLoading {len(session_files)} session files from: {DATA_DIR}")
all_rows = []
for f in session_files:
    sid = int(re.search(r"session(\d+)", os.path.basename(f)).group(1))
    df = pd.read_csv(f, encoding="utf-8-sig"); df.columns = df.columns.str.strip()
    df.rename(columns={df.columns[0]:"timestamp", df.columns[1]:"rssi",
                        df.columns[2]:"label"}, inplace=True)
    df["timestamp"] = fix_ts(df["timestamp"])
    df["rssi"] = pd.to_numeric(df["rssi"], errors="coerce")
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["rssi","label"]).reset_index(drop=True)
    zs = np.abs(stats.zscore(df["rssi"])); df = df[zs < 3].reset_index(drop=True)
    df["session_id"] = sid
    # ── RELABEL ──
    df["binary_label"] = np.where(df["label"] == "walking", "walking", "not_walking")
    all_rows.append(df)

master = pd.concat(all_rows, ignore_index=True)
print(f"Rows after cleaning: {len(master)}")
print("\nBinary label distribution:")
print(master["binary_label"].value_counts().to_string())
print("\nOriginal class → binary mapping used:")
orig = master.groupby(["label","binary_label"]).size().reset_index(name="rows")
print(orig.to_string(index=False))

#window generation:

print("\n" + "=" * 68)
print("PHASE 2 - Feature extraction (per-session windowing, 32 features)")
print("=" * 68)

rows = []
for sid, grp in master.groupby("session_id"):
    rssi = grp["rssi"].values
    labels = grp["binary_label"].values
    for s in range(0, len(rssi)-WINDOW_SIZE, STEP_SIZE):
        w = rssi[s:s+WINDOW_SIZE]
        maj = pd.Series(labels[s:s+WINDOW_SIZE]).mode()[0]
        row = {"session_id": sid, "label": maj}
        row.update(dict(zip(FEATURE_NAMES, extract_features(w))))
        rows.append(row)

df_win = pd.DataFrame(rows)
print(f"Total windows: {len(df_win)}")
print("Window-level label distribution:")
print(df_win["label"].value_counts().to_string())

#LOSO performing:

print("\n" + "=" * 68)
print("PHASE 3 - Leave-One-Session-Out CV (honest cross-session evaluation)")
print("=" * 68)
loso = {}
for holdout in sorted(master["session_id"].unique()):
    tr = df_win[df_win["session_id"] != holdout]
    te = df_win[df_win["session_id"] == holdout]
    X_tr = tr[FEATURE_NAMES].values; y_tr = tr["label"].values
    X_te = te[FEATURE_NAMES].values; y_te = te["label"].values
    if len(X_te) == 0: continue
    le = LabelEncoder().fit(np.concatenate([y_tr, y_te]))
    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                  random_state=42, n_jobs=-1).fit(X_tr, le.transform(y_tr))
    pred = le.inverse_transform(clf.predict(X_te))
    loso[holdout] = accuracy_score(y_te, pred)
    print(f"  session{holdout}: {loso[holdout]*100:6.2f}%")
print(f"\n  MEAN LOSO ACCURACY: {np.mean(list(loso.values()))*100:.2f}%")


#training model finally

print("\n" + "=" * 68)
print("PHASE 4 - Training final deployment model on all 9 sessions")
print("=" * 68)
X = df_win[FEATURE_NAMES].values
y = df_win["label"].values
le = LabelEncoder().fit(y)
model = RandomForestClassifier(
    n_estimators=500, class_weight="balanced_subsample",
    random_state=42, n_jobs=-1).fit(X, le.transform(y))

# Quick resubstitution report making
pred_train = le.inverse_transform(model.predict(X))
print(f"Training accuracy (sanity check): {accuracy_score(y, pred_train)*100:.2f}%")
print(f"Classes: {list(le.classes_)}")

#Saving the  model artefacts in the exact format live_har.py expects
joblib.dump(model,         os.path.join(BASE_PATH, "model.pkl"))
joblib.dump(le,            os.path.join(BASE_PATH, "label_encoder.pkl"))
joblib.dump(FEATURE_NAMES, os.path.join(BASE_PATH, "feature_names.pkl"))
print(f"\nSaved in {BASE_PATH}:")
print("  model.pkl: binary RF classifier")
print("  label_encoder.pkl: maps walking/not_walking ↔ ints")
print("  feature_names.pkl: 32 feature names (for live_har.py sanity check)")

# Feature importance which is available in file feature_names.pkl
fi = pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
print("\nTop 8 most informative features for walking detection:")
print(fi.head(8).round(4).to_string())

print("\n" + "=" * 68)
print(f"DONE - Binary classifier ready. LOSO accuracy: "
      f"{np.mean(list(loso.values()))*100:.1f}%")
print("Now run:  python live_har.py --port COM5     (or --demo)")
print("=" * 68)
