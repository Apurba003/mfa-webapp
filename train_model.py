import os
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# ========== CONFIG ==========
MODEL_KEY = "my_user"   # change to your user id
SAMPLE_FILES = [
    "data/sample1.csv",
    "data/sample2.csv",
    "data/sample3.csv",
    "data/sample4.csv",
    "data/sample5.csv",
    # add more samples...
]
MODEL_DIR = "Model"
MAX_POS = 30            # number of dwell & flight positions to keep (pad/truncate)
NU = 0.9                # One-Class SVM nu parameter
THRESHOLD_PERCENTILE = 50.0   # percentile on training scores used as acceptance threshold
# ===========================

os.makedirs(MODEL_DIR, exist_ok=True)


def build_position_feature(dwell, flight, max_pos=30):
    """Return fixed-length position-wise feature vector from dwell & flight arrays."""
    # dwell -> length max_pos (pad with -1)
    d = np.array(dwell) if dwell is not None else np.array([])
    f = np.array(flight) if flight is not None else np.array([])

    d_p = np.full(max_pos, -1.0, dtype=float)
    f_p = np.full(max_pos, -1.0, dtype=float)

    if d.size > 0:
        d_p[: min(max_pos, d.size)] = d[:max_pos]
    if f.size > 0:
        # flight usually length = n_keys - 1; store in first positions
        f_p[: min(max_pos, f.size)] = f[:max_pos]

    # also append summary stats for robustness
    mean_d = float(np.mean(d)) if d.size > 0 else -1.0
    std_d = float(np.std(d)) if d.size > 0 else 0.0
    mean_f = float(np.mean(f)) if f.size > 0 else -1.0
    std_f = float(np.std(f)) if f.size > 0 else 0.0
    n_keys = float(d.size)
    meta = np.array([mean_d, std_d, mean_f, std_f, n_keys], dtype=float)

    return np.concatenate([d_p, f_p, meta])


def sample_to_feature(path, max_pos=30):
    """Read single-sample CSV (event rows) and return position-wise feature vector."""
    df = pd.read_csv(path)
    # prefer dwell_time and flight_time columns if present
    if 'dwell_time' in df.columns:
        dwell = df['dwell_time'].to_numpy()
    elif {'press_time', 'release_time'}.issubset(df.columns):
        dwell = (df['release_time'] - df['press_time']).to_numpy()
    else:
        dwell = np.array([])

    if 'flight_time' in df.columns:
        flight = df['flight_time'].to_numpy()
        # drop trailing NaN if present
        if flight.size > 0 and np.isnan(flight[-1]):
            flight = flight[~np.isnan(flight)]
    elif {'press_time', 'release_time'}.issubset(df.columns):
        rel = df['release_time'].to_numpy()
        press = df['press_time'].to_numpy()
        flight = press[1:] - rel[:-1] if rel.size > 1 else np.array([])
    else:
        flight = np.array([])

    # remove NaNs if any
    if dwell.size > 0:
        dwell = dwell[~np.isnan(dwell)]
    if flight.size > 0:
        flight = flight[~np.isnan(flight)]

    return build_position_feature(dwell, flight, max_pos=max_pos)


# Collect features
features = []
for p in SAMPLE_FILES:
    if not os.path.exists(p):
        print(f"Warning: file not found, skipping: {p}")
        continue
    try:
        feat = sample_to_feature(p, max_pos=MAX_POS)
        features.append(feat)
        print(f"Loaded {p} -> feature_len={len(feat)}")
    except Exception as e:
        print(f"Error reading {p}: {e}")

if len(features) < 5:
    raise ValueError(f"Need >=5 samples for stable per-user model; found {len(features)}")

X = np.vstack(features)
print(f"Training on {X.shape[0]} samples, feature dim = {X.shape[1]}")

# scale, train
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)

svm = OneClassSVM(kernel='rbf', gamma='scale', nu=NU).fit(Xs)

scores = svm.decision_function(Xs)
threshold = float(np.percentile(scores, THRESHOLD_PERCENTILE))

model_obj = {
    'scaler': scaler,
    'model': svm,
    'threshold': threshold,
    'meta': {
        'user': MODEL_KEY,
        'n_samples': int(X.shape[0]),
        'max_pos': int(MAX_POS),
        'nu': float(NU),
        'threshold_percentile': float(THRESHOLD_PERCENTILE),
        'created_at': datetime.utcnow().isoformat() + "Z"
    }
}

out_path = os.path.join(MODEL_DIR, f"{MODEL_KEY}.pkl")
joblib.dump(model_obj, out_path)
print(f"\nModel saved -> {out_path}")
print(f"Samples used: {X.shape[0]}, Threshold: {threshold:.6f}")
