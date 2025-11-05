import os
import joblib
import numpy as np
import pandas as pd


MODEL_PATH = "Model/my_user.pkl"   # path to saved model file
PROBE_FILE = "data/sample6.csv"   # one typing attempt CSV (same phrase)



def build_position_feature(dwell, flight, max_pos):
    d = np.array(dwell) if dwell is not None else np.array([])
    f = np.array(flight) if flight is not None else np.array([])

    d_p = np.full(max_pos, -1.0, dtype=float)
    f_p = np.full(max_pos, -1.0, dtype=float)
    if d.size > 0:
        d_p[: min(max_pos, d.size)] = d[:max_pos]
    if f.size > 0:
        f_p[: min(max_pos, f.size)] = f[:max_pos]

    mean_d = float(np.mean(d)) if d.size > 0 else -1.0
    std_d = float(np.std(d)) if d.size > 0 else 0.0
    mean_f = float(np.mean(f)) if f.size > 0 else -1.0
    std_f = float(np.std(f)) if f.size > 0 else 0.0
    n_keys = float(d.size)
    meta = np.array([mean_d, std_d, mean_f, std_f, n_keys], dtype=float)
    return np.concatenate([d_p, f_p, meta])


def sample_to_feature(path, max_pos):
    df = pd.read_csv(path)
    if 'dwell_time' in df.columns:
        dwell = df['dwell_time'].to_numpy()
    elif {'press_time', 'release_time'}.issubset(df.columns):
        dwell = (df['release_time'] - df['press_time']).to_numpy()
    else:
        dwell = np.array([])

    if 'flight_time' in df.columns:
        flight = df['flight_time'].to_numpy()
        if flight.size>0 and np.isnan(flight[-1]):
            flight = flight[~np.isnan(flight)]
    elif {'press_time', 'release_time'}.issubset(df.columns):
        rel = df['release_time'].to_numpy()
        press = df['press_time'].to_numpy()
        flight = press[1:] - rel[:-1] if rel.size > 1 else np.array([])
    else:
        flight = np.array([])

    if dwell.size>0:
        dwell = dwell[~np.isnan(dwell)]
    if flight.size>0:
        flight = flight[~np.isnan(flight)]

    return build_position_feature(dwell, flight, max_pos)


# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model_obj = joblib.load(MODEL_PATH)
scaler = model_obj['scaler']
svm = model_obj['model']
threshold = model_obj['threshold']
meta = model_obj['meta']
max_pos = int(meta.get('max_pos', 30))

# Build probe feature
feat = sample_to_feature(PROBE_FILE, max_pos=max_pos)
feat_scaled = scaler.transform(feat.reshape(1, -1))
score = float(svm.decision_function(feat_scaled)[0])

# Map score -> confidence: normalize between threshold and 0 (0 -> worst)
# simple linear mapping: confidence = clip((score - min_score)/(max_score - min_score))
# choose max_score = max(training_scores, 0) is unknown here; use 0 as "very good"
min_score = threshold * 2.0   # heuristic worst score
max_score = 0.0
conf = np.clip((score - min_score) / (max_score - min_score), 0.0, 1.0) * 100.0

print(f"Score: {score:.6f}, Threshold: {threshold:.6f}, Confidence: {conf:.2f}%")
print("Result ->", "ACCEPT (same user)" if score >= threshold else "REJECT (different)")

# Show summary feature differences vs training mean (if available)
if 'scaler' in model_obj:
    # we don't save training mean in meta, but scaler has mean_ in original space
    train_mean = model_obj['scaler'].mean_
    raw_feat = feat  # before scaling
    diffs = raw_feat - train_mean
    # print a short summary: mean dwell diff, mean flight diff, n_keys diff (meta tail)
    tail_idx = max_pos*2
    mean_d_diff = diffs[tail_idx + 0]
    std_d_diff = diffs[tail_idx + 1]
    mean_f_diff = diffs[tail_idx + 2]
    std_f_diff = diffs[tail_idx + 3]
    nkeys_diff = diffs[tail_idx + 4]
    print("\nFeature differences (raw - train_mean):")
    print(f" mean_dwell  diff: {mean_d_diff:.3f}")
    print(f" std_dwell   diff: {std_d_diff:.3f}")
    print(f" mean_flight diff: {mean_f_diff:.3f}")
    print(f" std_flight  diff: {std_f_diff:.3f}")
    print(f" n_keys      diff: {nkeys_diff:.3f}")
