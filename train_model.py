import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import pickle
import os

def extract_features(df):

    dwell = df['dwell_time'].dropna()
    flight = df['flight_time'].dropna()
    
    features = {
        'dwell_mean': dwell.mean(),
        'dwell_std': dwell.std(),
        'dwell_median': dwell.median(),
        'dwell_min': dwell.min(),
        'dwell_max': dwell.max(),
        'dwell_q25': dwell.quantile(0.25),
        'dwell_q75': dwell.quantile(0.75),
        'flight_mean': flight.mean(),
        'flight_std': flight.std(),
        'flight_median': flight.median(),
        'flight_min': flight.min(),
        'flight_max': flight.max(),
        'flight_q25': flight.quantile(0.25),
        'flight_q75': flight.quantile(0.75),
    }
    
    return features


csv_files = ['data/sample1.csv', 'data/sample2.csv', 'data/sample3.csv', 'data/sample4.csv', 'data/sample5.csv']
attempts = []
features_list = []

for i, csv_file in enumerate(csv_files):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.lower()
    
    attempts.append(df) 

    features = extract_features(df)
    features_list.append(features)


X = pd.DataFrame(features_list).values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



model = IsolationForest(
    contamination=0.2, 
    random_state=42,
    n_estimators=100,
    bootstrap=True
)

model.fit(X_scaled)

avg_length=sum([len(att) for att in attempts])/len(attempts)

model_data = {
    'model': model,
    'scaler': scaler,
    'password_length': int(avg_length),
    'num_training_attempts': len(attempts),
}
i=1
while(os.path.exists(f"models/model{i}.pkl")):
    i+=1
with open(f"models/model{i}.pkl", 'wb') as f:
    pickle.dump(model_data, f)



feature_df = pd.DataFrame(features_list)
stats_df = feature_df[['dwell_mean', 'dwell_std', 'flight_mean', 'flight_std']].describe()
print(stats_df.round(2))