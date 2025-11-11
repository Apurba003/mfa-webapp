import pandas as pd
import numpy as np
import pickle

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



with open('Model/password_keystroke_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
expected_length = model_data['password_length']

print("Model loaded successfully")




df = pd.read_csv('data/sample6.csv')
df.columns = df.columns.str.strip().str.lower()

print(f"Loaded {len(df)} keystrokes for verification")


if abs(len(df) - expected_length) > 2: 
    print(f"WARNING: Expected {expected_length} keystrokes, got {len(df)}")
    print("This might be a different password!")


features = extract_features(df)
X = pd.DataFrame([features]).values


X_scaled = scaler.transform(X)


prediction = model.predict(X_scaled)[0]
score = model.decision_function(X_scaled)[0]


is_authentic = (prediction == 1)
confidence = round((score + 0.5) * 100, 2)
confidence = max(0, min(100, confidence)) 




if is_authentic:
    print("AUTHENTICATION SUCCESSFUL")
    print("Status: VERIFIED ")
    print(f"Confidence: {confidence}%")
    print(f"Decision Score: {score:.4f}")
    print("\n➜ This typing pattern matches the enrolled user!")
else:
    print("AUTHENTICATION FAILED")
    print("Status: REJECTED ")
    print(f"Confidence: {confidence}%")
    print(f"Decision Score: {score:.4f}")





print("\nVerification Sample Statistics:")
print(f"  Avg Dwell Time: {features['dwell_mean']:.2f} ms")
print(f"  Avg Flight Time: {features['flight_mean']:.2f} ms")


print("\nSecurity Level:")
if is_authentic and confidence > 70:
    print("   HIGH - Strong match")
elif is_authentic and confidence > 50:
    print("   MEDIUM - Acceptable match")
elif is_authentic:
    print("   LOW - Weak match, consider re-enrollment")
else:
    print("   REJECTED - No match")
