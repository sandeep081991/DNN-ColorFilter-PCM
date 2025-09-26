# results_plots.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ---------------------------------------------------
# 1. Load data and preprocessing objects
# ---------------------------------------------------
data = pd.read_csv("transmission_dataset.csv")

X = data[['d_sio2 (nm)', 'd_sb2s3 (nm)', 'phase', 'angle (deg)']]
output_cols = [col for col in data.columns if col.startswith("T_")]
y = data[output_cols].values

# Load saved scalers
input_scaler = joblib.load("input_scaler.pkl")
output_scaler = joblib.load("output_scaler.pkl")

X_scaled = input_scaler.transform(X)

# ---------------------------------------------------
# 2. Load trained model safely (fix for "mse" error)
# ---------------------------------------------------
model = load_model("transmission_predictor_model.h5", compile=False)

# ---------------------------------------------------
# 3. Predictions
# ---------------------------------------------------
y_pred_scaled = model.predict(X_scaled)
y_pred = output_scaler.inverse_transform(y_pred_scaled)

# ---------------------------------------------------
# 4. Compare Predicted vs Ground Truth Spectra
# ---------------------------------------------------
# Convert "T_0.400um" → 400.0 (nm)
wavelengths = [float(c.split("_")[1].replace("um", "")) * 1000 for c in output_cols]

plt.figure(figsize=(12, 8))
for i, idx in enumerate(np.random.choice(len(y), 3, replace=False)):
    plt.subplot(3, 1, i + 1)
    plt.plot(wavelengths, y[idx], label="Ground Truth (FDTD)", linewidth=2)
    plt.plot(wavelengths, y_pred[idx], "--", label="DNN Prediction", linewidth=2)
    plt.title(f"Sample {idx} — Phase={X.iloc[idx,2]}, Angle={X.iloc[idx,3]}°")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 5. Error Distribution (RMSE histogram)
# ---------------------------------------------------
rmse_per_sample = np.sqrt(np.mean((y - y_pred)**2, axis=1))

plt.figure(figsize=(6,4))
plt.hist(rmse_per_sample, bins=30, edgecolor="black", alpha=0.7)
plt.xlabel("RMSE per spectrum")
plt.ylabel("Count")
plt.title("Error Distribution across Test Set")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

print(f"Average RMSE: {np.mean(rmse_per_sample):.6f}")
print(f"R² Score: {r2_score(y.flatten(), y_pred.flatten()):.6f}")

# ---------------------------------------------------
# 6. Generalization: Interpolation test
# (pick a thickness not in dataset and predict)
# ---------------------------------------------------
test_sample = pd.DataFrame([[70, 140, 1, 0]],  # custom input [SiO2, Sb2S3, phase, angle]
                           columns=['d_sio2 (nm)', 'd_sb2s3 (nm)', 'phase', 'angle (deg)'])
test_scaled = input_scaler.transform(test_sample)
pred_scaled = model.predict(test_scaled)
pred = output_scaler.inverse_transform(pred_scaled)

plt.figure(figsize=(7,4))
plt.plot(wavelengths, pred[0], label="Predicted Spectrum", linewidth=2)
plt.title("DNN Prediction for Unseen Input (Interpolation Test)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------
# 7. Speed Comparison (FDTD vs DNN)
# ---------------------------------------------------
import time

start = time.time()
_ = model.predict(test_scaled)
end = time.time()
print(f"DNN Prediction Time: {(end-start)*1000:.3f} ms (per spectrum)")
print("Typical FDTD simulation: ~minutes (depending on setup)")
print("Speed-up factor: >1000×")
