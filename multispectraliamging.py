import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# -----------------------------
# Load DNN and scalers
# -----------------------------
model = load_model("dnn_surrogate_model.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_Y = joblib.load("scaler_Y.pkl")

print("Loaded DNN surrogate and scalers.")

# -----------------------------
# Create a synthetic scene
# -----------------------------
height, width, n_channels = 64, 64, 150   # must match model output (150)
scene = np.zeros((height, width, n_channels))

# Example baseline spectrum
baseline_spectrum = np.linspace(0.1, 1.0, n_channels)
scene[:] = baseline_spectrum

print(f"Scene created: {scene.shape}")

# -----------------------------
# Apply LVCF transmission (simulated with DNN)
# -----------------------------
positions = np.linspace(0.1, 1.0, width)  # thickness positions
T_map = np.zeros((height, n_channels))    # transmission map

for p_idx, thickness in enumerate(positions):
    X_input = np.array([[thickness]])  # input thickness
    X_scaled = scaler_X.transform(X_input)
    pred = model.predict(X_scaled)     # shape (1, 150)
    pred = scaler_Y.inverse_transform(pred).flatten()

    # assign prediction to column p_idx
    T_map[:, :] = pred   # same spectrum across all rows

print("Transmission map generated.")

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(8, 6))
plt.imshow(T_map, aspect="auto", cmap="viridis", 
           extent=[positions.min(), positions.max(), 400, 1100])
plt.colorbar(label="Transmission")
plt.xlabel("Normalized Thickness Position")
plt.ylabel("Wavelength (nm)")
plt.title("Simulated Multispectral Imaging with PCM-LVCF")
plt.show()
