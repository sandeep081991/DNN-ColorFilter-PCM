import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
import tensorflow.keras.losses

# Load the trained model and scalers
model = load_model('transmission_predictor_model.h5', custom_objects={'mse': tensorflow.keras.losses.mse})

# Load input scaler and output scaler (full MinMaxScaler object)
input_scaler = joblib.load('input_scaler.pkl')
output_scaler = joblib.load('output_scaler.pkl')

# Define input parameters: [SiO2_thickness, Sb2S3_thickness, Phase, Angle_deg]
input_params = np.array([[70,200, 0, 0]])

# Scale the input features using the input scaler
input_scaled = input_scaler.transform(input_params)

# Predict the transmission spectrum using the model
predicted_transmission_scaled = model.predict(input_scaled)

# Rescale the predicted output back to the original scale using inverse_transform
predicted_transmission = output_scaler.inverse_transform(predicted_transmission_scaled)

# Plot the predicted transmission spectrum
plt.plot(predicted_transmission.flatten())
plt.xlabel('Wavelength or Index')
plt.ylabel('Transmission')
plt.title('Predicted Transmission Spectrum\n(Thickness1=70nm, Thickness2=150nm, Phase=0, Angle=5)')
plt.grid(True)
plt.show()
