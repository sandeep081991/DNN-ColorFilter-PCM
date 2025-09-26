import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib

# 1. Load data
data = pd.read_csv('transmission_dataset.csv')

# 2. Select inputs and outputs
#X = data[['SiO2_thickness', 'Sb2S3_thickness', 'Phase', 'Angle_deg']].values
X = data[['d_sio2 (nm)', 'd_sb2s3 (nm)', 'phase', 'angle (deg)']]

output_cols = [col for col in data.columns if col.startswith('T_')]
y = data[output_cols].values

# 3. Normalize outputs using MinMaxScaler
output_scaler = MinMaxScaler()
y_scaled = output_scaler.fit_transform(y)

# Save output scaler
joblib.dump(output_scaler, 'output_scaler.pkl')

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# 5. Scale inputs
input_scaler = StandardScaler()
X_train_scaled = input_scaler.fit_transform(X_train)
X_test_scaled = input_scaler.transform(X_test)

# Save input scaler
joblib.dump(input_scaler, 'input_scaler.pkl')

# 6. Build a deeper model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(y_train.shape[1], activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 7. Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 8. Train
history = model.fit(X_train_scaled, y_train,
                    validation_split=0.2,
                    epochs=200,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=1)

# 9. Evaluate
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Loss (MSE): {test_loss:.6f}")
print(f"Test MAE: {test_mae:.6f}")

# 10. Save model
model.save('transmission_predictor_model.h5')

# 11. Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
