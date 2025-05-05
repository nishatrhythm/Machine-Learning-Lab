# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Load Dataset
data = pd.read_csv('Electric_Production.csv')
values = data['IPG2211A2N'].values

# Preprocess Data
scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values.reshape(-1, 1))

# Sliding Window for Time Series
window_size = 30
X = np.array([values_scaled[i:i+window_size] for i in range(len(values_scaled) - window_size)])
y = np.array([values_scaled[i+window_size] for i in range(len(values_scaled) - window_size)])
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split Dataset into Train (70%), Validation (15%), and Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define CNN Model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(window_size, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1)  # Output layer for regression
])

# Compile the Model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_val, y_val), verbose=1)

# Evaluate on Test Set
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Plot Train vs Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Train vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predict on Test Set
y_pred = model.predict(X_test)

# Inverse transform the predictions and actual values
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# Plot Actual vs Predicted Values
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label='Actual', linewidth=2)
plt.plot(y_pred_rescaled, label='Predicted', linewidth=2)
plt.title('Actual vs Predicted Values')
plt.xlabel('Time Steps')
plt.ylabel('Electric Production')
plt.legend()
plt.show()

# Residual Plot
residuals = y_test_rescaled.flatten() - y_pred_rescaled.flatten()
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals', color='red')
plt.axhline(0, color='black', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Time Steps')
plt.ylabel('Residuals')
plt.legend()
plt.show()