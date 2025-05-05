import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate a synthetic binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the dataset (feature scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a simple neural network model
model = Sequential()

# Input layer with 20 features, hidden layer with 64 neurons and ReLU activation
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

# Output layer with 1 neuron and sigmoid activation (for binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("\nTraining the model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nModel evaluation on the test set:")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predict on the test data (showing first 5 predictions)
predictions = model.predict(X_test[:5])
predictions = (predictions > 0.5).astype(int)  # Convert predictions to binary (0 or 1)

print("\nPredictions for the first 5 test samples:")
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: Predicted Class = {pred[0]}, Actual Class = {y_test[i]}")

# Example: Printing final weights for understanding
weights = model.get_weights()
print("\nFinal weights of the model:")
for layer_num, layer_weights in enumerate(weights):
    print(f"Layer {layer_num + 1}: shape = {layer_weights.shape}")