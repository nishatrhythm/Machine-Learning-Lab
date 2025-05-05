import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the image data to [0, 1]
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Fit the data generator on the training data
datagen.fit(X_train)

# Define the CNN model
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Second Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Third Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Flatten the output of the convolutional layers to feed into the dense layer
model.add(Flatten())

# Fully connected layer with 256 neurons
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Dropout layer to reduce overfitting

# Output layer with 10 neurons for the 10 classes (softmax activation)
model.add(Dense(10, activation='softmax'))

# Compile the model with Adam optimizer and sparse categorical crossentropy loss function
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary to understand its architecture
model.summary()

# Train the model with data augmentation
print("\nTraining the model with data augmentation...")
history = model.fit(datagen.flow(X_train, y_train, batch_size=64), 
                    epochs=20, 
                    validation_data=(X_test, y_test), 
                    verbose=1)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_acc * 100:.2f}%")

# Save the training history plot to a file
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Save the plot to a file (e.g., 'cnn_training_history_optimized.png')
plt.savefig('cnn_training_history_optimized.png')
print("Training history plot saved as 'cnn_training_history_optimized.png'")

# Example prediction
print("\nPredicting on the first 5 test samples:")
predictions = model.predict(X_test[:5])
for i, pred in enumerate(predictions):
    predicted_class = np.argmax(pred)
    actual_class = np.argmax(y_test[i])
    print(f"Sample {i+1}: Predicted Class = {predicted_class}, Actual Class = {actual_class}")