import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from itertools import product
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import csv
import os
from datetime import datetime
import logging

# Suppress TensorFlow warnings for cleaner output
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Step 1: Load and Preprocess the Data
dataset_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
data = pd.read_csv(dataset_url)

def load_and_preprocess_data(data, sequence_length=30, test_size=0.2):
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    if data.isnull().values.any():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset contains missing values. Handling missing values...")
        data = data.dropna()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Temp']])
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Total samples: {X.shape[0]}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training samples: {X_train.shape[0]}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Testing samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, scaler

sequence_length = 30
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data, sequence_length, test_size=0.2)

# Step 2: Define the LSTM Model Builder Function
def build_model(num_layers, num_neurons, dropout_rate, input_shape):
    model = Sequential()
    
    for i in range(num_layers):
        return_sequences = True if i < num_layers - 1 else False
        if i == 0:
            model.add(LSTM(units=num_neurons, activation='tanh', return_sequences=return_sequences, input_shape=input_shape))
        else:
            model.add(LSTM(units=num_neurons, activation='tanh', return_sequences=return_sequences))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1))
    
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mse'])
    
    return model

# Step 3: Define a Smaller Hyperparameter Grid
def get_hyperparameter_grid():
    num_neurons_options = [50, 100]
    dropout_rates = [0.2, 0.3]
    num_layers_options = [1, 2]
    epochs_options = [20, 30]
    
    hyperparameter_combinations = list(product(
        num_neurons_options,
        dropout_rates,
        num_layers_options,
        epochs_options
    ))
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Total hyperparameter combinations: {len(hyperparameter_combinations)}")
    return hyperparameter_combinations

hyperparameter_combinations = get_hyperparameter_grid()

# Step 4: Perform Hyperparameter Optimization
def hyperparameter_optimization(X_train, X_test, y_train, y_test, hyperparameter_combinations, csv_file):
    header = ['Number of Neurons', 'Dropout Rate', 'Number of Layers', 'Epochs', 'Test Loss', 'MAE', 'MSE']
    
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    for idx, (neurons, dropout, layers, epochs) in enumerate(hyperparameter_combinations):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting training for model {idx+1}/{len(hyperparameter_combinations)}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Configuration: Neurons={neurons}, Dropout={dropout}, Layers={layers}, Epochs={epochs}")
        
        model = build_model(layers, neurons, dropout, input_shape)
        
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training model...")
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0
        )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Evaluating model...")
        evaluation = model.evaluate(X_test, y_test, verbose=0)
        test_loss, mae, mse = evaluation
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Results: Test Loss={test_loss:.4f}, MAE={mae:.4f}, MSE={mse:.4f}")
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([neurons, dropout, layers, epochs, test_loss, mae, mse])

results_csv = 'hyperparameter_results.csv'
hyperparameter_optimization(X_train, X_test, y_train, y_test, hyperparameter_combinations, results_csv)

# Step 5: Identify and Save the Best Hyperparameters
def find_best_hyperparameters(csv_file, metric='MSE'):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Analyzing results to find best hyperparameters...")
    results = pd.read_csv(csv_file)
    
    if metric not in results.columns:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Metric '{metric}' not found in the results.")
        return
    
    best_results = results.sort_values(by=metric, ascending=True).reset_index(drop=True)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Top 5 configurations based on {metric}:")
    print(best_results.head(5))
    
    best_configuration = best_results.iloc[0]
    best_configuration.to_frame().T.to_csv('best_hyperparameters.csv', index=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Best hyperparameter configuration saved to 'best_hyperparameters.csv'.")

find_best_hyperparameters(results_csv, metric='MSE')