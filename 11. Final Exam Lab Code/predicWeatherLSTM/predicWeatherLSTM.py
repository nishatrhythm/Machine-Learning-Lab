import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import os

np.random.seed(42)
random.seed(42)

print("\n" + "="*60)
print("LOADING DATA".center(60))
print("="*60)

data = pd.read_csv('seattle-weather.csv')
print(f"Dataset loaded: seattle-weather.csv")

data = data.drop(['date', 'weather'], axis=1, errors='ignore')

data = data.fillna(data.mean())

features = ['precipitation', 'temp_min', 'wind']
target = 'temp_max'

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features + [target]])
print(f"Features normalized")

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :-1])
        y.append(data[i + seq_length, -1])
    return np.array(X), np.array(y)

seq_length = 10  
X, y = create_sequences(scaled_data, seq_length)
print(f"Sequence shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

print("\n" + "="*60)
print("BUILDING AND TRAINING LSTM MODEL".center(60))
print("="*60)

def build_model(units, learning_rate, dropout_rate):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(seq_length, X.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units // 2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def random_search(X_train, y_train, X_val, y_val, n_iter=10):
    param_grid = {
        'units': [32, 64, 128],
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [16, 32, 64],
        'dropout_rate': [0.2, 0.3, 0.4],
        'epochs': [50, 100]
    }
    
    best_score = float('inf')
    best_params = None
    best_model = None
    all_results = []
    
    for i in range(n_iter):
        params = {
            'units': random.choice(param_grid['units']),
            'learning_rate': random.choice(param_grid['learning_rate']),
            'batch_size': random.choice(param_grid['batch_size']),
            'dropout_rate': random.choice(param_grid['dropout_rate']),
            'epochs': random.choice(param_grid['epochs'])
        }
        
        print(f"\n=== Starting Random Search Iteration {i+1}/{n_iter} ===")
        print(f"Testing parameters: {params}")
        
        model = build_model(
            units=params['units'],
            learning_rate=params['learning_rate'],
            dropout_rate=params['dropout_rate']
        )
        
        print("Training model...")
        history = model.fit(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        val_loss = min(history.history['val_loss'])
        print(f"Iteration {i+1} completed. Validation MSE: {val_loss:.6f}")
        
        result = params.copy()
        result['val_loss'] = val_loss
        all_results.append(result)
        
        if val_loss < best_score:
            best_score = val_loss
            best_params = params
            best_model = model
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('hyperparameters_grid.csv', index=False)
    print("- Saved all hyperparameter combinations to 'hyperparameters_grid.csv'")
    
    best_params_df = pd.DataFrame([best_params])
    best_params_df.to_csv('best_hyperparameters.csv', index=False)
    print("- Saved best hyperparameters to 'best_hyperparameters.csv'")
    
    print("\n=== Random Search Completed ===")
    return best_model, best_params, best_score

X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

best_model, best_params, best_score = random_search(X_train_sub, y_train_sub, X_val, y_val, n_iter=10)

print("\n" + "="*60)
print("EVALUATING MODEL".center(60))
print("="*60)

print("\nBest parameters found:", best_params)
print("Best validation score (MSE):", best_score)

print("\nEvaluating best model on test set...")
y_pred = best_model.predict(X_test, verbose=0)

y_test_reshaped = y_test.reshape(-1, 1)
y_pred_reshaped = y_pred.reshape(-1, 1)

y_test_dummy = np.hstack([np.zeros((y_test_reshaped.shape[0], len(features))), y_test_reshaped])
y_pred_dummy = np.hstack([np.zeros((y_pred_reshaped.shape[0], len(features))), y_pred_reshaped])

y_test_inv = scaler.inverse_transform(y_test_dummy)[:, -1]
y_pred_inv = scaler.inverse_transform(y_pred_dummy)[:, -1]

mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)

print("Test MSE:", mse)
print("Test MAE:", mae)

print("\n" + "="*60)
print("GENERATING PLOTS".center(60))
print("="*60)

plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title('Actual vs Predicted Temperature')
plt.xlabel('Sample')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.savefig('temp_prediction.png')
print("- Saved temperature prediction plot to 'temp_prediction.png'")

best_model_history = best_model.history.history if hasattr(best_model, 'history') else None
if best_model_history and 'loss' in best_model_history:
    plt.figure(figsize=(10, 6))
    plt.plot(best_model_history['loss'], label='Training Loss')
    plt.plot(best_model_history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png')
    print("- Saved loss curve to 'loss_curve.png'")

best_model.save('lstm_weather_model.h5')
print("- Model saved as 'lstm_weather_model.h5'")

print("\n" + "="*60)
print("DONE".center(60))
print("="*60)