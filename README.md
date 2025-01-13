# Hyperparameter Optimization for LSTM on Daily Minimum Temperatures Dataset

This repository demonstrates hyperparameter optimization for Long Short-Term Memory (LSTM) networks on the **Daily Minimum Temperatures Dataset**. The objective is to predict the next day's temperature using time-series data and evaluate the impact of various hyperparameters on the model's performance.

## Dataset

The dataset used is the **Daily Minimum Temperatures Dataset**, which contains daily minimum temperatures recorded in Melbourne, Australia. It spans from 1981 to 1990 and can be directly downloaded from [this link](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv).

## Features

- **Dynamic hyperparameter optimization** using combinations of:
  - Activation Functions: `tanh`, `relu`
  - Number of Neurons: `50`, `100`
  - Dropout Rates: `0.2`, `0.3`
  - Number of Layers: `1`, `2`
  - Optimizers: `adam`, `rmsprop`
  - Learning Rates: `0.001`, `0.01`
  - Epochs: `20`, `30`
- Results logged into CSV files:
  - `hyperparameter_results.csv`: Contains metrics for all hyperparameter combinations.
  - `best_hyperparameters.csv`: Contains the best-performing hyperparameter configuration.
- Visualization of training and validation losses for the best model.
- **Best model saved** as `best_lstm_model.h5` for future use.

## Execution Environment

This project was executed on **Google Colab** using the **v2-8 TPU**. The total execution time was approximately **5378.465 seconds**.

## Files

- **`hyperparameter_results.csv`**: Contains performance metrics (MSE, MAE) for all hyperparameter combinations.
- **`best_hyperparameters.csv`**: Contains the best-performing hyperparameter configuration based on MSE.
- **`best_lstm_model.h5`**: The trained LSTM model using the best hyperparameters.

## Results

The best-performing hyperparameters are as follows:

| Activation Function | Number of Neurons | Dropout Rate | Number of Layers | Optimizer | Learning Rate | Epochs | Test Loss | MAE    | MSE    |
|---------------------|-------------------|--------------|------------------|-----------|---------------|--------|-----------|--------|--------|
| `tanh`             | 100               | 0.2          | 2                | `rmsprop` | 0.01          | 20     | 0.007019  | 0.0659 | 0.0070 |

## Performance

- **Test Loss**: `0.007019`
- **Mean Absolute Error (MAE)**: `0.0659`
- **Mean Squared Error (MSE)**: `0.0070`

## Visualization

![Loss Curve](Hyperparameter%20Optimization%20for%20LSTM/loss_curve.png)  
The graph shows the training and validation loss for the best model over epochs.

## References

- Dataset: [Daily Minimum Temperatures Dataset](https://github.com/jbrownlee/Datasets)
- LSTM implementation and hyperparameter optimization techniques.