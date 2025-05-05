# SUnivariate MLP (Single Input, Single Output)

from numpy import array
from keras.models import Sequential
from keras.layers import Dense

def split_sequence(seq, n_steps):
    X, y = [], []
    for i in range(len(seq) - n_steps):
        X.append(seq[i:i+n_steps])
        y.append(seq[i+n_steps])
    return array(X), array(y)

# Dataset
seq = [10, 20, 30, 40, 50]
n_steps = 3
X, y = split_sequence(seq, n_steps)

print("Input-Output pairs:")
for i in range(len(X)):
    print(f"X: {X[i]} -> y: {y[i]}")

# Model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=500, verbose=0)

# Prediction
x_input = array([30, 40, 50]).reshape(1, n_steps)
yhat = model.predict(x_input, verbose=0)
print(f"\nGiven input: {x_input.flatten()}")
print(f"Predicted next value: {yhat[0][0]:.2f}")