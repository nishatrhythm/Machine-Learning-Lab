# Multi-step MLP (Single Input, Multiple Outputs)

from numpy import array
from keras.models import Sequential
from keras.layers import Dense

def split_sequence(seq, n_in, n_out):
    X, y = [], []
    for i in range(len(seq) - n_in - n_out + 1):
        X.append(seq[i:i+n_in])
        y.append(seq[i+n_in:i+n_in+n_out])
    return array(X), array(y)

seq = [10, 20, 30, 40, 50, 60]
X, y = split_sequence(seq, 3, 2)

print("Input-Output pairs:")
for i in range(len(X)):
    print(f"X: {X[i]} -> y: {y[i]}")

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=3))
model.add(Dense(2))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=500, verbose=0)

x_input = array([40, 50, 60]).reshape(1, 3)
yhat = model.predict(x_input, verbose=0)
print(f"\nInput: {x_input.flatten()}")
print(f"Predicted next 2 values: {[round(v, 2) for v in yhat[0]]}")