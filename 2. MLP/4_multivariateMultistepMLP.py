# Multivariate Multi-step MLP

from numpy import array, hstack
from keras.models import Sequential
from keras.layers import Dense

in1 = array([10, 20, 30, 40, 50])
in2 = array([15, 25, 35, 45, 55])
out = in1 + in2
dataset = hstack((in1.reshape(-1,1), in2.reshape(-1,1), out.reshape(-1,1)))

def split_sequence(seq, n_in, n_out):
    X, y = [], []
    for i in range(len(seq) - n_in - n_out + 1):
        X.append(seq[i:i+n_in, :])
        y.append(seq[i+n_in:i+n_in+n_out, :])
    return array(X), array(y)

X, y = split_sequence(dataset, 2, 2)
X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
y = y.reshape((y.shape[0], y.shape[1]*y.shape[2]))

print("Input-Output pairs:")
for i in range(len(X)):
    print(f"X: {X[i]} -> y: {y[i]}")

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=X.shape[1]))
model.add(Dense(y.shape[1]))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=500, verbose=0)

x_input = array([[30, 35, 65], [40, 45, 85]]).reshape(1, X.shape[1])
yhat = model.predict(x_input, verbose=0)
print(f"\nInput: {x_input.flatten()}")
print(f"Predicted values (next 2 steps, 3 features each): {yhat[0].reshape(2, 3)}")