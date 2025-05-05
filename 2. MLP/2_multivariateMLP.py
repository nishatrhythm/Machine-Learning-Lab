# Multivariate MLP (Multiple Inputs, Single Output)

from numpy import array, hstack
from keras.models import Sequential
from keras.layers import Dense

in1 = array([10, 20, 30, 40, 50])
in2 = array([15, 25, 35, 45, 55])
out = in1 + in2
dataset = hstack((in1.reshape(-1,1), in2.reshape(-1,1), out.reshape(-1,1)))

def split_multi(seq, n_steps):
    X, y = [], []
    for i in range(len(seq) - n_steps):
        X.append(seq[i:i+n_steps, :])
        y.append(seq[i+n_steps, :])
    return array(X), array(y)

X, y = split_multi(dataset, 2)
X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))

print("Input-Output pairs:")
for i in range(len(X)):
    print(f"X: {X[i]} -> y: {y[i]}")

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=X.shape[1]))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=500, verbose=0)

x_input = array([[30, 35, 65], [40, 45, 85]]).reshape(1, 6)
yhat = model.predict(x_input, verbose=0)
print(f"\nInput: {x_input.flatten()}")
print(f"Predicted values (in1, in2, sum): {[round(val, 2) for val in yhat[0]]}")