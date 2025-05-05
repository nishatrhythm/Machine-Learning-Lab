import numpy as np
import csv

data = [0, 1, 2, 3, 4, 5, 6, 7, 34, 35, 36, 78, 79, 80]

with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(data)

read_data = []
with open('data.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        read_data = [int(x) for x in row]

print("Data read from CSV:", read_data)

tensor_2d = np.array([
    [5, 78, 2, 34, 0],
    [6, 79, 3, 35, 1],
    [7, 80, 4, 36, 2]
])

print("\n2D Tensor:")
print(tensor_2d)