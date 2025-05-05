import numpy as np
import pandas as pd

# Create a NumPy array
data = np.array([[1, 2, 3], 
                 [4, 5, 6], 
                 [7, 8, 9]])

print("NumPy array:")
print(data)
print("\nArray shape:", data.shape)
print("Array dimensions:", data.ndim)
print("Data type:", data.dtype)

# Basic operations
print("\nSum of all elements:", np.sum(data))
print("Mean of all elements:", np.mean(data))
print("Maximum value:", np.max(data))
print("Minimum value:", np.min(data))

# Array manipulation
print("\nTransposed array:")
print(data.T)

# Creating a pandas DataFrame from NumPy array
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
print("\nPandas DataFrame:")
print(df)