import numpy as np

# ------------- Tensor Shapes -------------
print("===== Tensor Shapes =====")

# Creating tensors of different dimensions
scalar = np.array(5)                              # 0D tensor (scalar)
vector = np.array([1, 2, 3, 4])                   # 1D tensor (vector)
matrix = np.array([[1, 2, 3], [4, 5, 6]])         # 2D tensor (matrix)
tensor_3d = np.array([[[1, 2], [3, 4]], 
                      [[5, 6], [7, 8]]])          # 3D tensor

# Checking shapes
print(f"Scalar shape: {scalar.shape}")
print(f"Vector shape: {vector.shape}")
print(f"Matrix shape: {matrix.shape}")
print(f"3D tensor shape: {tensor_3d.shape}")

# Reshaping tensors
reshaped_vector = vector.reshape(2, 2)
print(f"\nReshaped vector (2x2):\n{reshaped_vector}")
print(f"New shape: {reshaped_vector.shape}")

# Using -1 to automatically calculate one dimension
auto_reshape = matrix.reshape(3, -1)
print(f"\nAuto-reshaped matrix (3x2):\n{auto_reshape}")
print(f"New shape: {auto_reshape.shape}")

# Flattening a tensor
flattened = tensor_3d.flatten()
print(f"\nFlattened 3D tensor: {flattened}")
print(f"Flattened shape: {flattened.shape}")

# Transposing matrices
transposed = matrix.T
print(f"\nOriginal matrix:\n{matrix}")
print(f"Transposed matrix:\n{transposed}")

# ------------- Tensor Dot Product -------------
print("\n\n===== Tensor Dot Product =====")

# Basic dot product between vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)  # Same as a.dot(b)
print(f"Vector dot product: {dot_product}")  # Should be 1*4 + 2*5 + 3*6 = 32

# Matrix multiplication using dot
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
matrix_mult = np.dot(A, B)
print(f"\nMatrix multiplication result:\n{matrix_mult}")

# Tensordot - more general tensor dot product
# Let's create two 3D tensors
tensor1 = np.arange(24).reshape(2, 3, 4)
tensor2 = np.arange(24).reshape(4, 3, 2)

print(f"\nTensor1 shape: {tensor1.shape}")
print(f"Tensor2 shape: {tensor2.shape}")

# Tensordot with explicit axes
# Contract the last axis of tensor1 with the first axis of tensor2
result1 = np.tensordot(tensor1, tensor2, axes=([2], [0]))
print(f"\nTensordot with axes=([2], [0])")
print(f"Result shape: {result1.shape}")
print(f"Result is a 4D tensor with shape (2, 3, 3, 2)")

# Tensordot with a single integer (number of last dimensions to contract)
result2 = np.tensordot(tensor1, tensor2, axes=1)
print(f"\nTensordot with axes=1")
print(f"Result shape: {result2.shape}")

# Tensordot as matrix multiplication (2D case)
C = np.array([[1, 2, 3], [4, 5, 6]])
D = np.array([[7, 8], [9, 10], [11, 12]])
result3 = np.tensordot(C, D, axes=([1], [0]))
print(f"\nTensordot as matrix multiplication:")
print(f"Result:\n{result3}")
print(f"Same as np.dot(C, D):\n{np.dot(C, D)}")

# Advanced: Einstein summation (similar to tensordot but more flexible)
print("\n===== Bonus: Einstein Summation =====")
result4 = np.einsum('ijk,klm->ijlm', tensor1, tensor2)
print(f"Einstein summation result shape: {result4.shape}")
print(f"Same as tensordot result: {np.array_equal(result1, result4)}")