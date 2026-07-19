# ============================================
# Lab 17: PyTorch Basics
# ============================================
# Agenda (~50 min):
#   1. What PyTorch is and why it's important  (~8 min)
#   2. Tensors: creating and working with them (~10 min)
#   3. Tensor operations and NumPy bridge      (~10 min)
#   4. Autograd: automatic differentiation     (~12 min)
#   5. A tiny neural network                   (~10 min)
#
# Install first if needed:  pip install torch

import torch

# ---- What is PyTorch? ----
# The most popular deep learning library (used by OpenAI, Meta, Tesla...).
# Core idea: a "tensor" = a NumPy array that can
#   1. run on a GPU (massive speed-up)
#   2. remember its own gradients (needed for training)

# ---- A tensor looks and feels like a NumPy array ----
# prices = torch.tensor([100, 102, 105, 103])
# print(prices)                             # tensor([100, 102, 105, 103])
# print(type(prices))                       # <class 'torch.Tensor'>
# print(prices * 2)                         # vectorized, just like NumPy


# ---- GPU check (the big reason PyTorch exists) ----
# print(torch.cuda.is_available())          # True if you have an NVIDIA GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
# t = prices.to(device)                   # move the tensor to the GPU


# ============================================
# 2. TENSORS: CREATING AND WORKING WITH THEM
# ============================================

# ---- Creating tensors (same builders as NumPy) ----
# a = torch.tensor([1, 2, 3, 4, 5])         # 1D
# b = torch.tensor([[1, 2, 3],
#                   [4, 5, 6]])             # 2D (2 rows, 3 cols)
# print(a)
# print(b)

# print(torch.zeros(5))                     # [0., 0., 0., 0., 0.]
# print(torch.ones(2, 3))                   # 2x3 of 1s
# print(torch.arange(0, 10, 2))             # [0, 2, 4, 6, 8]
# print(torch.linspace(0, 1, 5))            # [0., 0.25, 0.5, 0.75, 1.]
# print(torch.rand(3))                      # 3 random floats in [0, 1)
# print(torch.randn(2, 2))                  # normal distribution (ML weights!)


# ---- Key attributes ----
# b = torch.tensor([[1, 2, 3],
#                   [4, 5, 6]], dtype=torch.float32)
# print(b)
# print()
# print(b.shape)                            # torch.Size([2, 3])
# print()
# print(b.ndim)                             # 2
# print()
# print(b.device)                           # cpu (or cuda:0)
# print()
# print(b.dtype)                            # torch.int64 -> one type for all


# m = torch.tensor([1, 2, 3, 4, 5])


# two_dim_array = torch.tensor(
#     [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
# )


# three_dim_array = torch.tensor(
#     [[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]
# )

# print(three_dim_array)

# print()
# print()
# print("dim = ", end="")
# print(three_dim_array.ndim)


# array = torch.tensor([
#     [6,7,8,9,6,7,8,9,6,7,8,9,6,7,8,9,6,7,88,9],
#     [6,7,8,9,6,7,8,9,6,7,8,9,6,7,8,9,6,7,88,9],
#     [6,7,8,9,6,7,8,9,6,7,8,9,6,7,8,9,6,7,88,9],
#     [6,7,8,9,6,7,8,9,6,7,8,9,6,7,8,9,6,7,88,9],
#     [6,7,8,9,6,7,8,9,6,7,8,9,6,7,8,9,6,7,78998,9],
#     [6,7,8,9,6,7,8,9,6,7,8,9,6,7,8,9,6,7,88,9]
# ])

# print(array[-2,-2])


# ---- Indexing and slicing: exactly like NumPy (Lab 16) ----
# m = torch.tensor(
#     [
#         [1,  2,  3,  4],
#         [5,  6,  7,  8],
#         [9, 10, 11, 12]
#     ])
# print(m[-1, -2])                            # tensor(11)
# print(m[:, 1])                            # whole 2nd column
# print(m[-1,1:-1])
# print(m[m > 6])                           # boolean indexing works too!
# print(m[-1, -2].item())                     # .item() -> plain Python number


# ---- Reshaping ----
# nums = torch.arange(12)
# print(nums)
# grid = nums.reshape(3, 4)
# print("")
# print(grid)
# print(grid.flatten())


# ============================================
# 3. TENSOR OPERATIONS AND NUMPY BRIDGE
# ============================================

# ---- Element-wise math: no loops ----
# a = torch.tensor([1, 2, 3, 4])
# b = torch.tensor([10, 20, 30, 40])
# print(a + b)                              # [11, 22, 33, 44]
# print(a * b)                              # [10, 40, 90, 160] element-wise
# print(b / a)                              # [10., 10., 10., 10.]
# print(a ** 2)                             # [1, 4, 9, 16]

# ---- Broadcasting with a single number ----
# prices = torch.tensor([100., 250., 80., 120.])
# print(prices * 1.18)                      # 18% tax on all at once

# ---- Aggregations ----
# sales = torch.tensor([250., 300., 180., 420., 390.])
# print(sales.sum())                        # tensor(1540.)
# print(sales.mean())                       # tensor(308.)
# print(sales.min(), sales.max())

# ---- Matrix multiplication (the heart of neural networks) ----
# A = torch.tensor([[1., 2.],
#                   [3., 4.]])
# B = torch.tensor([[5., 6.],
#                   [7., 8.]])
# print(A * B)                              # element-wise (NOT matrix mult)
# print(A @ B)                              # true matrix multiplication

# ---- NumPy bridge: convert both ways ----
# import numpy as np
# arr = np.array([1, 2, 3])
# t = torch.from_numpy(arr)                 # NumPy -> tensor
# print(t)
# back = t.numpy()                          # tensor -> NumPy
# print(back)

