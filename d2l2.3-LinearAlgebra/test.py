import torch

# x = torch.tensor(3.0)
# y = torch.tensor(2.0)
# print(x + y, x * y, x / y, x**y)

# x = torch.arange(3)
# print(x[2])

# x = torch.arange(12).reshape(3, 4)
# print(x.T) # take transpose of x

# A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
# print(A == A.T) # check if A is symmetric

# x = torch.arange(24).reshape(2, 3, 4) #2 matrices, 3 rows, 4 columns
# print(x)

# A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
# B = A.clone()  # Assign a copy of A to B by allocating new memory
# print(A, A + B)

# a = 2
# X = torch.arange(24).reshape(2, 3, 4)
# print(a + X, (a * X).shape)

# x = torch.arange(3, dtype=torch.float32)
# print(x, x.sum()) # sum of all elements

# A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
# print(A.shape, A.sum(axis=1).shape)
# print(A.sum(axis=1)) # sum of all rows
# sum_A = A.sum(axis=1, keepdims=True) # keep the dimension
# print(sum_A, sum_A.shape) # sum of all rows, keep the dimension
# print(A.cumsum(axis=0)) # cumulative sum along columns

# x = torch.ones(3, dtype=torch.float32)
# y = torch.arange(3, dtype = torch.float32)
# print(x,y,torch.dot(x, y))

# A=torch.arange(6).reshape(2,3)
# B=torch.arange(6).reshape(3,2)
# print(torch.mm(A,B)) # matrix multiplication

# u = torch.tensor([3.0, -4.0])
# print(torch.norm(u)) # L2 norm
# print(torch.abs(u).sum()) # L1 norm
# print(torch.norm(torch.ones((4, 9)))) # Frobenius norm