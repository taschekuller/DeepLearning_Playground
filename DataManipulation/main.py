import torch

# x = torch.arange(12, dtype=torch.float32)

# print(x.numel()) : #of element
# print(x.shape) : #length along axis

# tX = x.reshape(3, 4) : reshape -> 3 rows, 4 columns
# print(tX)

# print(torch.zeros((2, 3, 4))) : construct two tensor with 3 rows 4 columns containing 0's 
# print(torch.ones((2, 3, 4))) : construct two tensor with 3 rows 4 columns containing 1's 

# print(torch.randn(3, 4)) : create a list 3 rows 4 columns with random numbers
# print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])) : print a tensor with given structure 

# li = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# print(li[:2,:]) # access 1st 2nd indices
# print(li[1:3]) # access from 1 to 3 element of list 

# print(torch.exp(li))

# x = torch.tensor([1.0, 2, 4, 8])
# y = torch.tensor([2, 2, 2, 2])

# print(x + y, x - y, x * y, x / y, x ** y)

# X = torch.arange(12, dtype=torch.float32).reshape((3,4))
# Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# print(torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)) #for dim=0 3+3 rows, for dimn=1 4+4 columns

# a = torch.arange(3).reshape((3, 1))
# b = torch.arange(2).reshape((1, 2))
# print(a+b) # even though sizes are different, we can still perform element-wise operation by invoking broadcast mechanism in torch


# # Create tensors X and Y
# X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
# Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# # Create tensor Z with the same shape and type as Y, filled with zeros
# Z = torch.zeros_like(Y)
# print('id(Z):', id(Z))  # Print the memory address of Z

# # Perform an in-place operation on Z
# Z[:] = X + Y #without [:] we can not update the memory address in this example
# print('id(Z):', id(Z))  # Print the memory address of Z again to verify it hasn't changed


# A = X.numpy() #define x before run
# B = torch.from_numpy(A)
# print(type(A), type(B))

