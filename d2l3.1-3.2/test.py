import math
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# n = 10000
# a = torch.ones(n)
# b = torch.ones(n)
# c = torch.zeros(n)
# t = time.time()
# for i in range(n):
#     c[i] = a[i] + b[i]
# print(f'{time.time() - t:.5f} sec')

############################################
# def normal(x, mu, sigma):
#     p = 1 / math.sqrt(2 * math.pi * sigma**2) # math.sqrt: square root
#     return p * np.exp(-0.5 * (x - mu)**2 / sigma**2) # np.exp: exponential


# x = np.arange(-7, 7, 0.01)

# params = [(0, 1), (0, 2), (3, 1)]
# plt.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
#         ylabel='p(x)', figsize=(4.5, 2.5),
#         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
############################################

def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)  
    return p * np.exp(-0.5 * (x - mu)**2 / sigma**2)  
x = np.arange(-7, 7, 0.01)

params = [(0, 1), (0, 2), (3, 1)]

plt.figure(figsize=(4.5, 2.5))

for mu, sigma in params:
    plt.plot(x, normal(x, mu, sigma), label=f'mean {mu}, std {sigma}')

plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()

# Show the plot
plt.show()