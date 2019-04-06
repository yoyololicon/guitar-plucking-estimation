from numba import jit
from time import time
import numpy as np
from scipy.fftpack import rfft


@jit()
def fft2(x, N):
    return rfft(x, N)


fftsize = 2 ** 19
x = np.random.randn(2048)

n = 100

a = time()
for i in range(n):
    y = fft2(x, fftsize)
b = time()
print((b - a) / n * 1000)
