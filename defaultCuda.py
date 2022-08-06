import math
import numpy as np
from numba import jit, cuda, types, float32
import time

@jit
def cpu_matrix_mult_jit(x, y, z):
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            tmp = 0
            for k in range(x.shape[1]):
                tmp += x[i, k] * y[k, j]
            z[i, j] = tmp
    return z
