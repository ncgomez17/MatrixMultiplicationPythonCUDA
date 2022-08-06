import math
import numpy as np
from numba import jit, cuda, types, float32
import time

# Bloques
M = 128
#Threads
N = 32

@cuda.jit
def matrix_mult_naive(x, y, z):
    i, j = cuda.grid(2)
    if i < z.shape[0] and j < z.shape[1]:
        tmp = 0
        for k in range(x.shape[1]):
            tmp += x[i, k] * y[k, j]
        z[i, j] = tmp

# Código para llamar al método matrix_mult_naive
def naive(x, y, z):
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_z = cuda.device_array(z.shape, np.float64)

    tpb = (N, N)
    # Bloques por grid
    bpd_x = math.ceil(x.shape[0]/tpb[0])
    bpd_y = math.ceil(y.shape[1]/tpb[1])
    bpg = (bpd_x, bpd_y)

    matrix_mult_naive[bpg, tpb](d_x, d_y, d_z)

    return d_z.copy_to_host()