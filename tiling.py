import math
import numpy as np
from numba import  cuda
import time

# Bloques
M =256
#Threads
N = 32

# Multiplicación de matrices en gpu mediante el método Tiling usando memoria globa>@cuda.jit
def matrix_mult_tiling(A, B, C):
        x, y = cuda.grid(2)

        # Threads por bloque
        TPB = N

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        # bloques por grid
        bpg = cuda.gridDim.x


        if x >= C.shape[0] and y >= C.shape[1]:
                # Salir si (x, y) está fuera del límite válido
                return None

        tmp = 0.
        for i in range(bpg):
                for j in range(TPB):
                        tmp += A[x, ty + i * TPB] * B[tx + i * TPB, y]
        C[x, y] = tmp

# Código para llamar al método matrix_mult_tiling
def tiling(x, y, z):
        # Parte de inicialización
        d_x = cuda.to_device(x)
        d_y = cuda.to_device(y)
        d_z = cuda.to_device(z)

        block_size = (N, N)
        grid_size = (int(M/N), int(M/N))

        matrix_mult_tiling[grid_size,block_size](d_x, d_y, d_z)
        return d_z.copy_to_host()