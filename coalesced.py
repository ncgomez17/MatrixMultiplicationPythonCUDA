import math
import numpy as np
from numba import jit, cuda, types, float32
import time

M = 256
N = 32

@cuda.jit
def matrix_multiplication(A, B, C):
    # Threads por bloque
    TPB = N
    # Define un array en la memoria compartida
    # El tamaño y tipo de los arrays debe conocerse en tiempo de compilación
    subMatrixA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    subMatrixB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # bloques por grid
    bpg = cuda.gridDim.x

    if x >= C.shape[0] and y >= C.shape[1]:
        # Salir si (x, y) está fuera del límite válido
        return

    # Cada subproceso calcula un elemento en la matriz de resultados.
    # El producto escalar se divide en productos escalares de vectores TPB.
    tmp = 0.
    for i in range(bpg):
                # Precarga de datos en memoria compartida
        subMatrixA[tx, ty] = A[x, ty + i * int(M/N)]
        subMatrixB[tx, ty] = B[tx + i * int(M/N), y]

        # Sincronización de threads.
        cuda.syncthreads()

        # Calcula el producto parcial en la memoria compartida.
        rows, cols = subMatrixB.shape
        for j in range(N):
            tmp += subMatrixA[tx, j] * B[tx, j]

        # Sincronización de threads.
        cuda.syncthreads()

    C[x, y] = tmp

def coalesced(x,y,z):

    block_size = (N, N)
    grid_size = (int(M/N), int(M/N))

    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y.T)
    d_z = cuda.to_device(z)
    matrix_multiplication[grid_size, block_size](d_x, d_y, d_z)
    return d_z.copy_to_host()
