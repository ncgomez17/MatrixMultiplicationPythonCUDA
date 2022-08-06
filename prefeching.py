import math
import numpy as np
from numba import jit, cuda, types, float32
import time

# Multiplicación de matrices en gpu mediante el método Pre-fetching usando memoria>@cuda.jit
def matrix_mult_pre_feching(A, B, C):
    # Threads por bloque
    TPB = N
    # Define un array en la memoria compartida
    # El tamaño y tipo de los arrays debe conocerse en tiempo de compilación
    subMatrixA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    subMatrixB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    preFech1 = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    preFech2 = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # bloques por grid
    bpg = cuda.gridDim.x

    if x >= C.shape[0] and y >= C.shape[1]:
        # Salir si (x, y) está fuera del límite válido
        return None

    # Cada subproceso calcula un elemento en la matriz de resultados.
    # El producto escalar se divide en productos escalares de vectores TPB.
    tmp = 0.
    for i in range(bpg):
	        # Precarga de datos en memoria compartida
        if(i == 0):
          subMatrixA[tx, ty] = A[x, ty + i * TPB]
          subMatrixB[tx, ty] = B[tx + i * TPB, y]
        else:
          subMatrixA[tx, ty] = preFech1[tx, ty]
          subMatrixB[tx, ty] = preFech2[tx, ty]

        preFech1[tx, ty] = A[x, ty + (i+1) * TPB]
        preFech2[tx, ty] = B[tx + (i+1) * TPB, y]

        # Sincronización de threads.
        cuda.syncthreads()
        # Calcula el producto parcial en la memoria compartida.
        for j in range(TPB):
            tmp += subMatrixA[tx, j] * subMatrixB[j, ty]

        # Sincronización de threads.
        cuda.syncthreads()

    C[x, y] = tmp

# Parte de inicialización
# Bloques
M = 256
#Threads
N = 16

def prefetch(x, y, z):
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_z = cuda.to_device(z)

    block_size = (N, N)
    grid_size = (int(M/N), int(M/N))

    matrix_mult_pre_feching[grid_size,block_size](d_x, d_y, d_z)
    return d_z.copy_to_host()


x = np.full((256,256), 3, dtype=np.float32)
y = np.full((256,256), 5, dtype=np.float32)
z = np.full((256,256), 0, dtype=np.float32)

start = time.time()
prefetch(x, y, z)
print ('Tiempo pre-fetching memoria compartida', time.time()-start, 's\n')	