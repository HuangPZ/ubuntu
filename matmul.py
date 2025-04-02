import numpy as np
from numba import njit, prange
import time
import pyRAPL

# Matrix size (NxN)
N = 512
Batch = 1000

# Generate fixed input matrices
A = np.random.randint(0, 2**15, size=(Batch*N, N), dtype=np.int64)
B = np.random.randint(0, 2**15, size=(N, N), dtype=np.int64)
C = np.zeros((Batch*N, N), dtype=np.int64)

@njit(parallel=True)
def matrix_multiply(A, B, C, N, total_rows):
    for i in prange(total_rows):
        for j in range(N):
            temp = 0
            for k in range(N):
                temp += A[i, k] * B[k, j]
            C[i, j] = temp

# Set up pyRAPL
pyRAPL.setup()
meter = pyRAPL.Measurement('matrix_mult')

# Warm-up (JIT compile)
matrix_multiply(A, B, C, N, Batch*N)

# Measure
meter.begin()
start_time = time.time()
# matrix_multiply(A, B, C, N, Batch*N)
# time.sleep(11.16)
end_time = time.time()
meter.end()
result = meter.result
print(meter)
# Total number of multiplications
total_mults = Batch * N * N * N
elapsed_time = end_time - start_time
energy_used = meter.result.pkg[0] / 1e6  - 17.5*elapsed_time # Convert µJ to J

print(f"total mults: {total_mults}")
print(f"Time elapsed: {end_time - start_time:.2f} seconds")

if 1:
    print(f"Energy consumed: {energy_used} J")
    print(f"Energy per 1M multiplication: {1e6*energy_used / total_mults:.2e} J")
else:
    print("⚠️  pyRAPL returned None. The execution may have been too fast.")
