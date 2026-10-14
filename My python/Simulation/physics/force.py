# force.py
from numba import njit, prange
import numpy as np

@njit
def gravity_force(beta, i, j, physical_param):
    return 0.5 * (beta[i, j] + beta[i, j + 1]) * physical_param.y * physical_param.g

@njit(parallel=True)
def calc_gravity_force_field(beta, physical_param, force_array):
    M, N = beta.shape
    for i in prange(M):
        for j in range(N - 1):
            force_array[i, j] = 0.5 * (beta[i, j] + beta[i, j + 1]) * physical_param.y * physical_param.g
        force_array[i, N - 1] = beta[i, N - 1] * physical_param.y * physical_param.g