from numba import njit, prange
import numpy as np

@njit
def divergence(u, w, dx, dz, i, j):
	return (u[i, j] - u[i - 1, j]) / dx + (w[i, j] - w[i, j - 1]) / dz

@njit(parallel=True)
def compute_velocity_1(u, w, u_1, w_1, u_2, w_2, beta, grid_param, physical_param):
    x = physical_param.x

    M = grid_param.M
    N = grid_param.N

    for i in prange (M - 2):
        for j in range (N - 2):
            u_1[i, j] = (u[i, j] - 0.5 * (beta[i, j] + beta[i + 1, j]) * x * u_2[i, j]) / (1 - 0.5 * (beta[i, j] + beta[i + 1, j]) * x)
            w_1[i, j] = (w[i, j] - 0.5 * (beta[i, j] + beta[i, j + 1]) * x * w_2[i, j]) / (1 - 0.5 * (beta[i, j] + beta[i, j + 1]) * x)