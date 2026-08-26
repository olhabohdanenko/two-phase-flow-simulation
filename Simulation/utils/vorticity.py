from numba import njit, prange
import numpy as np

@njit
def compute_vorticity(u, w, grid_param):
    M = grid_param.M
    N = grid_param.N

    dx = grid_param.dx
    dz = grid_param.dz

    vort = np.zeros((M, N))

    for j in range(2, N - 2):
        dw_dx = (w[1, j] - w[0, j]) / dx
        du_dz = (u[0, j + 1] - u[0, j - 1]) / (2.0 * dz)
        vort[0, j] = dw_dx - du_dz

    for i in range(2, M - 2):
        for j in range(2, N - 2):
            dw_dx = (w[i + 1, j] - w[i - 1, j]) / (2.0 * dx)
            du_dz = (u[i, j + 1] - u[i, j - 1]) / (2.0 * dz)
            vort[i, j] = dw_dx - du_dz

    for j in range(2, N - 2):
        dw_dx = (w[M - 1, j] - w[M - 2, j]) / dx
        du_dz = (u[M - 1, j + 1] - u[M - 1, j - 1]) / (2.0 * dz)
        vort[M - 1, j] = dw_dx - du_dz

    return vort