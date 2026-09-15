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

@njit
def compute_vorticity_2d(u, w, dx, dz):
    """Обчислення вихору для одного 2D кадру (M, N)"""
    M, N = u.shape
    vort = np.zeros((M, N))

    # Границя i = 0
    for j in range(2, N - 2):
        dw_dx = (w[1, j] - w[0, j]) / dx
        du_dz = (u[0, j + 1] - u[0, j - 1]) / (2.0 * dz)
        vort[0, j] = dw_dx - du_dz

    # Внутрішня область
    for i in range(2, M - 2):
        for j in range(2, N - 2):
            dw_dx = (w[i + 1, j] - w[i - 1, j]) / (2.0 * dx)
            du_dz = (u[i, j + 1] - u[i, j - 1]) / (2.0 * dz)
            vort[i, j] = dw_dx - du_dz

    # Границя i = M - 1
    for j in range(2, N - 2):
        dw_dx = (w[M - 1, j] - w[M - 2, j]) / dx
        du_dz = (u[M - 1, j + 1] - u[M - 1, j - 1]) / (2.0 * dz)
        vort[M - 1, j] = dw_dx - du_dz

    return vort

@njit(parallel=True)
def compute_vorticity_3d(u_series, w_series, dx, dz):
    """Обчислення вихору для всього часового ряду (num_steps, M, N)"""
    num_steps = u_series.shape[0]
    M = u_series.shape[1]
    N = u_series.shape[2]
    
    vort_series = np.zeros((num_steps, M, N))
    
    for t in prange(num_steps):
        vort_series[t] = compute_vorticity_2d(u_series[t], w_series[t], dx, dz)
        
    return vort_series