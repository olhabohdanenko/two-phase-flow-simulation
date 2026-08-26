from numba import njit, prange

import numpy as np

# Boundary Condition
@njit
def apply_boundary_condition_velocity(u, w, u_, w_, grid_param, simulate_param):
    M = grid_param.M
    N = grid_param.N

    # Ліва
    # Вхід рідини
    u[0, :] = u[1, :] = simulate_param.V_x
    u_[0, :] = u_[1, :] = simulate_param.V_x
    w[0, :] = w[1, :] = 0.0
    w_[0, :] = w_[1, :] = 0.0

    # Права
    # Вільний вихід рідини
    u[M - 1, :] = u[M - 2, :] = u[M - 3, :]
    w[M - 1, :] = w[M - 2, :] = w[M - 3, :]
    u_[M - 1, :] = u_[M - 2, :] = u_[M - 3, :]
    w_[M - 1, :] = w_[M - 2, :] = w_[M - 3, :]

    # Нижня
    # Вільне ковзання
    u[:, 0] = u[:, 1] = u[:, 2]
    u_[:, 0] = u_[:, 1] = u_[:, 2]
    w[:, 0] = w[:, 1] = 0.0
    w_[:, 0] = w_[:, 1] = 0.0

    # Верхня
    # Вільне ковзання
    u[:, N - 1] = u[:, N - 2] = u[:, N - 3]
    w[:, N - 2] = w[:, N - 3] = 0.0
    u_[:, N - 1] = u_[:, N - 2] = u_[:, N - 3]
    w_[:, N - 2] = w_[:, N - 3] = 0.0

@njit
def apply_boundary_condition_beta(beta, u_2, grid_param, simulate_param):
    # Ліва
    # Вхід рідини
    beta[0, :] = beta[1, :] = beta[2, :]

    # Права
    # Вільний вихід рідини
    beta[grid_param.M - 1, :] = beta[grid_param.M - 2, :] = beta[grid_param.M - 3, :]

    # Нижня
    # Вільне ковзання
    beta[:, 0] = beta[:, 1] = beta[:, 2]

    # Верхня
    # Вільне ковзання
    beta[:, grid_param.N - 1] = beta[:, grid_param.N - 2] = beta[:, grid_param.N - 3]

@njit
def apply_boundary_condition_inject(beta, u_2, grid_param, simulate_param):
    X_inject = 0.01
    Y_inject = 0.01
    I_max_x = int(X_inject / grid_param.dx) + 1
    I_max_y = int(Y_inject / grid_param.dz) + 1
    for i in range(1, I_max_x):
        for j in range(1, I_max_y):
            beta[i, grid_param.N - j] = 1E-4

@njit
def apply_boundary_condition(u, w, u_, w_, grid_param, simulate_param):
    for i in range(1, grid_param.M - 1):
        x = (i - 0.5) * grid_param.dx
        
        in_nozzle = False
        for pos in simulate_param.nozzle_positions:
            if abs(x - pos) <= simulate_param.nozzle_r:
                in_nozzle = True
                break
                
        if in_nozzle:
            w[i, 0] = w[i, 1] = simulate_param.V_z
            u[i, 0] = u[i, 1] = 0.0
            w_[i, 0] = w_[i, 1] = simulate_param.V_z
            u_[i, 0] = u_[i, 1] = 0.0