from numba import njit, prange

import numpy as np

"""
Потік зліва направо з вільним ковзанням на дні і стелі
"""
@njit
def apply_boundary_condition_velocity(u, w, grid_param, simulate_param):
    M = grid_param.M
    N = grid_param.N

    """
    Потік зліва направо
    """
    # Ліва
    # Вхід рідини
    u[0, :] = u[1, :] = simulate_param.V_x
    w[0, :] = w[1, :] = 0.0

    # Права
    # Вільний вихід рідини
    u[M - 1, :] = u[M - 2, :] = u[M - 3, :]
    w[M - 1, :] = w[M - 2, :] = w[M - 3, :]

    # Нижня
    # Вільне ковзання
    u[:, 0] = u[:, 1] = u[:, 2]
    w[:, 0] = w[:, 1] = 0.0

    # Верхня
    # Вільне ковзання
    u[:, N - 1] = u[:, N - 2] = u[:, N - 3]
    w[:, N - 1] = w[:, N - 2] = w[:, N - 3] = 0.0

@njit
def apply_boundary_condition_velocity_2(u_2, w_2, grid_param, simulate_param):
    M = grid_param.M
    N = grid_param.N

    """
    Просто чаша
    """
    # Ліва
    # Вільне ковзання
    u_2[0, :] = u_2[1, :] = 0.0
    w_2[0, :] = w_2[1, :] = w_2[2, :]

    # Права
    # Вільне ковзання
    # u_2[M - 1, :] = u_2[M - 2, :] = u_2[M - 3, :] = 0.0
    u_2[M - 1, :] = u_2[M - 2, :] = u_2[M - 3, :] = u_2[M - 4, :]
    w_2[M - 1, :] = w_2[M - 2, :] = w_2[M - 3, :]

    # Нижня
    # Вільне ковзання
    u_2[:, 0] = u_2[:, 1] = u_2[:, 2]
    w_2[:, 0] = w_2[:, 1] = 0.0

    # Верхня
    # Вільний вихід
    u_2[:, N - 1] = u_2[:, N - 2] = u_2[:, N - 3]
    w_2[:, N - 1] = w_2[:, N - 2] = w_2[:, N - 3] = 0.0
    # w_2[:, N - 1] = w_2[:, N - 2] = w_2[:, N - 3] = w_2[:, N - 4]


@njit
def apply_boundary_condition_beta(beta, u_2, grid_param, simulate_param):
    M = grid_param.M
    N = grid_param.N
    # Ліва
    # Вхід рідини
    beta[0, :] = beta[1, :] = beta[2, :]

    # Права
    # Вільний вихід рідини
    beta[M - 1, :] = beta[M - 2, :] = beta[M - 3, :]

    # Нижня
    # Вільне ковзання
    beta[:, 0] = beta[:, 1] = beta[:, 2]

    # Верхня
    # Вільне ковзання
    beta[:, N - 1] = beta[:, N - 2] = beta[:, N - 3]

"""
Потік зліва направо
"""
@njit
def apply_boundary_condition_pressure(p, grid_param, simulate_param):
    M = grid_param.M
    N = grid_param.N
    # Ліва
    # Вхід рідини
    p[0, :] = p[1, :] = p[2, :]

    # Права
    # Вільний вихід рідини
    p[M - 1, :] = p[M - 2, :] = 0.0

    # Нижня
    # Вільне ковзання
    p[:, 0] = p[:, 1] = p[:, 2]

    # Верхня
    # Вільне ковзання
    p[:, N - 1] = p[:, N - 2] = p[:, N - 3]

@njit
def apply_boundary_condition_inject(beta, u_2, grid_param, simulate_param):
    X_inject = 0.02
    Y_inject = 0.01
    I_max_x = int(X_inject / grid_param.dx) + 1
    I_max_y = int(Y_inject / grid_param.dz) + 1
    for i in range(1, I_max_x):
        for j in range(1, I_max_y):
            beta[i, grid_param.N - j] = 1E-3

@njit
def apply_boundary_condition(u, w, grid_param, simulate_param):
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