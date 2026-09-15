from numba import njit, prange
import numpy as np

from ..solvers import *

from ..convection.velocity import *
from ..convection.beta import *
from ..physics import *

from .slae import build_rhs
from ..numerics.operators import divergence
from ..boundary import apply_boundary_condition_pressure

# steps
@njit(parallel=True)
def step1(u, u_, w, w_, beta, grid_param, simulate_param, physical_param):
    M = grid_param.M
    N = grid_param.N

    dx = grid_param.dx
    dz = grid_param.dz

    Re = simulate_param.Re
    tau = simulate_param.tau

    for i in prange(2, M - 2):
        for j in range(2, N - 2):
            u_[i, j] = u[i, j] + tau * (-conv_u_tvd(u, dx, i, j) - conv_uw_tvd(u, w, dz, i, j) + diff_u_full(u, w, beta, dx, dz, Re, i, j, physical_param))
        
    for i in prange(2, M - 2):
        for j in range(2, N - 2):
            w_[i, j] = w[i, j] + tau * (-conv_w_tvd(w, dz, i, j) - conv_wu_tvd(u, w, dx, i, j) + diff_w_full(u, w, beta, dx, dz, Re, i, j, physical_param) - gravity_force(beta, i, j, physical_param))

def step2(u_, w_, u_2, w_2, beta, p, phi, grid_param, simulate_param, physical_param, PETSc_param):
    PETSc_param.update(beta, physical_param.y)
    rhs = -build_rhs(u_, w_, u_2, w_2, beta, grid_param, simulate_param, physical_param)

    p = PETSc_param.solve(rhs, p)

@njit(parallel=True)
def step3(u, u_, w, w_, p, grid_param, simulate_param):
    M = grid_param.M
    N = grid_param.N

    dx = grid_param.dx
    dz = grid_param.dz

    tau = simulate_param.tau

    for i in prange(2, M - 2):
        for j in range(2, N - 2):
            u[i, j] = u_[i, j] - tau * (p[i + 1, j] - p[i, j]) / dx

    for i in prange(2, M - 2):
        for j in range(2, N - 2):
            w[i, j] = w_[i, j] - tau * (p[i, j + 1] - p[i, j]) / dz

@njit(parallel=True)
def step4(u_1, w_1, u_2, w_2, beta, grid_param, simulate_param, physical_param):
    x = physical_param.x
    y = physical_param.y

    M = grid_param.M
    N = grid_param.N

    dx = grid_param.dx
    dz = grid_param.dz

    inertia = physical_param.C_A + x
    stokes_factor = physical_param.C_1D / physical_param.d**2
    newtons_factor = physical_param.C_2D / physical_param.d

    nu = physical_param.nu
    D = physical_param.D
    g = physical_param.g

    Re = simulate_param.Re
    Sh = simulate_param.Sh
    tau = simulate_param.tau

    for i in range(2, M - 2):
        for j in range(2, N - 2):
            v21_x = u_2[i, j] - u_1[i, j]
            v21_z = w_2[i, j] - w_1[i, j]
            v21_mod = (v21_x**2 + v21_z**2)**0.5 + 1E-12
            
            f_D_x = -(stokes_factor * nu_x(u_1, nu, dx, Re, i, j) * v21_x + newtons_factor * v21_mod * v21_x)
            f_D_z = -(stokes_factor * nu_z(w_1, nu, dz, Re, i, j) * v21_z + newtons_factor * v21_mod * v21_z)
            
            u_2[i, j] = u_2[i, j] + (tau / inertia) * f_D_x
            w_2[i, j] = w_2[i, j] + (tau / inertia) * (-y * g + f_D_z)
            
            # beta_x = 0.5 * (beta[i + 1, j] + beta[i, j])
            # beta_z = 0.5 * (beta[i, j + 1] + beta[i, j])
            
            # beta_x_safe = max(beta_x, 1E-12)
            # beta_z_safe = max(beta_z, 1E-12)
            
            # grad_beta_x = (beta[i + 1, j] - beta[i, j]) / dx
            # grad_beta_z = (beta[i, j + 1] - beta[i, j]) / dz
            
            # w_2_dx = -1.0 / beta_x_safe * D_x(u_2, D, dx, Sh, i, j) * grad_beta_x
            # w_2_dz = -1.0 / beta_z_safe * D_z(w_2, D, dz, Sh, i, j) * grad_beta_z
            
            # u_2[i, j] += w_2_dx
            # w_2[i, j] += w_2_dz

@njit(parallel=True)
def step4_semi_fully_implicit(u_1, w_1, u_2, w_2, beta, grid_param, simulate_param, physical_param):
    x = physical_param.x
    y = physical_param.y

    M = grid_param.M
    N = grid_param.N

    dx = grid_param.dx
    dz = grid_param.dz

    inertia = physical_param.C_A + x
    stokes_factor = physical_param.C_1D / physical_param.d**2
    newtons_factor = physical_param.C_2D / physical_param.d

    nu = physical_param.nu
    D = physical_param.D
    g = physical_param.g

    Re = simulate_param.Re
    Sh = simulate_param.Sh
    tau = simulate_param.tau

    for i in range(2, M - 2):
        for j in range(2, N - 2):
            # Відносна швидкість та її модуль
            v21_x = u_2[i, j] - u_1[i, j]
            v21_z = w_2[i, j] - w_1[i, j]
            v21_mod = (v21_x**2 + v21_z**2)**0.5 + 1E-12
            
            # Локальні базові коефіцієнти взаємодії
            K_x = stokes_factor * nu_x(u_1, nu, dx, Re, i, j) + newtons_factor * v21_mod
            K_z = stokes_factor * nu_z(w_1, nu, dz, Re, i, j) + newtons_factor * v21_mod
            
            # Безрозмірні комплекси релаксації імпульсу
            alpha_x = (tau / inertia) * K_x
            alpha_z = (tau / inertia) * K_z
            gamma_x = tau * K_x  
            gamma_z = tau * K_z

            # # Градієнти та дифузійні поправки
            # beta_x = 0.5 * (beta[i + 1, j] + beta[i, j])
            # beta_z = 0.5 * (beta[i, j + 1] + beta[i, j])
            
            # beta_x_safe = max(beta_x, 1E-12)
            # beta_z_safe = max(beta_z, 1E-12)
            
            # grad_beta_x = (beta[i + 1, j] - beta[i, j]) / dx
            # grad_beta_z = (beta[i, j + 1] - beta[i, j]) / dz
            
            # w_2_dx = -1.0 / beta_x_safe * D_x(u_2, D, dx, Sh, i, j) * grad_beta_x
            # w_2_dz = -1.0 / beta_z_safe * D_z(w_2, D, dz, Sh, i, j) * grad_beta_z

            w_2_dx = 0.0
            w_2_dz = 0.0


            # Формуємо праві частини (explicit праве крило для системи)
            # Для рідини базове значення — її поточна швидкість
            rhs_u1 = u_1[i, j]
            rhs_w1 = w_1[i, j]
            
            # Для частинок додаємо зовнішні сили (гравітацію та дифузійні поправки)
            rhs_u2 = u_2[i, j] + w_2_dx
            rhs_w2 = w_2[i, j] + (tau / inertia) * (-y * g) + w_2_dz

            # Точний аналітичний розв'язок системи 2х2 (Крамер)
            denom_x = (1.0 + gamma_x) * (1.0 + alpha_x) - gamma_x * alpha_x
            denom_z = (1.0 + gamma_z) * (1.0 + alpha_z) - gamma_z * alpha_z
            
            if denom_x < 1E-14: denom_x = 1E-14
            if denom_z < 1E-14: denom_z = 1E-14

            u_1[i, j] = ((1.0 + alpha_x) * rhs_u1 + gamma_x * rhs_u2) / denom_x
            u_2[i, j] = (alpha_x * rhs_u1 + (1.0 + gamma_x) * rhs_u2) / denom_x
            
            w_1[i, j] = ((1.0 + alpha_z) * rhs_w1 + gamma_z * rhs_w2) / denom_z
            w_2[i, j] = (alpha_z * rhs_w1 + (1.0 + gamma_z) * rhs_w2) / denom_z

@njit(parallel=True)
def step5(u_2, w_2, beta, beta_, grid_param, simulate_param, physical_param):
    M = grid_param.M
    N = grid_param.N

    dx = grid_param.dx
    dz = grid_param.dz

    tau = simulate_param.tau

    for i in prange(2, M - 2):
        for j in range(2, N - 2):
            beta_[i, j] = beta[i, j] + tau * (
                        -conv_beta_tvd(u_2, w_2, beta, dx, dz, i, j) + diff_beta(u_2, w_2, beta, dx, dz, i, j, simulate_param, physical_param))

@njit(parallel=True)
def step_end(u, w, u_2, w_2, beta, Div, grid_param, simulate_param, physical_param):
    y = physical_param.y

    M = grid_param.M
    N = grid_param.N

    dx = grid_param.dx
    dz = grid_param.dz

    for i in prange(2, M - 2):
        for j in range(2, N - 2):
            div_beta_w2 = -conv_beta_tvd(u_2, w_2, beta, dx, dz, i, j) + conv_beta_tvd(u, w, beta, dx, dz, i, j)

            Div[i, j] = divergence(u, w, dx, dz, i, j) - y * (div_beta_w2)