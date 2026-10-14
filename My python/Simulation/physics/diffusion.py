from numba import njit, prange
import numpy as np

from .viscosity import nu_x, nu_z, nu_x_, nu_z_, zeta_x, zeta_z, zeta_x_, zeta_z_, D_x, D_z, compute_effective_viscosity, nu_krieger_dougherty, harmonic_mean, nu_smag
from ..numerics.operators import divergence

# Diffusion
@njit
def diff_u(u, w, beta, dx, dz, Re, i, j, physical_param):
    nu = physical_param.nu
    zeta = physical_param.zeta

    factor = physical_param.A_12 + physical_param.y

    #right/east
    beta_e = beta[i + 1, j]
    nu_e = nu_x(u, nu, dx, Re, i, j)
    zeta_e = zeta_x(u, zeta, dx, Re, i, j)

    F_e_nu = nu_e * (1 + beta_e * factor)
    F_e_zeta = zeta_e + nu_e / 3.0 + beta_e * (zeta_e * physical_param.y + nu_e / 3.0 * factor)

    u_e = (u[i + 1, j] - u[i, j]) / dx
    Div_e = divergence(u, w, dx, dz, i + 1, j)

    #left/west
    beta_w = beta[i, j]
    nu_w = nu_x(u, nu, dx, Re, i - 1, j)
    zeta_w = zeta_x(u, zeta, dx, Re, i - 1, j)

    F_w_nu = nu_w * (1 + beta_w * factor)
    F_w_zeta = zeta_w + nu_w / 3.0 + beta_w * (zeta_w * physical_param.y + nu_w / 3.0 * factor)

    u_w = (u[i, j] - u[i - 1, j]) / dx
    Div_w = divergence(u, w, dx, dz, i, j)


    diff_x = (1.0 / dx) * (F_e_nu * u_e - F_w_nu * u_w)
    diff_x += (1.0 / dx) * (F_e_zeta * Div_e - F_w_zeta * Div_w)

    #top/north
    beta_n = 0.25 * (beta[i, j] + beta[i + 1, j] + beta[i, j + 1] + beta[i + 1, j + 1])
    F_n = nu_z_(w, nu, dz, Re, i, j) * (1 + beta_n * factor)
    u_n = (u[i, j + 1] - u[i, j]) / dz

    #down/south
    beta_s = 0.25 * (beta[i, j - 1] + beta[i + 1, j - 1] + beta[i, j] + beta[i + 1, j])
    F_s = nu_z_(w, nu, dz, Re, i, j - 1) * (1 + beta_s * factor)
    u_s = (u[i, j] - u[i, j - 1]) / dz


    diff_z = (1.0 / dz) * (F_n * u_n - F_s * u_s)

    return diff_x + diff_z
	
@njit
def diff_w(u, w, beta, dx, dz, Re, i, j, physical_param):
    nu = physical_param.nu
    zeta = physical_param.zeta

    factor = physical_param.A_12 + physical_param.y

    #right/east
    beta_e = 0.25 * (beta[i + 1, j] + beta[i + 1, j + 1] + beta[i, j] + beta[i, j + 1])
    F_e = nu_x_(u, nu, dx, Re, i, j) * (1 + beta_e * factor)
    w_e = (w[i + 1, j] - w[i, j]) / dx

    #left/west
    beta_w = 0.25 * (beta[i, j] + beta[i, j + 1] + beta[i - 1, j] + beta[i - 1, j + 1])
    F_w = nu_x_(u, nu, dx, Re, i - 1, j) * (1 + beta_w * factor)
    w_w = (w[i, j] - w[i - 1, j]) / dx


    diff_x = (1.0 / dx) * (F_e * w_e - F_w * w_w)

    #top/north
    beta_n = beta[i, j + 1]
    nu_n = nu_z(w, nu, dz, Re, i, j)
    zeta_n = zeta_z(w, zeta, dz, Re, i, j)

    F_n_nu = nu_n * (1 + beta_n * factor)
    F_n_zeta = zeta_n + nu_n / 3.0 + beta_n * (zeta_n * physical_param.y + nu_n / 3.0 * factor)

    w_n = (w[i, j + 1] - w[i, j]) / dz
    Div_n = divergence(u, w, dx, dz, i, j + 1)

    #down/south
    beta_s = beta[i, j]
    nu_s = nu_z(w, nu, dz, Re, i, j - 1)
    zeta_s = zeta_z(w, zeta, dz, Re, i, j - 1)

    F_s_nu = nu_s * (1 + beta_s * factor)
    F_s_zeta = zeta_s + nu_s / 3.0 + beta_s * (zeta_s * physical_param.y + nu_s / 3.0 * factor)

    w_s = (w[i, j] - w[i, j - 1]) / dz
    Div_s = divergence(u, w, dx, dz, i, j)


    diff_z = (1.0 / dz) * (F_n_nu * w_n - F_s_nu * w_s)
    diff_z += (1.0 / dz) * (F_n_zeta * Div_n - F_s_zeta * Div_s)

    return diff_x + diff_z

@njit
def diff_beta(u_2, w_2, beta, dx, dz, i, j, simulate_param, physical_param):
    Re = simulate_param.Re
    D = physical_param.D

    #right/east
    nu_e = D_x(u_2, D, dx, Re, i, j)
    beta_e = (beta[i + 1, j] - beta[i, j]) / dx

    #left/west
    nu_w = D_x(u_2, D, dx, Re, i - 1, j)
    beta_w = (beta[i, j] - beta[i - 1, j]) / dx

    diff_x = (1.0 / dx) * (nu_e * beta_e - nu_w * beta_w)

    #top/north
    nu_n = D_z(w_2, D, dz, Re, i, j)
    beta_n = (beta[i, j + 1] - beta[i, j]) / dz

    #down/south
    nu_s = D_z(w_2, D, dz, Re, i, j - 1)
    beta_s = (beta[i, j] - beta[i, j - 1]) / dz

    diff_z = (1.0 / dz) * (nu_n * beta_n - nu_s * beta_s)

    return diff_x + diff_z

@njit
def diff_u_full(u, w, beta, dx, dz, Re, i, j, physical_param):
    nu = physical_param.nu
    zeta = physical_param.zeta

    factor = physical_param.A_12 + physical_param.y

    # right/east
    beta_e = beta[i + 1, j]
    nu_e = nu_x(u, nu, dx, Re, i, j)
    zeta_e = zeta_x(u, zeta, dx, Re, i, j)

    F_e_nu = 2.0 * nu_e * (1 + beta_e * factor)
    F_e_zeta = zeta_e * (1 + beta_e * physical_param.y) - (2.0 / 3.0) * nu_e * (1 + beta_e * factor)

    u_e = (u[i + 1, j] - u[i, j]) / dx
    Div_e = divergence(u, w, dx, dz, i + 1, j)

    # left/west
    beta_w = beta[i, j]
    nu_w = nu_x(u, nu, dx, Re, i - 1, j)
    zeta_w = zeta_x(u, zeta, dx, Re, i - 1, j)

    F_w_nu = 2.0 * nu_w * (1 + beta_w * factor)
    F_w_zeta = zeta_w * (1 + beta_w * physical_param.y) - (2.0 / 3.0) * nu_w * (1 + beta_w * factor)

    u_w = (u[i, j] - u[i - 1, j]) / dx
    Div_w = divergence(u, w, dx, dz, i, j)

    diff_x = (1.0 / dx) * (F_e_nu * u_e - F_w_nu * u_w)
    diff_x += (1.0 / dx) * (F_e_zeta * Div_e - F_w_zeta * Div_w)

    # top/north
    beta_n = 0.25 * (beta[i, j] + beta[i + 1, j] + beta[i, j + 1] + beta[i + 1, j + 1])
    F_n = nu_z_(w, nu, dz, Re, i, j) * (1 + beta_n * factor)
    u_n = (u[i, j + 1] - u[i, j]) / dz + (w[i + 1, j] - w[i, j]) / dx

    # down/south
    beta_s = 0.25 * (beta[i, j - 1] + beta[i + 1, j - 1] + beta[i, j] + beta[i + 1, j])
    F_s = nu_z_(w, nu, dz, Re, i, j - 1) * (1 + beta_s * factor)
    u_s = (u[i, j] - u[i, j - 1]) / dz + (w[i + 1, j - 1] - w[i, j - 1]) / dx

    diff_z = (1.0 / dz) * (F_n * u_n - F_s * u_s)

    return diff_x + diff_z


@njit
def diff_w_full(u, w, beta, dx, dz, Re, i, j, physical_param):
    nu = physical_param.nu
    zeta = physical_param.zeta

    factor = physical_param.A_12 + physical_param.y

    # right/east
    beta_e = 0.25 * (beta[i + 1, j] + beta[i + 1, j + 1] + beta[i, j] + beta[i, j + 1])
    F_e = nu_x_(u, nu, dx, Re, i, j) * (1 + beta_e * factor)
    w_e = (w[i + 1, j] - w[i, j]) / dx + (u[i, j + 1] - u[i, j]) / dz

    # left/west
    beta_w = 0.25 * (beta[i, j] + beta[i, j + 1] + beta[i - 1, j] + beta[i - 1, j + 1])
    F_w = nu_x_(u, nu, dx, Re, i - 1, j) * (1 + beta_w * factor)
    w_w = (w[i, j] - w[i - 1, j]) / dx + (u[i - 1, j + 1] - u[i - 1, j]) / dz

    diff_x = (1.0 / dx) * (F_e * w_e - F_w * w_w)

    # top/north
    beta_n = beta[i, j + 1]
    nu_n = nu_z(w, nu, dz, Re, i, j)
    zeta_n = zeta_z(w, zeta, dz, Re, i, j)

    F_n_nu = 2.0 * nu_n * (1 + beta_n * factor)
    F_n_zeta = zeta_n * (1 + beta_n * physical_param.y) - (2.0 / 3.0) * nu_n * (1 + beta_n * factor)

    w_n = (w[i, j + 1] - w[i, j]) / dz
    Div_n = divergence(u, w, dx, dz, i, j + 1)

    # down/south
    beta_s = beta[i, j]
    nu_s = nu_z(w, nu, dz, Re, i, j - 1)
    zeta_s = zeta_z(w, zeta, dz, Re, i, j - 1)

    F_s_nu = 2.0 * nu_s * (1 + beta_s * factor)
    F_s_zeta = zeta_s * (1 + beta_s * physical_param.y) - (2.0 / 3.0) * nu_s * (1 + beta_s * factor)

    w_s = (w[i, j] - w[i, j - 1]) / dz
    Div_s = divergence(u, w, dx, dz, i, j)

    diff_z = (1.0 / dz) * (F_n_nu * w_n - F_s_nu * w_s)
    diff_z += (1.0 / dz) * (F_n_zeta * Div_n - F_s_zeta * Div_s)

    return diff_x + diff_z

@njit
def diff_u_update(u, w, beta, dx, dz, Re, i, j, physical_param):
    nu = physical_param.nu
    zeta = physical_param.zeta

    nu_i_j = compute_effective_viscosity(u, w, beta, dx, dz, i, j, physical_param)
    nu_i1_j = compute_effective_viscosity(u, w, beta, dx, dz, i + 1, j, physical_param)

    factor = physical_param.A_12 + physical_param.y

    # right/east
    beta_e = beta[i + 1, j]
    nu_e = nu_i1_j
    zeta_e = nu_e * (2.0 / 3.0)

    F_e_nu = 2.0 * nu_e * (1 + beta_e * factor)
    F_e_zeta = zeta_e * (1 + beta_e * physical_param.y) - (2.0 / 3.0) * nu_e * (1 + beta_e * factor)

    u_e = (u[i + 1, j] - u[i, j]) / dx
    Div_e = divergence(u, w, dx, dz, i + 1, j)

    # left/west
    beta_w = beta[i, j]
    nu_w = nu_i_j
    zeta_w = nu_w * (2.0 / 3.0)

    F_w_nu = 2.0 * nu_w * (1 + beta_w * factor)
    F_w_zeta = zeta_w * (1 + beta_w * physical_param.y) - (2.0 / 3.0) * nu_w * (1 + beta_w * factor)

    u_w = (u[i, j] - u[i - 1, j]) / dx
    Div_w = divergence(u, w, dx, dz, i, j)

    diff_x = (1.0 / dx) * (F_e_nu * u_e - F_w_nu * u_w)
    diff_x += (1.0 / dx) * (F_e_zeta * Div_e - F_w_zeta * Div_w)

    # top/north
    beta_n = 0.25 * (beta[i, j] + beta[i + 1, j] + beta[i, j + 1] + beta[i + 1, j + 1])

    nu_n_t = 0.5 * (compute_effective_viscosity(u, w, beta, dx, dz, i, j + 1, physical_param) + compute_effective_viscosity(u, w, beta, dx, dz, i + 1, j + 1, physical_param))
    nu_n_d = 0.5 * (nu_i_j + nu_i1_j)
    nu_n = harmonic_mean(nu_n_t, nu_n_d)

    F_n = nu_n * (1 + beta_n * factor)
    u_n = (u[i, j + 1] - u[i, j]) / dz + (w[i + 1, j] - w[i, j]) / dx

    # down/south
    beta_s = 0.25 * (beta[i, j - 1] + beta[i + 1, j - 1] + beta[i, j] + beta[i + 1, j])

    nu_s_t = 0.5 * (nu_i_j + nu_i1_j)
    nu_s_d = 0.5 * (compute_effective_viscosity(u, w, beta, dx, dz, i, j - 1, physical_param) + compute_effective_viscosity(u, w, beta, dx, dz, i + 1, j - 1, physical_param))
    nu_s = harmonic_mean(nu_s_t, nu_s_d)

    F_s = nu_s * (1 + beta_s * factor)
    u_s = (u[i, j] - u[i, j - 1]) / dz + (w[i + 1, j - 1] - w[i, j - 1]) / dx

    diff_z = (1.0 / dz) * (F_n * u_n - F_s * u_s)

    return diff_x + diff_z


@njit
def diff_w_update(u, w, beta, dx, dz, Re, i, j, physical_param):
    nu = physical_param.nu
    zeta = physical_param.zeta

    nu_i_j = compute_effective_viscosity(u, w, beta, dx, dz, i, j, physical_param)
    nu_i_j1 = compute_effective_viscosity(u, w, beta, dx, dz, i, j + 1, physical_param)

    factor = physical_param.A_12 + physical_param.y

    # right/east
    beta_e = 0.25 * (beta[i + 1, j] + beta[i + 1, j + 1] + beta[i, j] + beta[i, j + 1])

    nu_e_r = 0.5 * (compute_effective_viscosity(u, w, beta, dx, dz, i + 1, j, physical_param) + compute_effective_viscosity(u, w, beta, dx, dz, i + 1, j + 1, physical_param))
    nu_e_l = 0.5 * (nu_i_j + nu_i_j1)
    nu_e = harmonic_mean(nu_e_r, nu_e_l)
    
    F_e = nu_e * (1 + beta_e * factor)
    w_e = (w[i + 1, j] - w[i, j]) / dx + (u[i, j + 1] - u[i, j]) / dz

    # left/west
    beta_w = 0.25 * (beta[i, j] + beta[i, j + 1] + beta[i - 1, j] + beta[i - 1, j + 1])

    nu_w_r = 0.5 * (nu_i_j + nu_i_j1)
    nu_w_l = 0.5 * (compute_effective_viscosity(u, w, beta, dx, dz, i - 1, j, physical_param) + compute_effective_viscosity(u, w, beta, dx, dz, i - 1, j + 1, physical_param))
    nu_w = harmonic_mean(nu_w_r, nu_w_l)

    F_w = nu_w * (1 + beta_w * factor)
    w_w = (w[i, j] - w[i - 1, j]) / dx + (u[i - 1, j + 1] - u[i - 1, j]) / dz

    diff_x = (1.0 / dx) * (F_e * w_e - F_w * w_w)

    # top/north
    beta_n = beta[i, j + 1]
    nu_n = nu_i_j1
    zeta_n = nu_n * (2.0 / 3.0)

    F_n_nu = 2.0 * nu_n * (1 + beta_n * factor)
    F_n_zeta = zeta_n * (1 + beta_n * physical_param.y) - (2.0 / 3.0) * nu_n * (1 + beta_n * factor)

    w_n = (w[i, j + 1] - w[i, j]) / dz
    Div_n = divergence(u, w, dx, dz, i, j + 1)

    # down/south
    beta_s = beta[i, j]
    nu_s = nu_i_j
    zeta_s = nu_s * (2.0 / 3.0)

    F_s_nu = 2.0 * nu_s * (1 + beta_s * factor)
    F_s_zeta = zeta_s * (1 + beta_s * physical_param.y) - (2.0 / 3.0) * nu_s * (1 + beta_s * factor)

    w_s = (w[i, j] - w[i, j - 1]) / dz
    Div_s = divergence(u, w, dx, dz, i, j)

    diff_z = (1.0 / dz) * (F_n_nu * w_n - F_s_nu * w_s)
    diff_z += (1.0 / dz) * (F_n_zeta * Div_n - F_s_zeta * Div_s)

    return diff_x + diff_z

@njit
def diff_beta_update(u_2, w_2, beta, dx, dz, i, j, simulate_param, physical_param):
    Re = simulate_param.Re
    Sh = simulate_param.Sh

    D = physical_param.D

    nu_center = D + nu_smag(u_2, w_2, dx, dz, i, j) / Sh

    #right/east
    nu_r = D + nu_smag(u_2, w_2, dx, dz, i + 1, j) / Sh

    nu_e = harmonic_mean(nu_r, nu_center)
    beta_e = (beta[i + 1, j] - beta[i, j]) / dx

    #left/west
    nu_l = D + nu_smag(u_2, w_2, dx, dz, i - 1, j) / Sh

    nu_w = harmonic_mean(nu_l, nu_center)
    beta_w = (beta[i, j] - beta[i - 1, j]) / dx

    diff_x = (1.0 / dx) * (nu_e * beta_e - nu_w * beta_w)

    #top/north
    nu_t = D + nu_smag(u_2, w_2, dx, dz, i, j + 1) / Sh

    nu_n = harmonic_mean(nu_t, nu_center)
    beta_n = (beta[i, j + 1] - beta[i, j]) / dz

    #down/south
    nu_d = D + nu_smag(u_2, w_2, dx, dz, i, j - 1) / Sh

    nu_s = harmonic_mean(nu_d, nu_center)
    beta_s = (beta[i, j] - beta[i, j - 1]) / dz

    diff_z = (1.0 / dz) * (nu_n * beta_n - nu_s * beta_s)

    return diff_x + diff_z