from numba import njit, prange
import numpy as np

from .helpers import *
from ..physics.limiters import *

# Convection u-u_2 frac{\beta}{\partial x} + w-w_2 frac{\beta}{\partial z}
@njit
def conv_beta_2_center(u, w, u_2, w_2, beta, dx, dz, i, j):
	conv_x = ((u[i, j] - u_2[i, j]) * 0.5 * (beta[i + 1, j] + beta[i, j]) - (u[i - 1, j] - u_2[i - 1, j]) * 0.5 * (beta[i, j] + beta[i - 1, j])) / dx
	conv_z = ((w[i, j] - w_2[i, j]) * 0.5 * (beta[i, j + 1] + beta[i, j]) - (w[i, j - 1] - w_2[i, j - 1]) * 0.5 * (beta[i, j] + beta[i, j - 1])) / dz
	return conv_x + conv_z

@njit
def conv_beta_2_upwind(u, w, u_2, w_2, beta, dx, dz, i, j):
    du_e = u[i, j] - u_2[i, j]
    du_w = u[i - 1, j] - u_2[i - 1, j]
    dw_n = w[i, j] - w_2[i, j]
    dw_s = w[i, j - 1] - w_2[i, j - 1]

    beta_e = beta[i, j] if du_e > 0.0 else beta[i + 1, j]
    beta_w = beta[i - 1, j] if du_w > 0.0 else beta[i, j]

    beta_n = beta[i, j] if dw_n > 0.0 else beta[i, j + 1]
    beta_s = beta[i, j - 1] if dw_s > 0.0 else beta[i, j]

    conv_x = (du_e * beta_e - du_w * beta_w) / dx
    conv_z = (dw_n * beta_n - dw_s * beta_s) / dz

    return conv_x + conv_z

# TVD
@njit
def F_beta2_x(u, u_2, beta, i, j):
    r = r_beta2_x(u, u_2, beta, i, j)
    return F_beta2_x_LO(u, u_2, beta, i, j) + vanleer(r) * (F_beta2_x_HO(u, u_2, beta, i, j) - F_beta2_x_LO(u, u_2, beta, i, j))


@njit
def F_beta2_z(w, w_2, beta, i, j):
    r = r_beta2_z(w, w_2, beta, i, j)
    return F_beta2_z_LO(w, w_2, beta, i, j) + vanleer(r) * (F_beta2_z_HO(w, w_2, beta, i, j) - F_beta2_z_LO(w, w_2, beta, i, j))

@njit
def conv_beta_2_tvd(u, w, u_2, w_2, beta, dx, dz, i, j):
	conv_x = (F_beta2_x(u, u_2, beta, i, j) - F_beta2_x(u, u_2, beta, i - 1, j)) / dx
	conv_z = (F_beta2_z(w, w_2, beta, i, j) - F_beta2_z(w, w_2, beta, i, j - 1)) / dz

	return conv_x + conv_z

# QUICK
@njit
def F_beta2_x_QUICK(u, u_2, beta, i, j):
    U_rel = u[i, j] - u_2[i, j]
    if U_rel >= 0:
        beta_interp = (3.0 * beta[i + 1, j] + 6.0 * beta[i, j] - beta[i - 1, j]) / 8.0
    else:
        beta_interp = (3.0 * beta[i, j] + 6.0 * beta[i + 1, j] - beta[i + 2, j]) / 8.0
    return U_rel * beta_interp

@njit
def F_beta2_z_QUICK(w, w_2, beta, i, j):
    W_rel = w[i, j] - w_2[i, j]
    if W_rel >= 0:
        beta_interp = (3.0 * beta[i, j + 1] + 6.0 * beta[i, j] - beta[i, j - 1]) / 8.0
    else:
        beta_interp = (3.0 * beta[i, j] + 6.0 * beta[i, j + 1] - beta[i, j + 2]) / 8.0
    return W_rel * beta_interp

@njit
def conv_beta_2_quick(u, w, u_2, w_2, beta, dx, dz, i, j):
    conv_x = (F_beta2_x_QUICK(u, u_2, beta, i, j) - F_beta2_x_QUICK(u, u_2, beta, i - 1, j)) / dx
    conv_z = (F_beta2_z_QUICK(w, w_2, beta, i, j) - F_beta2_z_QUICK(w, w_2, beta, i, j - 1)) / dz
    return conv_x + conv_z

# SOUS
@njit
def F_beta2_x_SOUS(u, u_2, beta, i, j):
    U_rel = u[i, j] - u_2[i, j]
    if U_rel >= 0:
        beta_interp = 1.5 * beta[i, j] - 0.5 * beta[i - 1, j]
    else:
        beta_interp = 1.5 * beta[i + 1, j] - 0.5 * beta[i + 2, j]
    return U_rel * beta_interp

@njit
def F_beta2_z_SOUS(w, w_2, beta, i, j):
    W_rel = w[i, j] - w_2[i, j]
    if W_rel >= 0:
        beta_interp = 1.5 * beta[i, j] - 0.5 * beta[i, j - 1]
    else:
        beta_interp = 1.5 * beta[i, j + 1] - 0.5 * beta[i, j + 2]
    return W_rel * beta_interp

@njit
def conv_beta_2_sous(u, w, u_2, w_2, beta, dx, dz, i, j):
    conv_x = (F_beta2_x_SOUS(u, u_2, beta, i, j) - F_beta2_x_SOUS(u, u_2, beta, i - 1, j)) / dx
    conv_z = (F_beta2_z_SOUS(w, w_2, beta, i, j) - F_beta2_z_SOUS(w, w_2, beta, i, j - 1)) / dz
    return conv_x + conv_z

# WENO 3
@njit
def F_beta2_x_WENO3(u, u_2, beta, i, j):
    U_rel = u[i, j] - u_2[i, j]
    if U_rel >= 0:
        beta_interp = weno3_face(beta[i - 1, j], beta[i, j], beta[i + 1, j])
    else:
        beta_interp = weno3_face(beta[i + 2, j], beta[i + 1, j], beta[i, j])
    return U_rel * beta_interp

@njit
def F_beta2_z_WENO3(w, w_2, beta, i, j):
    W_rel = w[i, j] - w_2[i, j]
    if W_rel >= 0:
        beta_interp = weno3_face(beta[i, j - 1], beta[i, j], beta[i, j + 1])
    else:
        beta_interp = weno3_face(beta[i, j + 2], beta[i, j + 1], beta[i, j])
    return W_rel * beta_interp

@njit
def conv_beta_2_weno3(u, w, u_2, w_2, beta, dx, dz, i, j):
    conv_x = (F_beta2_x_WENO3(u, u_2, beta, i, j) - F_beta2_x_WENO3(u, u_2, beta, i - 1, j)) / dx
    conv_z = (F_beta2_z_WENO3(w, w_2, beta, i, j) - F_beta2_z_WENO3(w, w_2, beta, i, j - 1)) / dz
    return conv_x + conv_z

# CDS4
@njit
def F_beta2_x_CDS4(u, u_2, beta, i, j):
    U = u[i, j] - u_2[i, j]
    # Інтерполяція 4-го порядку на грань
    beta_interp = (-beta[i - 1, j] + 9.0 * beta[i, j] + 9.0 * beta[i + 1, j] - beta[i + 2, j]) / 16.0
    return U * beta_interp

@njit
def F_beta2_z_CDS4(w, w_2, beta, i, j):
    W = w[i, j] - w_2[i, j]
    # Інтерполяція 4-го порядку на грань
    beta_interp = (-beta[i, j - 1] + 9.0 * beta[i, j] + 9.0 * beta[i, j + 1] - beta[i, j + 2]) / 16.0
    return W * beta_interp

@njit
def conv_beta_2_cds4(u, w, u_2, w_2, beta, dx, dz, i, j):
    conv_x = (F_beta2_x_CDS4(u, u_2, beta, i, j) - F_beta2_x_CDS4(u, u_2, beta, i - 1, j)) / dx
    conv_z = (F_beta2_z_CDS4(w, w_2, beta, i, j) - F_beta2_z_CDS4(w, w_2, beta, i, j - 1)) / dz
    return conv_x + conv_z

# HLPA
@njit
def F_beta2_x_HLPA(u, u_2, beta, i, j):
    U = u[i, j] - u_2[i, j]
    if U >= 0:
        beta_interp = hlpa_face(beta[i - 1, j], beta[i, j], beta[i + 1, j])
    else:
        beta_interp = hlpa_face(beta[i + 2, j], beta[i + 1, j], beta[i, j])
    return U * beta_interp

@njit
def F_beta2_z_HLPA(w, w_2, beta, i, j):
	W = w[i, j] - w_2[i, j]
	if W >= 0:
		beta_interp = hlpa_face(beta[i, j - 1], beta[i, j], beta[i, j + 1])
	else:
		beta_interp = hlpa_face(beta[i, j + 2], beta[i, j + 1], beta[i, j])
	return W * beta_interp

@njit
def conv_beta_2_hlpa(u, w, u_2, w_2, beta, dx, dz, i, j):
    conv_x = (F_beta2_x_HLPA(u_2, beta, i, j) - F_beta2_x_HLPA(u_2, beta, i - 1, j)) / dx
    conv_z = (F_beta2_z_HLPA(w_2, beta, i, j) - F_beta2_z_HLPA(w_2, beta, i, j - 1)) / dz
    return conv_x + conv_z