from numba import njit, prange
import numpy as np

from .helpers import *
from ..physics.limiters import *

# beta
# Convection u_2 frac{\beta}{\partial x} + w_2 frac{\beta}{\partial z}
@njit
def conv_beta_central(u_2, w_2, beta, dx, dz, i, j):
	conv_x = (u_2[i, j] * 0.5 * (beta[i + 1, j] + beta[i, j]) - u_2[i - 1, j] * 0.5 * (beta[i, j] + beta[i - 1, j])) / dx
	conv_z = (w_2[i, j] * 0.5 * (beta[i, j + 1] + beta[i, j]) - w_2[i, j - 1] * 0.5 * (beta[i, j] + beta[i, j - 1])) / dz	
	return conv_x + conv_z

@njit
def conv_beta_upwind(u_2, w_2, beta, dx, dz, i, j):
	conv_x, conv_z = 0.0, 0.0
	
	if u_2[i, j] > 0:
		conv_x = u_2[i, j] * (beta[i, j] - beta[i - 1, j]) / dx
	else:
		conv_x = u_2[i, j] * (beta[i + 1, j] - beta[i, j]) / dx

	if w_2[i, j] > 0:
		conv_z = w_2[i, j] * (beta[i, j] - beta[i, j - 1]) / dz
	else:
		conv_z = w_2[i, j] * (beta[i, j + 1] - beta[i, j]) / dz
	return conv_x + conv_z

# TVD
@njit
def F_beta_x(u_2, beta, i, j):
    r = r_beta_x(u_2, beta, i, j)
    return F_beta_x_LO(u_2, beta, i, j) + vanleer(r) * (F_beta_x_HO(u_2, beta, i, j) - F_beta_x_LO(u_2, beta, i, j))


@njit
def F_beta_z(w_2, beta, i, j):
    r = r_beta_z(w_2, beta, i, j)
    return F_beta_z_LO(w_2, beta, i, j) + vanleer(r) * (F_beta_z_HO(w_2, beta, i, j) - F_beta_z_LO(w_2, beta, i, j))

@njit
def conv_beta_tvd(u_2, w_2, beta, dx, dz, i, j):
	conv_x = (F_beta_x(u_2, beta, i, j) - F_beta_x(u_2, beta, i - 1, j)) / dx
	conv_z = (F_beta_z(w_2, beta, i, j) - F_beta_z(w_2, beta, i, j - 1)) / dz

	return conv_x + conv_z

# QUICK
@njit
def F_beta_x_QUICK(u_2, beta, i, j):
    U = u_2[i, j]
    if U >= 0:
        beta_interp = (3.0 * beta[i + 1, j] + 6.0 * beta[i, j] - beta[i - 1, j]) / 8.0
    else:
        beta_interp = (3.0 * beta[i, j] + 6.0 * beta[i + 1, j] - beta[i + 2, j]) / 8.0
    return U * beta_interp

@njit
def F_beta_z_QUICK(w_2, beta, i, j):
    W = w_2[i, j]
    if W >= 0:
        beta_interp = (3.0 * beta[i, j + 1] + 6.0 * beta[i, j] - beta[i, j - 1]) / 8.0
    else:
        beta_interp = (3.0 * beta[i, j] + 6.0 * beta[i, j + 1] - beta[i, j + 2]) / 8.0
    return W * beta_interp

@njit
def conv_beta_quick(u_2, w_2, beta, dx, dz, i, j):
    conv_x = (F_beta_x_QUICK(u_2, beta, i, j) - F_beta_x_QUICK(u_2, beta, i - 1, j)) / dx
    conv_z = (F_beta_z_QUICK(w_2, beta, i, j) - F_beta_z_QUICK(w_2, beta, i, j - 1)) / dz
    return conv_x + conv_z

# SOUS
@njit
def F_beta_x_SOUS(u_2, beta, i, j):
    U = u_2[i, j]
    if U >= 0:
        beta_interp = 1.5 * beta[i, j] - 0.5 * beta[i - 1, j]
    else:
        beta_interp = 1.5 * beta[i + 1, j] - 0.5 * beta[i + 2, j]
    return U * beta_interp

@njit
def F_beta_z_SOUS(w_2, beta, i, j):
    W = w_2[i, j]
    if W >= 0:
        beta_interp = 1.5 * beta[i, j] - 0.5 * beta[i, j - 1]
    else:
        beta_interp = 1.5 * beta[i, j + 1] - 0.5 * beta[i, j + 2]
    return W * beta_interp

@njit
def conv_beta_sous(u_2, w_2, beta, dx, dz, i, j):
    conv_x = (F_beta_x_SOUS(u_2, beta, i, j) - F_beta_x_SOUS(u_2, beta, i - 1, j)) / dx
    conv_z = (F_beta_z_SOUS(w_2, beta, i, j) - F_beta_z_SOUS(w_2, beta, i, j - 1)) / dz
    return conv_x + conv_z

# WENO 3
@njit
def F_beta_x_WENO3(u_2, beta, i, j):
    U = u_2[i, j]
    if U >= 0:
        beta_interp = weno3_face(beta[i - 1, j], beta[i, j], beta[i + 1, j])
    else:
        beta_interp = weno3_face(beta[i + 2, j], beta[i + 1, j], beta[i, j])
    return U * beta_interp

@njit
def F_beta_z_WENO3(w_2, beta, i, j):
    W = w_2[i, j]
    if W >= 0:
        beta_interp = weno3_face(beta[i, j - 1], beta[i, j], beta[i, j + 1])
    else:
        beta_interp = weno3_face(beta[i, j + 2], beta[i, j + 1], beta[i, j])
    return W * beta_interp

@njit
def conv_beta_weno3(u_2, w_2, beta, dx, dz, i, j):
    conv_x = (F_beta_x_WENO3(u_2, beta, i, j) - F_beta_x_WENO3(u_2, beta, i - 1, j)) / dx
    conv_z = (F_beta_z_WENO3(w_2, beta, i, j) - F_beta_z_WENO3(w_2, beta, i, j - 1)) / dz
    return conv_x + conv_z

# CDS4
@njit
def F_beta_x_CDS4(u_2, beta, i, j):
    U = u_2[i, j]
    # Інтерполяція 4-го порядку на грань
    beta_interp = (-beta[i - 1, j] + 9.0 * beta[i, j] + 9.0 * beta[i + 1, j] - beta[i + 2, j]) / 16.0
    return U * beta_interp

@njit
def F_beta_z_CDS4(w_2, beta, i, j):
    W = w_2[i, j]
    # Інтерполяція 4-го порядку на грань
    beta_interp = (-beta[i, j - 1] + 9.0 * beta[i, j] + 9.0 * beta[i, j + 1] - beta[i, j + 2]) / 16.0
    return W * beta_interp

@njit
def conv_beta_cds4(u_2, w_2, beta, dx, dz, i, j):
    conv_x = (F_beta_x_CDS4(u_2, beta, i, j) - F_beta_x_CDS4(u_2, beta, i - 1, j)) / dx
    conv_z = (F_beta_z_CDS4(w_2, beta, i, j) - F_beta_z_CDS4(w_2, beta, i, j - 1)) / dz
    return conv_x + conv_z

# HLPA
@njit
def F_beta_x_HLPA(u_2, beta, i, j):
    U = u_2[i, j]
    if U >= 0:
        beta_interp = hlpa_face(beta[i - 1, j], beta[i, j], beta[i + 1, j])
    else:
        beta_interp = hlpa_face(beta[i + 2, j], beta[i + 1, j], beta[i, j])
    return U * beta_interp

@njit
def F_beta_z_HLPA(w_2, beta, i, j):
	W = w_2[i, j]
	if W >= 0:
		beta_interp = hlpa_face(beta[i, j - 1], beta[i, j], beta[i, j + 1])
	else:
		beta_interp = hlpa_face(beta[i, j + 2], beta[i, j + 1], beta[i, j])
	return W * beta_interp

@njit
def conv_beta_hlpa(u_2, w_2, beta, dx, dz, i, j):
    conv_x = (F_beta_x_HLPA(u_2, beta, i, j) - F_beta_x_HLPA(u_2, beta, i - 1, j)) / dx
    conv_z = (F_beta_z_HLPA(w_2, beta, i, j) - F_beta_z_HLPA(w_2, beta, i, j - 1)) / dz
    return conv_x + conv_z