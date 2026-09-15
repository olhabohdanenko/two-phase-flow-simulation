from numba import njit
import numpy as np

import math

# Штучна в'язкість
@njit
def nu_x(u, nu, dx, Re, i, j):
	return nu

@njit
def nu_z(w, nu, dz, Re, i, j):
	return nu

@njit
def nu_x_(u, nu, dx, Re, i, j):
	return nu

@njit
def nu_z_(w, nu, dz, Re, i, j):
	return nu


@njit
def zeta_x(u, zeta, dx, Re, i, j):
	return zeta

@njit
def zeta_z(w, zeta, dz, Re, i, j):
	return zeta

@njit
def zeta_x_(u, zeta, dx, Re, i, j):
	return zeta

@njit
def zeta_z_(w, zeta, dz, Re, i, j):
	return zeta


@njit
def D_x(u, D, dx, Sh, i, j):
	return D

@njit
def D_z(w, D, dz, Sh, i, j):
	return D

# @njit
# def nu_x(u, nu, dx, Re, i, j):
# 	return nu + 0.5 * dx / Re * abs(u[i, j])

# @njit
# def nu_z(w, nu, dz, Re, i, j):
# 	return nu + 0.5 * dz / Re * abs(w[i, j])

# @njit
# def nu_x_(u, nu, dx, Re, i, j):
# 	return nu + 0.5 * dx / Re * abs(u[i, j] + u[i, j + 1])

# @njit
# def nu_z_(w, nu, dz, Re, i, j):
# 	return nu + 0.5 * dz / Re * abs(w[i, j] + w[i + 1, j])


# @njit
# def zeta_x(u, zeta, dx, Re, i, j):
# 	return zeta + 0.5 * dx / Re * abs(u[i, j])

# @njit
# def zeta_z(w, zeta, dz, Re, i, j):
# 	return zeta + 0.5 * dz / Re * abs(w[i, j])

# @njit
# def zeta_x_(u, zeta, dx, Re, i, j):
# 	return zeta + 0.5 * dx / Re * abs(u[i, j] + u[i, j + 1])

# @njit
# def zeta_z_(w, zeta, dz, Re, i, j):
# 	return zeta + 0.5 * dz / Re * abs(w[i, j] + w[i + 1, j])


# @njit
# def D_x(u, D, dx, Sh, i, j):
# 	return D + dx / Sh * abs(u[i, j])

# @njit
# def D_z(w, D, dz, Sh, i, j):
# 	return D + dz / Sh * abs(w[i, j])

# Гармонійне усереднення коефіцієнтів
@njit
def harmonic_mean(k1, k2):
	denom = k1 + k2
	if denom < 1E-14:
		return 0.0

	return (2.0 * k1 * k2) / denom

# Підсіткова в'язкість Смагоринського
@njit
def nu_smag(u, w, dx, dz, i, j, Cs = 0.15):
	du_dx = (u[i + 1, j] - u[i - 1, j]) / (2.0 * dx)
	dw_dz = (w[i, j + 1] - w[i, j - 1]) / (2.0 * dz)

	du_dz = (u[i, j + 1] - u[i, j - 1]) / (2.0 * dz)
	dw_dx = (w[i + 1, j] - w[i - 1, j]) / (2.0 * dx)

	S_xx = du_dx
	S_zz = dw_dz
	S_xz = 0.5 * (du_dz + dw_dx)

	S = math.sqrt(2.0 * S_xx**2 + 2.0 * S_zz**2 + 4.0 * S_xz**2)

	delta = math.sqrt(dx * dz)
	nu_t = (Cs * delta)**2 * S

	return nu_t

# Реалістична неньютонівська реологія Крігера-Догерті (Krieger-Dougherty)
@njit
def nu_krieger_dougherty(beta_val, nu_0, beta_max=0.63, intrinsic_visc=2.5):
	beta_clamped = max(0.0, min(beta_val, beta_max - 1e-4))

	exponent = intrinsic_visc * beta_max
	base = 1.0 - (beta_clamped / beta_max)

	return nu_0 * (base ** (-exponent))

# Дифузія 4-го порядку з датчиком гладкості (Jameson-style Dissipation)
@njit
def jst_dissipation_1d(u, dx, kappa2=0.5, kappa4=0.02):
	n = len(u)
	theta = np.zeros(n)
	d = np.zeros(n)

	for i in range(1, n - 1):
		num = abs(u[i + 1] - 2.0 * u[i] + u[i - 1])
		den = abs(u[i + 1]) + 2.0 * abs(u[i]) + abs(u[i - 1]) + 1e-12
		theta[i] = num / den

	for i in range(2, n - 3):
		e2 = kappa2 * max(theta[i], theta[i + 1])
		e4 = max(0.0, kappa4 - e2)
		
		diff1 = u[i + 1] - u[i]
		diff3 = u[i + 2] - 3.0 * u[i + 1] + 3.0 * u[i] - u[i - 1]
		
		d_face = e2 * diff1 - e4 * diff3
		d[i] += d_face / dx

	return d