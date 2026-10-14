from numba import njit
import numpy as np

import math

# Штучна в'язкість
@njit
def nu_x(u, nu, dx, Re, i, j):
	return nu + 0.5 * dx / Re * abs(u[i, j])

@njit
def nu_z(w, nu, dz, Re, i, j):
	return nu + 0.5 * dz / Re * abs(w[i, j])

@njit
def nu_x_(u, nu, dx, Re, i, j):
	return nu + 0.5 * dx / Re * abs(u[i, j] + u[i, j + 1])

@njit
def nu_z_(w, nu, dz, Re, i, j):
	return nu + 0.5 * dz / Re * abs(w[i, j] + w[i + 1, j])


@njit
def zeta_x(u, zeta, dx, Re, i, j):
	return zeta + 0.5 * dx / Re * abs(u[i, j])

@njit
def zeta_z(w, zeta, dz, Re, i, j):
	return zeta + 0.5 * dz / Re * abs(w[i, j])

@njit
def zeta_x_(u, zeta, dx, Re, i, j):
	return zeta + 0.5 * dx / Re * abs(u[i, j] + u[i, j + 1])

@njit
def zeta_z_(w, zeta, dz, Re, i, j):
	return zeta + 0.5 * dz / Re * abs(w[i, j] + w[i + 1, j])


@njit
def D_x(u, D, dx, Sh, i, j):
	return D + dx / Sh * abs(u[i, j])

@njit
def D_z(w, D, dz, Sh, i, j):
	return D + dz / Sh * abs(w[i, j])

# Гармонійне усереднення коефіцієнтів
@njit
def harmonic_mean(k1, k2):
	denom = k1 + k2
	if denom < 1E-14:
		return 0.0

	return (2.0 * k1 * k2) / denom

# Модель Смагоринського
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

@njit
def nu_krieger_dougherty(beta_val, nu_0, beta_max=0.63, intrinsic_visc=2.5):
	beta_clamped = max(0.0, min(beta_val, beta_max - 1e-4))

	exponent = intrinsic_visc * beta_max
	base = 1.0 - (beta_clamped / beta_max)

	return nu_0 * (base ** (-exponent))

@njit
def jst_dissipation_2d(u, dx, dz, kappa2=0.5, kappa4=0.02):
    nx, nz = u.shape
    d = np.zeros_like(u)

    theta_x = np.zeros_like(u)
    
    for i in range(1, nx - 1):
        for j in range(nz):
            num = abs(u[i + 1, j] - 2.0 * u[i, j] + u[i - 1, j])
            den = abs(u[i + 1, j]) + 2.0 * abs(u[i, j]) + abs(u[i - 1, j]) + 1e-12
            theta_x[i, j] = num / den
            
    for i in range(2, nx - 3):
        for j in range(nz):
            e2 = kappa2 * max(theta_x[i, j], theta_x[i + 1, j])
            e4 = max(0.0, kappa4 - e2)
            
            diff1 = u[i + 1, j] - u[i, j]
            diff3 = u[i + 2, j] - 3.0 * u[i + 1, j] + 3.0 * u[i, j] - u[i - 1, j]
            
            d_face_x = e2 * diff1 - e4 * diff3
            d[i, j] += d_face_x / dx

    theta_z = np.zeros_like(u)
    
    for i in range(nx):
        for j in range(1, nz - 1):
            num = abs(u[i, j + 1] - 2.0 * u[i, j] + u[i, j - 1])
            den = abs(u[i, j + 1]) + 2.0 * abs(u[i, j]) + abs(u[i, j - 1]) + 1e-12
            theta_z[i, j] = num / den
            
    for i in range(nx):
        for j in range(2, nz - 3):
            e2 = kappa2 * max(theta_z[i, j], theta_z[i, j + 1])
            e4 = max(0.0, kappa4 - e2)
            
            diff1 = u[i, j + 1] - u[i, j]
            diff3 = u[i, j + 2] - 3.0 * u[i, j + 1] + 3.0 * u[i, j] - u[i, j - 1]
            
            d_face_z = e2 * diff1 - e4 * diff3
            d[i, j] += d_face_z / dz

    return d

@njit
def compute_effective_viscosity(u, w, beta, dx, dz, i, j, physical_param):
    nu_0 = physical_param.nu
    
    nu_rheology = nu_krieger_dougherty(beta[i, j], nu_0)
    

    nu_turb = nu_smag(u, w, dx, dz, i, j, Cs=0.15)
    

    nu_eff = nu_rheology + nu_turb
    
    return nu_eff