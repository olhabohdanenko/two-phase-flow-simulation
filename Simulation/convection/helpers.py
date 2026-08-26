from numba import njit, prange
import numpy as np

@njit
def uw_central(u, w, i, j):
	return 0.25 * (u[i, j] + u[i, j + 1]) * (w[i, j] + w[i + 1, j])

@njit
def weno3_face(f_m1, f_0, f_p1):
    # Показники гладкості
    b0 = (f_0 - f_m1) ** 2
    b1 = (f_p1 - f_0) ** 2

    # Ідеальні ваги
    d0 = 1.0 / 3.0
    d1 = 2.0 / 3.0

    # eps для захисту від 0 на однорідних ділянках (beta = const)
    eps = 1e-10

    alpha0 = d0 / ((eps + b0) ** 2)
    alpha1 = d1 / ((eps + b1) ** 2)

    sum_alpha = alpha0 + alpha1

    if sum_alpha < 1e-15:
        return f_0

    w0 = alpha0 / sum_alpha
    w1 = alpha1 / sum_alpha

    # Реконструкції на шаблонах (Upwind / Central)
    q0 = 1.5 * f_0 - 0.5 * f_m1
    q1 = 0.5 * f_0 + 0.5 * f_p1

    return w0 * q0 + w1 * q1

@njit
def hlpa_face(b_U, b_C, b_D):
	denom = b_D - b_U
	if abs(denom) < 1e-14:
		return b_C

	tilde_b = (b_C - b_U) / denom
	if 0.0 <= tilde_b <= 1.0:
		tilde_f = tilde_b + 0.5 * tilde_b * (1.0 - tilde_b)
	else:
		tilde_f = tilde_b  # Upwind

	return b_U + tilde_f * denom

# TVD

@njit
def F_u_LO(u, dx, i, j):
	u_face = 0.5 * (u[i, j] + u[i + 1, j])
	return u_face * u[i, j] if u_face >= 0 else u_face * u[i + 1, j]

@njit
def F_u_HO(u, dx, i, j):
	return 0.5 * (u[i, j]**2 + u[i + 1, j]**2)

@njit
def r_u(u, dx, i, j):
	chis = u[i, j] - u[i - 1, j] if u[i, j] >= 0 else u[i + 1, j] - u[i + 2, j]
	znam = u[i + 1, j] - u[i, j] if u[i, j] >= 0 else u[i, j] - u[i + 1, j]

	return chis / znam if abs(znam) >= 1E-14 else 0.0

@njit
def F_w_LO(w, dz, i, j):
	w_face = 0.5 * (w[i, j] + w[i, j + 1])
	return w_face * w[i, j] if w_face >= 0 else w_face * w[i, j + 1]

@njit
def F_w_HO(w, dz, i, j):
	return 0.5 * (w[i, j]**2 + w[i, j + 1]**2)

@njit
def r_w(w, dz, i, j):
	chis = w[i, j] - w[i, j - 1] if w[i, j] >= 0 else w[i, j + 1] - w[i, j + 2]
	znam = w[i, j + 1] - w[i, j] if w[i, j] >= 0 else w[i, j] - w[i, j + 1]

	return chis / znam if abs(znam) >= 1E-14 else 0.0

@njit
def G_u_LO(u, w, dz, i, j):
	w_face = 0.5 * (w[i, j] + w[i + 1, j])
	return w_face * u[i, j] if w_face >= 0 else w_face * u[i, j + 1]

@njit
def G_u_HO(u, w, dz, i, j):
	w_face = 0.5 * (w[i, j] + w[i + 1, j])
	return 0.5 * w_face * (u[i, j] + u[i, j + 1])

@njit
def r_uw(u, w, dz, i, j):
	w_face = 0.5 * (w[i, j] + w[i + 1, j])

	chis = u[i, j] - u[i, j - 1] if w_face >= 0 else u[i, j + 1] - u[i, j + 2]
	znam = u[i, j + 1] - u[i, j] if w_face >= 0 else u[i, j] - u[i, j + 1]

	return chis / znam if abs(znam) >= 1E-14 else 0.0

@njit
def G_w_LO(u, w, dx, i, j):
	u_face = 0.5 * (u[i, j] + u[i, j + 1])
	return u_face * w[i, j] if u_face >= 0 else u_face * w[i + 1, j]

@njit
def G_w_HO(u, w, dx, i, j):
	u_face = 0.5 * (u[i, j] + u[i, j + 1])
	return 0.5 * u_face * (w[i, j] + w[i + 1, j])

@njit
def r_wu(u, w, dx, i, j):
	u_face = 0.5 * (u[i, j] + u[i, j + 1])
	chis = w[i, j] - w[i - 1, j] if u_face >= 0 else w[i + 1, j] - w[i + 2, j]
	znam = w[i + 1, j] - w[i, j] if u_face >= 0 else w[i, j] - w[i + 1, j]

	return chis / znam if abs(znam) >= 1E-14 else 0.0

@njit
def F_beta_x_LO(u_2, beta, i, j):
    U = u_2[i, j]
    return U * beta[i, j] if U >= 0 else U * beta[i + 1, j]

@njit
def F_beta_x_HO(u_2, beta, i, j):
    U = u_2[i, j]
    return U * 0.5 * (beta[i, j] + beta[i + 1, j])

@njit
def r_beta_x(u_2, beta, i, j):
    U = u_2[i, j]
    chis = beta[i, j] - beta[i - 1, j] if U >= 0 else beta[i + 1, j] - beta[i + 2, j]
    znam = beta[i + 1, j] - beta[i, j] if U >= 0 else beta[i, j] - beta[i + 1, j]
    return chis / znam if abs(znam) >= 1E-14 else 0.0

@njit
def F_beta_z_LO(w_2, beta, i, j):
    W = w_2[i, j]
    return W * beta[i, j] if W >= 0 else W * beta[i, j + 1]

@njit
def F_beta_z_HO(w_2, beta, i, j):
    W = w_2[i, j]
    return W * 0.5 * (beta[i, j] + beta[i, j + 1])

@njit
def r_beta_z(w_2, beta, i, j):
    W = w_2[i, j]
    chis = beta[i, j] - beta[i, j - 1] if W >= 0 else beta[i, j + 1] - beta[i, j + 2]
    znam = beta[i, j + 1] - beta[i, j] if W >= 0 else beta[i, j] - beta[i, j + 1]
    return chis / znam if abs(znam) >= 1E-14 else 0.0

@njit
def F_beta2_x_LO(u, u_2, beta, i, j):
    U_rel = u[i, j] - u_2[i, j]
    return U_rel * beta[i, j] if U_rel >= 0 else U_rel * beta[i + 1, j]

@njit
def F_beta2_x_HO(u, u_2, beta, i, j):
    U_rel = u[i, j] - u_2[i, j]
    return U_rel * 0.5 * (beta[i, j] + beta[i + 1, j])

@njit
def r_beta2_x(u, u_2, beta, i, j):
    U_rel = u[i, j] - u_2[i, j]
    chis = beta[i, j] - beta[i - 1, j] if U_rel >= 0 else beta[i + 1, j] - beta[i + 2, j]
    znam = beta[i + 1, j] - beta[i, j] if U_rel >= 0 else beta[i, j] - beta[i + 1, j]
    return chis / znam if abs(znam) >= 1E-14 else 0.0

@njit
def F_beta2_z_LO(w, w_2, beta, i, j):
    W_rel = w[i, j] - w_2[i, j]
    return W_rel * beta[i, j] if W_rel >= 0 else W_rel * beta[i, j + 1]

@njit
def F_beta2_z_HO(w, w_2, beta, i, j):
    W_rel = w[i, j] - w_2[i, j]
    return W_rel * 0.5 * (beta[i, j] + beta[i, j + 1])

@njit
def r_beta2_z(w, w_2, beta, i, j):
    W_rel = w[i, j] - w_2[i, j]
    chis = beta[i, j] - beta[i, j - 1] if W_rel >= 0 else beta[i, j + 1] - beta[i, j + 2]
    znam = beta[i, j + 1] - beta[i, j] if W_rel >= 0 else beta[i, j] - beta[i, j + 1]
    return chis / znam if abs(znam) >= 1E-14 else 0.0

