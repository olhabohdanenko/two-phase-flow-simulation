import os
os.environ["NUMBA_NUM_THREADS"] = "4"

import numpy as np
from petsc4py import PETSc

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit, prange

import json
import h5py
from datetime import datetime

# Initialize
def Init_PETSc(dx, dz, M, N):
    da = PETSc.DMDA().create(dim=2, sizes=(M, N), dof=1, stencil_width=1)
    da.setUp()

    A = build_laplasian(da, dx, dz, M, N)

    b = da.createGlobalVec()
    x_vec = da.createGlobalVec()

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType('fgmres')

    pc = ksp.getPC()
    pc.setType('hypre')
    pc.setHYPREType('boomeramg')

    # ---- ГЛОБАЛЬНІ ОПЦІЇ ----
    opts = PETSc.Options()
    # opts.setValue('-ksp_rtol', '1e-10')
    # opts.setValue('-ksp_atol', '1e-12')
    opts.setValue('-ksp_rtol', '1e-6')
    opts.setValue('-ksp_atol', '1e-8')
    opts.setValue('-ksp_max_it', '1500')
    # -------------------------------------------------

    ksp.setFromOptions()
    ksp.setUp()

    return da, A, ksp, pc, b, x_vec

# Boundary Condition
@njit
def Apply_Boundary_Condition_Velosity(u, w, u_, w_, V_m, M, N):
    # Ліва
    # Вхід рідини
    u[0, :] = u[1, :] = V_m
    u_[0, :] = u_[1, :] = V_m
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
    w_[:, N - 2] = w_[:, N - 3] = 0.

@njit
def Apply_Boundary_Condition_Beta(beta, u_2, V_m, dx, dz, M, N):
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
	#beta[:, N - 1] = beta[:, N - 2] = 0.0

@njit
def Apply_Boundary_Condition_Inject(beta, u_2, V_m, dx, dz, M, N):
	X_inject = 0.01
	Y_inject = 0.01
	I_max_x = int(X_inject / dx) + 1
	I_max_y = int(Y_inject / dz) + 1
	for i in range(1, I_max_x):
		for j in range(1, I_max_y):
			beta[i, N - j] = 1E-4

@njit
def Apply_Boundary_Condition(u, w, u_, w_, V_m, dx, dz, M, N):
    # nozzle_positions = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45)
    nozzle_positions = (0.05, 0.15, 0.25, 0.35, 0.45)
    nozzle_r = 0.01

    for i in range(1, M - 1):
        x = (i - 0.5) * dx
        
        in_nozzle = False
        for pos in nozzle_positions:
            if abs(x - pos) <= nozzle_r:
                in_nozzle = True
                break
                
        if in_nozzle:
            w[i, 0] = w[i, 1] = V_m
            u[i, 0] = u[i, 1] = 0.0
            w_[i, 0] = w_[i, 1] = V_m
            u_[i, 0] = u_[i, 1] = 0.0
	
# Viscosity
@njit
def nu_x(u, nu, dx, Re, i, j):
	return nu + 0.5 * dx / Re * abs(u[i, j] + u[i + 1, j])

@njit
def nu_z(w, nu, dz, Re, i, j):
	return nu + 0.5 * dz / Re * abs(w[i, j] + w[i, j + 1])

@njit
def nu_x_(u, nu, dx, Re, i, j):
	return nu + 0.5 * dx / Re * abs(u[i, j] + u[i, j + 1])

@njit
def nu_z_(w, nu, dz, Re, i, j):
	return nu + 0.5 * dz / Re * abs(w[i, j] + w[i + 1, j])


@njit
def zeta_x(u, zeta, dx, Re, i, j):
	return zeta + 0.5 * dx / Re * abs(u[i, j] + u[i + 1, j])

@njit
def zeta_z(w, zeta, dz, Re, i, j):
	return zeta + 0.5 * dz / Re * abs(w[i, j] + w[i, j + 1])

@njit
def zeta_x_(u, zeta, dx, Re, i, j):
	return zeta + 0.5 * dx / Re * abs(u[i, j] + u[i, j + 1])

@njit
def zeta_z_(w, zeta, dz, Re, i, j):
	return zeta + 0.5 * dz / Re * abs(w[i, j] + w[i + 1, j])


@njit
def D_x(u, D, dx, Re, i, j):
	return D + dx / Re * abs(u[i, j])

@njit
def D_z(w, D, dz, Re, i, j):
	return D + dz / Re * abs(w[i, j])

# Limiters
@njit
def minmod(r):
	return max(0.0, min(1.0, r))

@njit
def vanalbada(r):
    return 0.0 if r <= 0.0 else (r * r + r) / (r * r + 1.0)

@njit
def koren(r):
    return max(0.0, min(2.0 * r, min((1.0 + 2.0 * r) / 3.0, 2.0)))

@njit
def ospre(r):
	return 0.0 if r <= 0.0 else (1.5 * r * (r + 1)) / (r * r + r + 1.0)

@njit
def vanleer(r):
	return 0.0 if r <= 0.0 else 2.0 * r / (1.0 + r)

@njit
def mc(r):  # Monotonized Central
    return max(0.0, min(min(2.0 * r, 0.5 * (1.0 + r)), 2.0))

@njit
def umist(r):  # Monotonized Central
    return max(0.0, min(min(2.0 * r, 0.25 + 0.75 * r), min(0.75 + 0.25 * r, 2.0)))

@njit
def superbee(r):
	return max(0.0, max(min(1.0, 2.0 * r), min(2.0, r)))

@njit
def smart(r):
	return max(0.0, min(min(2.0 * r, 0.75 * r + 0.25), 4.0))

@njit
def charm(r):
    """CHARM limiter (дуже добре зберігає екстремуми)"""
    return 0.0 if r <= 0.0 else (r * (3.0 * r + 1.0)) / ((r + 1.0) * (r + 1.0))

@njit
def quick(r):
    """QUICK limiter (TVD-обмежена версія класичної 3-го порядку)"""
    return max(0.0, min(min(2.0 * r, (3.0 + r) / 4.0), 2.0))

@njit
def sweby(r, beta=1.5):
    """Sweby limiter (узагальнений: при beta=1.0 це minmod, при beta=2.0 це superbee)"""
    if r <= 0.0:
        return 0.0
    return max(min(beta * r, 1.0), min(r, beta))

@njit
def vanalbada2(r):
    """Альтернативна/симетрична форма Van Albada"""
    return 0.0 if r <= 0.0 else (2.0 * r) / (r * r + 1.0)

@njit
def hcus(r):
    """HCUS (High-order CUS) limiter"""
    return 0.0 if r <= 0.0 else (1.5 * (r + abs(r))) / (r + 2.0)

@njit
def hquick(r):
    """HQUICK (High-order QUICK) limiter"""
    return 0.0 if r <= 0.0 else (2.0 * (r + abs(r))) / (r + 3.0)

@njit
def lax_wendroff(r):
    return 1.0

@njit
def beam_warming(r):
    return max(0.0, r)

@njit
def fromm(r):
    return max(0.0, 0.5 * (1.0 + r))

@njit
def ultrabee(r):
    """Ultra-Bee (Roe)
    Максимально компресивний лімітер.
    Ідеальний для чітких контактних розривів та ступенчатих функцій (мінімум чисельної дифузії).
    """
    return max(0.0, min(2.0 * r, 2.0))

@njit
def generalized_minmod(r, theta=1.5):
    """Generalized Minmod / Theta-limiter (1.0 <= theta <= 2.0)
    При theta=1.0 це стандартний Minmod, при theta=2.0 — Superbee.
    Дозволяє гнучко налаштовувати осциляції та дифузію.
    """
    if r <= 0.0:
        return 0.0
    return min(theta * r, min(0.5 * (1.0 + r), theta))

@njit
def gamma_scheme(r, beta=0.1):
    """Gamma scheme (Gaskell & Lau)
    Часто застосовується для переносу турбулентних величин.
    """
    if r <= 0.0:
        return 0.0
    return min(2.0 * r / beta, min(1.0 + r, 2.0))

@njit
def venkatakrishnan_1d(r):
    """1D адаптація лімітера Venkatakrishnan
    Дуже гладкий функціонал, забезпечує високу збіжність розрахунку до стаціонарного стану.
    """
    if r <= 0.0:
        return 0.0
    return (r * r + 2.0 * r) / (r * r + r + 2.0)

@njit
def cui(r):
    """CUI (Cubic Upwind Interpolation)
    3-й порядок точності для гладких рішень безTVD обмеження (лінійний).
    """
    return max(0.0, (3.0 + r) / 4.0)

@njit
def cubista(r):
    """CUBISTA (Chauhan et al.)
    Популярний у OpenFOAM для переносу скалярів та концентрованих величин.
    """
    if r <= 0.0:
        return 0.0
    return min(2.0 * r, min(0.25 + 0.75 * r, 2.0))
	

# Convection scheme
@njit
def uw_central(u, w, i, j):
	return 0.25 * (u[i, j] + u[i, j + 1]) * (w[i, j] + w[i + 1, j])

@njit
def weno3_face(f_m1, f_0, f_p1):
    # 1. Показники гладкості
    b0 = (f_0 - f_m1) ** 2
    b1 = (f_p1 - f_0) ** 2

    # 2. Ідеальні ваги
    d0 = 1.0 / 3.0
    d1 = 2.0 / 3.0

    # 3. Епсилон для захисту від 0 на однорідних ділянках (beta = const)
    eps = 1e-10  # КРИТИЧНО: має бути > 0!

    alpha0 = d0 / ((eps + b0) ** 2)
    alpha1 = d1 / ((eps + b1) ** 2)

    sum_alpha = alpha0 + alpha1

    # Захист від критичних випадків
    if sum_alpha < 1e-15:
        return f_0

    w0 = alpha0 / sum_alpha
    w1 = alpha1 / sum_alpha

    # 4. Реконструкції на шаблонах (Upwind / Central)
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

# Velosity
# Convection u frac{u}{\partial x}
@njit
def conv_u_central(u, dx, i, j):
	return (u[i + 1, j] * u[i + 1, j] - u[i - 1, j] * u[i - 1, j]) / 2.0 / dx

@njit
def conv_u_upwind(u, dx, i, j):
	u_transport_e = 0.5 * (u[i, j] + u[i + 1, j])
	u_e = u[i, j] if u_transport_e > 0 else u[i + 1, j]
	F_e = u_transport_e * u_e

	u_transport_w = 0.5 * (u[i - 1, j] + u[i, j])
	u_w = u[i - 1, j] if u_transport_w > 0 else u[i, j]
	F_w = u_transport_w * u_w

	return (F_e - F_w) / dx

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
def F_u(u, dx, i, j):
	r = r_u(u, dx, i, j)
	return F_u_LO(u, dx, i, j) + vanleer(r) * (F_u_HO(u, dx, i, j) - F_u_LO(u, dx, i, j))

@njit
def conv_u_tvd(u, dx, i, j):
	return (F_u(u, dx, i, j) - F_u(u, dx, i - 1, j)) / dx

# QUICK
@njit
def F_u_QUICK(u, i, j):
    u_face = 0.5 * (u[i, j] + u[i + 1, j])
    if u_face >= 0:
        u_interp = (3.0 * u[i + 1, j] + 6.0 * u[i, j] - u[i - 1, j]) / 8.0
    else:
        u_interp = (3.0 * u[i, j] + 6.0 * u[i + 1, j] - u[i + 2, j]) / 8.0
    return u_face * u_interp

@njit
def conv_u_quick(u, dx, i, j):
    return (F_u_QUICK(u, i, j) - F_u_QUICK(u, i - 1, j)) / dx

# SOUS
@njit
def F_u_SOUS(u, i, j):
    u_face = 0.5 * (u[i, j] + u[i + 1, j])
    if u_face >= 0:
        u_interp = 1.5 * u[i, j] - 0.5 * u[i - 1, j]
    else:
        u_interp = 1.5 * u[i + 1, j] - 0.5 * u[i + 2, j]
    return u_face * u_interp

@njit
def conv_u_sous(u, dx, i, j):
    return (F_u_SOUS(u, i, j) - F_u_SOUS(u, i - 1, j)) / dx

# WENO3
@njit
def F_u_WENO3(u, i, j):
	u_face = 0.5 * (u[i, j] + u[i + 1, j])
	if u_face >= 0:
		u_interp = weno3_face(u[i - 1, j], u[i, j], u[i + 1, j])
	else:
		u_interp = weno3_face(u[i + 2, j], u[i + 1, j], u[i, j])
	return u_face * u_interp

@njit
def conv_u_weno3(u, dx, i, j):
	return (F_u_WENO3(u, i, j) - F_u_WENO3(u, i - 1, j)) / dx

# Convection w frac{w}{\partial z}
@njit
def conv_w_central(w, dz, i, j):
	return (w[i, j + 1] * w[i, j + 1] - w[i, j - 1] * w[i, j - 1]) / 2.0 / dz
	
@njit
def conv_w_upwind(w, dz, i, j):
	w_transport_n = 0.5 * (w[i, j] + w[i, j + 1])
	w_n = w[i, j] if w_transport_n > 0 else w[i, j + 1]
	F_n = w_transport_n * w_n

	w_transport_s = 0.5 * (w[i, j - 1] + w[i, j])
	w_s = w[i, j - 1] if w_transport_s > 0 else w[i, j]
	F_s = w_transport_s * w_s

	return (F_n - F_s) / dz

# TVD
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
def F_w(w, dz, i, j):
	r = r_w(w, dz, i, j)
	return F_w_LO(w, dz, i, j) + vanleer(r) * (F_w_HO(w, dz, i, j) - F_w_LO(w, dz, i, j))

@njit
def conv_w_tvd(w, dz, i, j):
	return (F_w(w, dz, i, j) - F_w(w, dz, i, j - 1)) / dz

# QUICK
@njit
def F_w_QUICK(w, i, j):
    w_face = 0.5 * (w[i, j] + w[i, j + 1])
    if w_face >= 0:
        # 2 точки знизу (j-1, j), 1 зверху (j+1)
        w_interp = (3.0 * w[i, j + 1] + 6.0 * w[i, j] - w[i, j - 1]) / 8.0
    else:
        # 1 точка знизу (j), 2 зверху (j+1, j+2)
        w_interp = (3.0 * w[i, j] + 6.0 * w[i, j + 1] - w[i, j + 2]) / 8.0
    return w_face * w_interp

@njit
def conv_w_quick(w, dz, i, j):
    return (F_w_QUICK(w, i, j) - F_w_QUICK(w, i, j - 1)) / dz

# SOUS
@njit
def F_w_SOUS(w, i, j):
    w_face = 0.5 * (w[i, j] + w[i, j + 1])
    if w_face >= 0:
        w_interp = 1.5 * w[i, j] - 0.5 * w[i, j - 1]
    else:
        w_interp = 1.5 * w[i, j + 1] - 0.5 * w[i, j + 2]
    return w_face * w_interp

@njit
def conv_w_sous(w, dz, i, j):
    return (F_w_SOUS(w, i, j) - F_w_SOUS(w, i, j - 1)) / dz

# WENO 3
@njit
def F_w_WENO3(w, i, j):
    w_face = 0.5 * (w[i, j] + w[i, j + 1])
    if w_face >= 0:
        w_interp = weno3_face(w[i, j - 1], w[i, j], w[i, j + 1])
    else:
        w_interp = weno3_face(w[i, j + 2], w[i, j + 1], w[i, j])
    return w_face * w_interp

@njit
def conv_w_weno3(w, dz, i, j):
    return (F_w_WENO3(w, i, j) - F_w_WENO3(w, i, j - 1)) / dz

# Convection w frac{u}{\partial z}
@njit
def conv_uw_central(u, w, dz, i, j):
	return (uw_central(u, w, i, j) - uw_central(u, w, i, j - 1)) / dz

@njit
def conv_uw_upwind(u, w, dz, i, j):
	w_n = 0.5 * (w[i, j] + w[i + 1, j])
	w_s = 0.5 * (w[i, j - 1] + w[i + 1, j - 1])

	u_n = u[i, j] if w_n > 0 else u[i, j + 1]
	F_n = u_n * w_n

	u_s = u[i, j - 1] if w_s > 0 else u[i, j]
	F_s = u_s * w_s

	return (F_n - F_s) / dz

# TVD
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
def G_u(u, w, dz, i, j):
	r = r_uw(u, w, dz, i, j)
	return G_u_LO(u, w, dz, i, j) + vanleer(r) * (G_u_HO(u, w, dz, i, j) - G_u_LO(u, w, dz, i, j))

@njit
def conv_uw_tvd(u, w, dz, i, j):
	return (G_u(u, w, dz, i, j) - G_u(u, w, dz, i, j - 1)) / dz

# QUICK
@njit
def G_u_QUICK(u, w, dz, i, j):
    w_face = 0.5 * (w[i, j] + w[i + 1, j])
    if w_face >= 0:
        # Інтерполяція u по вертикалі (індекс j)
        u_interp = (3.0 * u[i, j + 1] + 6.0 * u[i, j] - u[i, j - 1]) / 8.0
    else:
        u_interp = (3.0 * u[i, j] + 6.0 * u[i, j + 1] - u[i, j + 2]) / 8.0
    return w_face * u_interp

@njit
def conv_uw_quick(u, w, dz, i, j):
    return (G_u_QUICK(u, w, dz, i, j) - G_u_QUICK(u, w, dz, i, j - 1)) / dz

# SOUS
@njit
def G_u_SOUS(u, w, dz, i, j):
    w_face = 0.5 * (w[i, j] + w[i + 1, j])
    if w_face >= 0:
        u_interp = 1.5 * u[i, j] - 0.5 * u[i, j - 1]
    else:
        u_interp = 1.5 * u[i, j + 1] - 0.5 * u[i, j + 2]
    return w_face * u_interp

@njit
def conv_uw_sous(u, w, dz, i, j):
    return (G_u_SOUS(u, w, dz, i, j) - G_u_SOUS(u, w, dz, i, j - 1)) / dz

# WENO 3
@njit
def G_u_WENO3(u, w, dz, i, j):
    w_face = 0.5 * (w[i, j] + w[i + 1, j])
    if w_face >= 0:
        u_interp = weno3_face(u[i, j - 1], u[i, j], u[i, j + 1])
    else:
        u_interp = weno3_face(u[i, j + 2], u[i, j + 1], u[i, j])
    return w_face * u_interp

@njit
def conv_uw_weno3(u, w, dz, i, j):
    return (G_u_WENO3(u, w, dz, i, j) - G_u_WENO3(u, w, dz, i, j - 1)) / dz

# Convection u frac{w}{\partial x}
@njit
def conv_wu_central(u, w, dx, i, j):
	return (uw_central(u, w, i, j) - uw_central(u, w, i - 1, j)) / dx
	
@njit
def conv_wu_upwind(u, w, dx, i, j):
	u_e = 0.5 * (u[i, j] + u[i, j + 1])
	u_w = 0.5 * (u[i - 1, j] + u[i - 1, j + 1])

	w_e = w[i, j] if u_e > 0 else w[i + 1, j]
	F_e = u_e * w_e

	w_w = w[i - 1, j] if u_w > 0 else w[i, j]
	F_w = u_w * w_w

	return (F_e - F_w) / dx

# TVD
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
def G_w(u, w, dx, i, j):
	r = r_wu(u, w, dx, i, j)
	return G_w_LO(u, w, dx, i, j) + vanleer(r) * (G_w_HO(u, w, dx, i, j) - G_w_LO(u, w, dx, i, j))

@njit
def conv_wu_tvd(u, w, dx, i, j):
	return (G_w(u, w, dx, i, j) - G_w(u, w, dx, i - 1, j)) / dx

# QUICK
@njit
def G_w_QUICK(u, w, dx, i, j):
    u_face = 0.5 * (u[i, j] + u[i, j + 1])
    if u_face >= 0:
        # Інтерполяція w по горизонталі (по індексу i)
        w_interp = (3.0 * w[i + 1, j] + 6.0 * w[i, j] - w[i - 1, j]) / 8.0
    else:
        w_interp = (3.0 * w[i, j] + 6.0 * w[i + 1, j] - w[i + 2, j]) / 8.0
    return u_face * w_interp

@njit
def conv_wu_quick(u, w, dx, i, j):
    return (G_w_QUICK(u, w, dx, i, j) - G_w_QUICK(u, w, dx, i - 1, j)) / dx

# SOUS
@njit
def G_w_SOUS(u, w, dx, i, j):
    u_face = 0.5 * (u[i, j] + u[i, j + 1])
    if u_face >= 0:
        w_interp = 1.5 * w[i, j] - 0.5 * w[i - 1, j]
    else:
        w_interp = 1.5 * w[i + 1, j] - 0.5 * w[i + 2, j]
    return u_face * w_interp

@njit
def conv_wu_sous(u, w, dx, i, j):
    return (G_w_SOUS(u, w, dx, i, j) - G_w_SOUS(u, w, dx, i - 1, j)) / dx

# WENO 3
@njit
def G_w_WENO3(u, w, dx, i, j):
    u_face = 0.5 * (u[i, j] + u[i, j + 1])
    if u_face >= 0:
        w_interp = weno3_face(w[i - 1, j], w[i, j], w[i + 1, j])
    else:
        w_interp = weno3_face(w[i + 2, j], w[i + 1, j], w[i, j])
    return u_face * w_interp

@njit
def conv_wu_weno3(u, w, dx, i, j):
    return (G_w_WENO3(u, w, dx, i, j) - G_w_WENO3(u, w, dx, i - 1, j)) / dx

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
def F_beta_x_LO(u_2, beta, i, j):
    U = u_2[i, j] # Швидкість на правій грані
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
def F_beta_x(u_2, beta, i, j):
    r = r_beta_x(u_2, beta, i, j)
    return F_beta_x_LO(u_2, beta, i, j) + vanleer(r) * (F_beta_x_HO(u_2, beta, i, j) - F_beta_x_LO(u_2, beta, i, j))

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
    # Інтерполяція 4-го порядку на грань (не залежить від напрямку потоку)
    beta_interp = (-beta[i - 1, j] + 9.0 * beta[i, j] + 9.0 * beta[i + 1, j] - beta[i + 2, j]) / 16.0
    return U * beta_interp

@njit
def F_beta_z_CDS4(w_2, beta, i, j):
    W = w_2[i, j]
    # Інтерполяція 4-го порядку на грань (не залежить від напрямку потоку)
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

# Convection u-u_2 frac{\beta}{\partial x} + w-w_2 frac{\beta}{\partial z}
@njit
def conv_beta_2_center(u, w, u_2, w_2, beta, dx, dz, i, j):
	conv_x = ((u[i, j] - u_2[i, j]) * 0.5 * (beta[i + 1, j] + beta[i, j]) - (u[i - 1, j] - u_2[i - 1, j]) * 0.5 * (beta[i, j] + beta[i - 1, j])) / dx
	conv_z = ((w[i, j] - w_2[i, j]) * 0.5 * (beta[i, j + 1] + beta[i, j]) - (w[i, j - 1] - w_2[i, j - 1]) * 0.5 * (beta[i, j] + beta[i, j - 1])) / dz
	return conv_x + conv_z

@njit
def conv_beta_2_upwind(u, w, u_2, w_2, beta, dx, dz, i, j):
    # 1. Відносні швидкості на гранях комірки (i, j)
    du_e = u[i, j] - u_2[i, j]
    du_w = u[i - 1, j] - u_2[i - 1, j]
    dw_n = w[i, j] - w_2[i, j]
    dw_s = w[i, j - 1] - w_2[i, j - 1]

    # 2. Upwind-значення beta на східній (e) та західній (w) гранях
    beta_e = beta[i, j] if du_e > 0.0 else beta[i + 1, j]
    beta_w = beta[i - 1, j] if du_w > 0.0 else beta[i, j]

    # 3. Upwind-значення beta на північній (n) та південній (s) гранях
    beta_n = beta[i, j] if dw_n > 0.0 else beta[i, j + 1]
    beta_s = beta[i, j - 1] if dw_s > 0.0 else beta[i, j]

    # 4. Консервативна дивергенція відносного потоку
    conv_x = (du_e * beta_e - du_w * beta_w) / dx
    conv_z = (dw_n * beta_n - dw_s * beta_s) / dz

    return conv_x + conv_z

# TVD
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
def F_beta2_x(u, u_2, beta, i, j):
    r = r_beta2_x(u, u_2, beta, i, j)
    return F_beta2_x_LO(u, u_2, beta, i, j) + vanleer(r) * (F_beta2_x_HO(u, u_2, beta, i, j) - F_beta2_x_LO(u, u_2, beta, i, j))

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

# QUICK
@njit
def F_beta2_x_CDS4(u, u_2, beta, i, j):
    U = u[i, j] - u_2[i, j]
    # Інтерполяція 4-го порядку на грань (не залежить від напрямку потоку)
    beta_interp = (-beta[i - 1, j] + 9.0 * beta[i, j] + 9.0 * beta[i + 1, j] - beta[i + 2, j]) / 16.0
    return U * beta_interp

@njit
def F_beta2_z_CDS4(w, w_2, beta, i, j):
    W = w[i, j] - w_2[i, j]
    # Інтерполяція 4-го порядку на грань (не залежить від напрямку потоку)
    beta_interp = (-beta[i, j - 1] + 9.0 * beta[i, j] + 9.0 * beta[i, j + 1] - beta[i, j + 2]) / 16.0
    return W * beta_interp

@njit
def conv_beta_2_cds4(u, w, u_2, w_2, beta, dx, dz, i, j):
    conv_x = (F_beta2_x_CDS4(u_2, beta, i, j) - F_beta2_x_CDS4(u_2, beta, i - 1, j)) / dx
    conv_z = (F_beta2_z_CDS4(w_2, beta, i, j) - F_beta2_z_CDS4(w_2, beta, i, j - 1)) / dz
    return conv_x + conv_z


@njit
def divergence(u, w, dx, dz, i, j):
	return (u[i, j] - u[i - 1, j]) / dx + (w[i, j] - w[i, j - 1]) / dz

# Diffusion
@njit
def diff_u(u, w, beta, nu, zeta, A_12, y, dx, dz, Re, i, j):
	factor = A_12 + y

	#right/east
	beta_e = beta[i + 1, j]
	nu_e = nu_x(u, nu, dx, Re, i, j)
	zeta_e = zeta_x(u, zeta, dx, Re, i, j)

	F_e_nu = nu_e * (1 + beta_e * factor)
	F_e_zeta = zeta_e + nu_e / 3.0 + beta_e * (zeta_e * y + nu_e / 3.0 * factor)

	u_e = (u[i + 1, j] - u[i, j]) / dx
	Div_e = divergence(u, w, dx, dz, i + 1, j)
	
	#left/west
	beta_w = beta[i, j]
	nu_w = nu_x(u, nu, dx, Re, i - 1, j)
	zeta_w = zeta_x(u, zeta, dx, Re, i - 1, j)

	F_w_nu = nu_w * (1 + beta_w * factor)
	F_w_zeta = zeta_w + nu_w / 3.0 + beta_w * (zeta_w * y + nu_w / 3.0 * factor)

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
def diff_w(u, w, beta, nu, zeta, A_12, y, dx, dz, Re, i, j):
	factor = A_12 + y

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
	F_n_zeta = zeta_n + nu_n / 3.0 + beta_n * (zeta_n * y + nu_n / 3.0 * factor)

	w_n = (w[i, j + 1] - w[i, j]) / dz
	Div_n = divergence(u, w, dx, dz, i, j + 1)
	
	#down/south
	beta_s = beta[i, j]
	nu_s = nu_z(w, nu, dz, Re, i, j - 1)
	zeta_s = zeta_z(w, zeta, dz, Re, i, j - 1)

	F_s_nu = nu_s * (1 + beta_s * factor)
	F_s_zeta = zeta_s + nu_s / 3.0 + beta_s * (zeta_s * y + nu_s / 3.0 * factor)

	w_s = (w[i, j] - w[i, j - 1]) / dz
	Div_s = divergence(u, w, dx, dz, i, j)
	

	diff_z = (1.0 / dz) * (F_n_nu * w_n - F_s_nu * w_s)
	diff_z += (1.0 / dz) * (F_n_zeta * Div_n - F_s_zeta * Div_s)
	
	return diff_x + diff_z

@njit
def diff_beta(u_2, w_2, beta, D, dx, dz, Re, i, j):

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

# Force
@njit
def gravity_force(beta, y, g, i, j):
	return 0.5 * (beta[i, j] + beta[i, j + 1]) * y * g

# SLAE
@njit
def build_rhs(u, w, u_, w_, u_2, w_2, beta, D, x, tau, dx, dz, Re, M, N):
	rhs = np.zeros((M, N))
	y = 1.0 - x 

	for i in range(2, M - 2):
		for j in range(2, N - 2):
			div_beta_w2 = - beta[i, j] * divergence(u_2, w_2, dx, dz, i, j) + conv_beta_2_tvd(u, w, u_2, w_2, beta, dx, dz, i, j)
			
			target_div = divergence(u_, w_, dx, dz, i, j) - y / (1.0 - y * beta[i, j]) * (div_beta_w2)
			
			rhs[i, j] = target_div / tau
			
	return rhs

def build_laplasian(da, dx, dz, M, N):
	A = da.createMatrix()
	A.setType('aij')
	(xs, xe), (ys, ye) = da.getRanges()

	row = PETSc.Mat.Stencil()
	col = PETSc.Mat.Stencil()

	dx2, dz2 = dx ** 2, dz ** 2

	for i in range(xs, xe):
		for j in range(ys, ye):
			row.i = i
			row.j = j

			# 1. Екстремальний фіктивний шар (Найбільш зовнішній) - просто глушимо
			if i == 0 or i == M - 1 or j == 0 or j == N - 1:
				A.setValueStencil(row, row, 1.0)
				continue

			# 2. ПЕРШИЙ внутрішній фіктивний шар - ТУТ ставимо ГУ для тиску
			# Права грань (Outlet) -> p = 0 (Діріхле)
			elif i == M - 2:
				A.setValueStencil(row, row, 1.0)
				continue
			# Ліва грань (Inlet) -> Нейман dp/dx = 0
			elif i == 1:
				A.setValueStencil(row, row, -1.0 / dx2)
				col.i = i + 1;
				col.j = j
				A.setValueStencil(row, col, 1.0 / dx2)
				continue
			# Нижня грань -> Нейман dp/dz = 0
			elif j == 1:
				A.setValueStencil(row, row, -1.0 / dz2)
				col.i = i;
				col.j = j + 1
				A.setValueStencil(row, col, 1.0 / dz2)
				continue
			# Верхня грань -> Нейман dp/dz = 0
			elif j == N - 2:
				A.setValueStencil(row, row, -1.0 / dz2)
				col.i = i;
				col.j = j - 1
				A.setValueStencil(row, col, 1.0 / dz2)
				continue

			# 3. ВНУТРІШНІ ТОЧКИ (від 2 до M-3) - звичайний Лапласіан
			else:
				A.setValueStencil(row, row, 2.0 / dx2 + 2.0 / dz2)
				col.i = i - 1;
				col.j = j;
				A.setValueStencil(row, col, -1.0 / dx2)
				col.i = i + 1;
				col.j = j;
				A.setValueStencil(row, col, -1.0 / dx2)
				col.i = i;
				col.j = j - 1;
				A.setValueStencil(row, col, -1.0 / dz2)
				col.i = i;
				col.j = j + 1;
				A.setValueStencil(row, col, -1.0 / dz2)

	A.assemblyBegin()
	A.assemblyEnd()
	return A
    
# Steps
@njit(parallel=True)
def Step1(u, u_, w, w_, p, beta, nu, zeta, g, A_12, y, dx, dz, tau, Re, M, N):
	for i in prange(2, M - 2):
		for j in range(2, N - 2):
			u_[i, j] = u[i, j] + tau * (-conv_u_weno3(u, dx, i, j) - conv_uw_weno3(u, w, dz, i, j) + diff_u(u, w, beta, nu, zeta, A_12, y, dx, dz, Re, i, j))
		
	for i in prange(2, M - 2):
		for j in range(2, N - 2):
			w_[i, j] = w[i, j] + tau * (-conv_w_weno3(w, dz, i, j) - conv_wu_weno3(u, w, dx, i, j) + diff_w(u, w, beta, nu, zeta, A_12, y, dx, dz, Re, i, j) - gravity_force(beta, y, g, i, j))

def Step2(u, w, u_, w_, u_2, w_2, beta, p, da, A, ksp, pc, b, x_vec, D, x_coord, dx, dz, tau, Re, M, N):
    rhs = -build_rhs(u, w, u_, w_, u_2, w_2, beta, D, x_coord, tau, dx, dz, Re, M, N)

    with da.getVecArray(b) as arr:
        arr[:, :] = 0.0 # Обнуляємо весь масив
        # Заповнюємо тільки фізичну область!
        arr[2:-2, 2:-2] = rhs[2:-2, 2:-2]

    ksp.solve(b, x_vec)

    with da.getVecArray(x_vec) as arr_x:
        p[:, :] = arr_x[:, :]

@njit(parallel=True)
def Step3(u, u_, w, w_, p, dx, dz, tau, M, N):
	for i in prange(2, M - 2):
		for j in range(2, N - 2):
			u[i, j] = u_[i, j] - tau * (p[i + 1, j] - p[i, j]) / dx
			w[i, j] = w_[i, j] - tau * (p[i, j + 1] - p[i, j]) / dz

@njit(parallel=True)
def compute_velosity_1(u, w, u_1, w_1, u_2, w_2, beta, rho_1, rho_2, M, N):
	x = rho_2 / rho_1
	for i in prange (M - 2):
		for j in range (N - 2):
			u_1[i, j] = (u[i, j] - 0.5 * (beta[i, j] + beta[i + 1, j]) * x * u_2[i, j]) / (1 - 0.5 * (beta[i, j] + beta[i + 1, j]) * x)
			w_1[i, j] = (w[i, j] - 0.5 * (beta[i, j] + beta[i, j + 1]) * x * w_2[i, j]) / (1 - 0.5 * (beta[i, j] + beta[i, j + 1]) * x)

@njit(parallel=True)
def Step4(u_1, w_1, u_2, w_2, beta, nu, D, d, C_A, C_1D, C_2D, x, y, g, tau, Re, dx, dz, M, N):
	inertia = C_A + x
	stokes_factor = C_1D / d**2
	newtons_factor = C_2D / d
	for i in prange(2, M - 2):
		for j in range(2, N - 2):
			v21_x = u_2[i, j] - u_1[i, j]
			v21_z = w_2[i, j] - w_1[i, j]
			v21_mod = (v21_x**2 + v21_z**2)**0.5 + 1E-12
			
			f_D_x = -(stokes_factor * nu_x(u_1, nu, dx, Re, i, j) * v21_x + newtons_factor * v21_mod * v21_x)
			f_D_z = -(stokes_factor * nu_z(w_1, nu, dz, Re, i, j) * v21_z + newtons_factor * v21_mod * v21_z)
			
			u_2[i, j] = u_2[i, j] + (tau / inertia) * f_D_x
			w_2[i, j] = w_2[i, j] + (tau / inertia) * (-y * g + f_D_z)
			
			beta_x = 0.5 * (beta[i + 1, j] + beta[i, j])
			beta_z = 0.5 * (beta[i, j + 1] + beta[i, j])
			
			beta_x_safe = max(beta_x, 1E-12)
			beta_z_safe = max(beta_z, 1E-12)
			
			grad_beta_x = (beta[i + 1, j] - beta[i, j]) / dx
			grad_beta_z = (beta[i, j + 1] - beta[i, j]) / dz
			
			w_2_dx = -1.0 / beta_x_safe * D_x(u_2, D, dx, Re, i, j) * grad_beta_x
			w_2_dz = -1.0 / beta_z_safe * D_z(w_2, D, dz, Re, i, j) * grad_beta_z
			
			u_2[i, j] += w_2_dx
			w_2[i, j] += w_2_dz

@njit(parallel=True)
def Step4_Fully_Implicit(u, w, u_1, w_1, u_2, w_2, beta, nu, D, d, C_A, C_1D, C_2D, x, y, g, tau, Re, dx, dz, M, N):
	inertia = C_A + x
	stokes_factor = C_1D / d**2
	newtons_factor = C_2D / d

	for i in prange(2, M - 2):
		for j in range(2, N - 2):
			# 1. Рахуємо відносні швидкості та їх модуль	
			v21_x = u_2[i, j] - u_1[i, j]
			v21_z = w_2[i, j] - w_1[i, j]
			v21_mod = (v21_x**2 + v21_z**2)**0.5 + 1E-12
			
			# 2. Локальні базові коефіцієнти взаємодії
			K_x = stokes_factor * nu_x(u_1, nu, dx, Re, i, j) + newtons_factor * v21_mod
			K_z = stokes_factor * nu_z(w_1, nu, dz, Re, i, j) + newtons_factor * v21_mod
			
			# Безрозмірні комплекси релаксації імпульсу
			alpha_x = (tau / inertia) * K_x
			alpha_z = (tau / inertia) * K_z
			gamma_x = tau * K_x  
			gamma_z = tau * K_z

			# 3. Рахуємо градієнти та дифузійні поправки
			beta_x = 0.5 * (beta[i + 1, j] + beta[i, j])
			beta_z = 0.5 * (beta[i, j + 1] + beta[i, j])
			
			beta_x_safe = max(beta_x, 1E-12)
			beta_z_safe = max(beta_z, 1E-12)
			
			grad_beta_x = (beta[i + 1, j] - beta[i, j]) / dx
			grad_beta_z = (beta[i, j + 1] - beta[i, j]) / dz
			
			w_2_dx = -1.0 / beta_x_safe * D_x(u_2, D, dx, Re, i, j) * grad_beta_x
			w_2_dz = -1.0 / beta_z_safe * D_z(w_2, D, dz, Re, i, j) * grad_beta_z

			# 4. Формуємо праві частини (explicit праве крило для системи)
			# Для рідини базове значення — це її поточна швидкість
			rhs_u1 = u_1[i, j]
			rhs_w1 = w_1[i, j]
			
			# Для частинок додаємо зовнішні сили (гравітацію та дифузійні поправки)
			rhs_u2 = u_2[i, j] + w_2_dx
			rhs_w2 = w_2[i, j] + (tau / inertia) * (-y * g) + w_2_dz

			# 5. Точний аналітичний розв'язок системи 2х2 (Крамер)
			denom_x = (1.0 + gamma_x) * (1.0 + alpha_x) - gamma_x * alpha_x
			denom_z = (1.0 + gamma_z) * (1.0 + alpha_z) - gamma_z * alpha_z
			
			if denom_x < 1E-14: denom_x = 1E-14
			if denom_z < 1E-14: denom_z = 1E-14

			# Оновлюємо масиви прямо на місці
			u_1[i, j] = ((1.0 + alpha_x) * rhs_u1 + gamma_x * rhs_u2) / denom_x
			u_2[i, j] = (alpha_x * rhs_u1 + (1.0 + gamma_x) * rhs_u2) / denom_x
			
			w_1[i, j] = ((1.0 + alpha_z) * rhs_w1 + gamma_z * rhs_w2) / denom_z
			w_2[i, j] = (alpha_z * rhs_w1 + (1.0 + gamma_z) * rhs_w2) / denom_z

@njit(parallel=True)
def Step5(u_2, w_2, beta, beta_, D, tau, Re, dx, dz, M, N):
	for i in prange(2, M - 2):
		for j in range(2, N - 2):
			beta_[i, j] = beta[i, j] + tau * (
						-conv_beta_tvd(u_2, w_2, beta, dx, dz, i, j) + diff_beta(u_2, w_2, beta, D, dx, dz, Re, i, j))

	for i in prange(2, M - 2):
		for j in range(2, N - 2):
			beta_[i, j] = max(0.0, min(1.0, beta_[i, j]))

@njit(parallel=True)
def StepEnd(u, w, u_2, w_2, beta, Div, D, dx, dz, x, Re, M, N):
	y = 1.0 - x

	for i in prange(2, M - 2):
		for j in range(2, N - 2):
			div_beta_w2 = - beta[i, j] * divergence(u_2, w_2, dx, dz, i, j) + conv_beta_2_tvd(u, w, u_2, w_2, beta, dx, dz, i, j)

			Div[i, j] = divergence(u, w, dx, dz, i, j) - y / (1.0 - y * beta[i, j]) * (div_beta_w2)

def Step_Euller(u, u_, u_1, u_2, w, w_, w_1, w_2, p, Div, beta, beta_, nu, zeta, V_m, V_m_1, g, A_12, rho_1, rho_2, x, y, D, Re, d, C_A, C_1D, C_2D, dx, dz, tau, da, A, ksp, pc, b, x_vec, M, N, time_step, Iter):
    Apply_Boundary_Condition_Velosity(u, w, u_, w_, V_m, M, N)
    Apply_Boundary_Condition(u, w, u_, w_, V_m_1, dx, dz, M, N)

    Step1(u, u_, w, w_, p, beta, nu, zeta, g, A_12, y, dx, dz, tau, Re, M, N)

    Step2(u, w, u_, w_, u_2, w_2, beta, p, da, A, ksp, pc, b, x_vec, D, x, dx, dz, tau, Re, M, N)
    Step3(u, u_, w, w_, p, dx, dz, tau, M, N)

    compute_velosity_1(u, w, u_1, w_1, u_2, w_2, beta, rho_1, rho_2, M, N)
    Step4_Fully_Implicit(u, w, u_1, w_1, u_2, w_2, beta, nu, D, d, C_A, C_1D, C_2D, x, y, g, tau, Re, dx, dz, M, N)

    Apply_Boundary_Condition_Beta(beta, u_2, V_m, dx, dz, M, N)
    if time_step > Iter:
        Apply_Boundary_Condition_Inject(beta, u_2, V_m, dx, dz, M, N)
    Step5(u_2, w_2, beta, beta_, D, tau, Re, dx, dz, M, N)
    beta[:, :] = beta_[:, :]


def Step1_RK(u_n, u_step1, u_1, u_2, w_n, w_step1, w_1, w_2, p, Div, beta_n, beta_step1, nu, zeta, V_m, V_m_1, g, A_12,
             rho_1, rho_2, x, y, D, Re, d, C_A, C_1D, C_2D, dx, dz, tau, da, A, ksp, pc, b, x_vec, M, N, time_step, Iter):
    Apply_Boundary_Condition_Velosity(u_n, w_n, u_step1, w_step1, V_m, M, N)
    Apply_Boundary_Condition(u_n, w_n, u_step1, w_step1, V_m_1, dx, dz, M, N)

    Step1(u_n, u_step1, w_n, w_step1, p, beta_n, nu, zeta, g, A_12, y, dx, dz, tau, Re, M, N)

    Step2(u_n, w_n, u_step1, w_step1, u_2, w_2, beta_n, p, da, A, ksp, pc, b, x_vec, D, x, dx, dz, tau, Re, M, N)
    Step3(u_step1, u_step1, w_step1, w_step1, p, dx, dz, tau, M, N)

    compute_velosity_1(u_step1, w_step1, u_1, w_1, u_2, w_2, beta_n, rho_1, rho_2, M, N)
    Step4_Fully_Implicit(u_step1, w_step1, u_1, w_1, u_2, w_2, beta_n, nu, D, d, C_A, C_1D, C_2D, x, y, g, tau, Re, dx, dz, M, N)

    Apply_Boundary_Condition_Beta(beta_n, u_2, V_m, dx, dz, M, N)
    if time_step > Iter:
        Apply_Boundary_Condition_Inject(beta_n, u_2, V_m, dx, dz, M, N)
    Step5(u_2, w_2, beta_n, beta_step1, D, tau, Re, dx, dz, M, N)

def Step2_RK(u_n, u_step1, u_step2, u_1, u_2, w_n, w_step1, w_step2, w_1, w_2, p, Div, beta_n, beta_step1, beta_step2, nu, zeta, V_m, V_m_1, g, A_12,
             rho_1, rho_2, x, y, D, Re, d, C_A, C_1D, C_2D, dx, dz, tau, da, A, ksp, pc, b, x_vec, M, N, time_step, Iter):
    
    Apply_Boundary_Condition_Velosity(u_step1, w_step1, u_step2, w_step2, V_m, M, N)
    Apply_Boundary_Condition(u_step1, w_step1, u_step2, w_step2, V_m_1, dx, dz, M, N)

    Step1(u_step1, u_step2, w_step1, w_step2, p, beta_step1, nu, zeta, g, A_12, y, dx, dz, tau, Re, M, N)

    Step2(u_step1, w_step1, u_step2, w_step2, u_2, w_2, beta_step1, p, da, A, ksp, pc, b, x_vec, D, x, dx, dz, tau, Re, M, N)
    Step3(u_step2, u_step2, w_step2, w_step2, p, dx, dz, tau, M, N)

    compute_velosity_1(u_step2, w_step2, u_1, w_1, u_2, w_2, beta_step1, rho_1, rho_2, M, N)
    Step4_Fully_Implicit(u_step2, w_step2, u_1, w_1, u_2, w_2, beta_step1, nu, D, d, C_A, C_1D, C_2D, x, y, g, tau, Re, dx, dz, M, N)

    Apply_Boundary_Condition_Beta(beta_step1, u_2, V_m, dx, dz, M, N)
    if time_step > Iter:
        Apply_Boundary_Condition_Inject(beta_step1, u_2, V_m, dx, dz, M, N)

    Step5(u_2, w_2, beta_step1, beta_step2, D, tau, Re, dx, dz, M, N)

    # if time_step % 10 == 0:
    #     StepEnd(u_n, w_n, u_2, w_2, beta_n, Div, D, dx, dz, x, Re, M, N)
    #     print(f"Step2 before mixing max(Div v) ={np.max(np.abs(Div))}")
    # if time_step % 10 == 0:
    #     StepEnd(u_step1, w_step1, u_2, w_2, beta_step1, Div, D, dx, dz, x, Re, M, N)
    #     print(f"Step2 before mixing max(Div v_step1) ={np.max(np.abs(Div))}")
    # if time_step % 10 == 0:
    #     StepEnd(u_step2, w_step2, u_2, w_2, beta_step1, Div, D, dx, dz, x, Re, M, N)
    #     print(f"Step2 before mixing max(Div v_step2) ={np.max(np.abs(Div))}")

    for i in prange(0, M):
        for j in range(0, N):
            u_step2[i, j] = 0.75 * u_n[i, j] + 0.25 * u_step2[i, j]
            w_step2[i, j] = 0.75 * w_n[i, j] + 0.25 * w_step2[i, j]

    for i in prange(0, M):
        for j in range(0, N):
            beta_step2[i, j] = 0.75 * beta_n[i, j] + 0.25 * beta_step2[i, j]

    Step2(u_step1, w_step1, u_step2, w_step2, u_2, w_2, beta_step1, p, da, A, ksp, pc, b, x_vec, D, x, dx, dz, tau, Re, M, N)
    Step3(u_step2, u_step2, w_step2, w_step2, p, dx, dz, tau, M, N)

def Step3_RK(u_n, u_step2, u_step3, u_n1, u_1, u_2, w_n, w_step2, w_step3, w_n1, w_1, w_2, p, Div, beta_n, beta_step2, beta_step3, beta_n1, 
             nu, zeta, V_m, V_m_1, g, A_12, rho_1, rho_2, x, y, D, Re, d, C_A, C_1D, C_2D, dx, dz, tau, da, A, ksp, pc, b, x_vec, M, N, time_step,Iter):
    Apply_Boundary_Condition_Velosity(u_step2, w_step2, u_step3, w_step3, V_m, M, N)
    Apply_Boundary_Condition(u_step2, w_step2, u_step3, w_step3, V_m_1, dx, dz, M, N)

    Step1(u_step2, u_step3, w_step2, w_step3, p, beta_step2, nu, zeta, g, A_12, y, dx, dz, tau, Re, M, N)

    Step2(u_step2, w_step2, u_step3, w_step3, u_2, w_2, beta_step2, p, da, A, ksp, pc, b, x_vec, D, x, dx, dz, tau, Re, M, N)
    Step3(u_step3, u_step3, w_step3, w_step3, p, dx, dz, tau, M, N)

    compute_velosity_1(u_step3, w_step3, u_1, w_1, u_2, w_2, beta_step2, rho_1, rho_2, M, N)
    Step4_Fully_Implicit(u_step3, w_step3, u_1, w_1, u_2, w_2, beta_step2, nu, D, d, C_A, C_1D, C_2D, x, y, g, tau, Re, dx, dz, M, N)

    Apply_Boundary_Condition_Beta(beta_step2, u_2, V_m, dx, dz, M, N)
    if time_step > Iter:
        Apply_Boundary_Condition_Inject(beta_step2, u_2, V_m, dx, dz, M, N)

    Step5(u_2, w_2, beta_step2, beta_step3, D, tau, Re, dx, dz, M, N)

    # if time_step % 10 == 0:
    #     StepEnd(u_step3, w_step3, u_2, w_2, beta_step2, Div, D, dx, dz, x, Re, M, N)
    #     print(f"Step3 before mixing max(Div) ={np.max(np.abs(Div))}")

    for i in prange(0, M):
        for j in range(0, N):
            u_n1[i, j] = 1.0 / 3.0 * u_n[i, j] + 2.0 / 3.0 * u_step3[i, j]
            w_n1[i, j] = 1.0 / 3.0 * w_n[i, j] + 2.0 / 3.0 * w_step3[i, j]

    for i in prange(0, M):
        for j in range(0, N):
            beta_n1[i, j] = 1.0 / 3.0 * beta_n[i, j] + 2.0 / 3.0 * beta_step3[i, j]

    Step2(u_step3, w_step3, u_n1, w_n1, u_2, w_2, beta_n1, p, da, A, ksp, pc, b, x_vec, D, x, dx, dz, tau, Re, M, N)
    Step3(u_n1, u_n1, w_n1, w_n1, p, dx, dz, tau, M, N)

    compute_velosity_1(u_n1, w_n1, u_1, w_1, u_2, w_2, beta_n1, rho_1, rho_2, M, N)
    Step4_Fully_Implicit(u_n1, w_n1, u_1, w_1, u_2, w_2, beta_n1, nu, D, d, C_A, C_1D, C_2D, x, y, g, tau, Re, dx, dz, M, N)

def Step_SSPRK3(u_n, u_step1, u_step2, u_step3, u_n1, u_1, u_2, w_n, w_step1, w_step2, w_step3, w_n1, w_1, w_2, p, Div,
                    beta_n, beta_step1, beta_step2, beta_step3, beta_n1, nu, zeta, V_m,
                    V_m_1, g, A_12, rho_1, rho_2, x, y, D, Re, d, C_A, C_1D, C_2D, dx, dz, tau, da, A, ksp, pc, b,
                    x_vec, M, N, time_step, Iter):
    
    # STEP 1
    Step1_RK(u_n, u_step1, u_1, u_2, w_n, w_step1, w_1, w_2, p, Div, beta_n, beta_step1, nu, zeta, V_m, V_m_1, g, A_12,
             rho_1, rho_2, x, y, D, Re, d, C_A, C_1D, C_2D, dx, dz, tau, da, A, ksp, pc, b, x_vec, M, N, time_step, Iter)
    # if time_step % 10 == 0:
    #     StepEnd(u_step1, w_step1, u_2, w_2, beta_step1, Div, D, dx, dz, x, Re, M, N)
    #     print(f"Step1 max(Div)={np.max(np.abs(Div))}")

    #STEP 2
    Step2_RK(u_n, u_step1, u_step2, u_1, u_2, w_n, w_step1, w_step2, w_1, w_2, p, Div, beta_n, beta_step1, beta_step2, nu, zeta, V_m, V_m_1, g, A_12,
             rho_1, rho_2, x, y, D, Re, d, C_A, C_1D, C_2D, dx, dz, tau, da, A, ksp, pc, b, x_vec, M, N, time_step, Iter)
    # if time_step % 10 == 0:
    #     StepEnd(u_step2, w_step2, u_2, w_2, beta_step2, Div, D, dx, dz, x, Re, M, N)
    #     print(f"Step2 max(Div)={np.max(np.abs(Div))}")

    Step3_RK(u_n, u_step2, u_step3, u_n1, u_1, u_2, w_n, w_step2, w_step3, w_n1, w_1, w_2, p, Div, beta_n, beta_step2, beta_step3, beta_n1, 
             nu, zeta, V_m, V_m_1, g, A_12, rho_1, rho_2, x, y, D, Re, d, C_A, C_1D, C_2D, dx, dz, tau, da, A, ksp, pc, b, x_vec, M, N, time_step,Iter)


    # if time_step % 10 == 0:
    #     StepEnd(u_n, w_n, u_2, w_2, beta_n, Div, D, dx, dz, x, Re, M, N)
    #     print(f"Step3 max(Div v_m)={np.max(np.abs(Div))}")
    # if time_step % 10 == 0:
    #     StepEnd(u_step1, w_step1, u_2, w_2, beta_step1, Div, D, dx, dz, x, Re, M, N)
    #     print(f"Step3 max(Div v_step1)={np.max(np.abs(Div))}")
    # if time_step % 10 == 0:
    #     StepEnd(u_step2, w_step2, u_2, w_2, beta_step2, Div, D, dx, dz, x, Re, M, N)
    #     print(f"Step3 max(Div v_step2)={np.max(np.abs(Div))}")
    # if time_step % 10 == 0:
    #     StepEnd(u_step3, w_step3, u_2, w_2, beta_step3, Div, D, dx, dz, x, Re, M, N)
    #     print(f"Step3 max(Div v_step3)={np.max(np.abs(Div))}")
    # if time_step % 10 == 0:
    #     StepEnd(u_n1, w_n1, u_2, w_2, beta_n1, Div, D, dx, dz, x, Re, M, N)
    #     print(f"Step3 max(Div v_n1)={np.max(np.abs(Div))}")

@njit
def compute_vorticity(u, w, dx, dz, M, N):
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


if __name__ == "__main__":
    # L = 0.5, H = 0.085, W = 0.16, width_nozzle = 0.01
    M, N = 204, 38  # Кількість комірок по горизонталі/вертикалі (з 2-ма заграничними)
    dx, dz = 0.0025, 0.0025  # Кроки по сітці по горизонтальному/вертикальному напрямах
    # V_m = 0.3676  # горизонтальна швидкість течії (одна комірка)
    # V_m_1 = 0.4  # вертикальна швидкість течії (одна комірка)
    V_m = 0.4  # горизонтальна швидкість течії (одна комірка)
    V_m_1 =  1.75  # вертикальна швидкість течії (одна комірка)
    tau = 5e-4  # крок за часом
    g = -9.82  # постійна всесвітнього тяжіння

    rho_1 = 1000.0  # Густина рідини
    rho_2 = 7800.0  # Густина твердої фази

    nu_1 = 1e-5  # Кінематична в'язкість
    zeta_1 = 1e-5  # Друга в'язкість

    A_12 = 2.5  # В'язкість за Ейнштейном

    D = 1e-4  # Рознесення частинок
    Re = 1000.0  # Сіткове число Рейнольдса

    C_1D = 18.0  # Лінійний опір (Стокса)
    C_2D = 0.0  # Квадратичний опір (Н'ютона)

    C_A = 0.5  # Приєднана маса

    d = 5E-5  # Усереднений діаметр частинок твердої фази

    x = rho_2 / rho_1
    y = 1.0 - x

    time_step = 0  # поточна ітерація симуляції
    time_step_max = 10000  # максимальна кількість ітерацій симуляції

    Iter = 11000  # кількість ітерацій до запуску частинок

    project_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = os.path.join(project_dir, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    metadata = {"M": M, "N": N, "dx": dx, "dz": dz, "tau": tau, "timestamp": timestamp,
                "V_m": V_m, "V_m_1": V_m_1,
                "rho_1": rho_1, "rho_2": rho_2,
                "nu_1": nu_1, "nu_2": nu_1,
                "A": A_12, "D": D, "Re": Re,
                "C_1D": C_1D, "C_2D": C_2D, "C_A": C_A, "d": d, "x": x, "y": y,
                "time_step_inject": Iter, "time_step_max": time_step_max}
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    def create_h5_file(step_num):
        h5_path = os.path.join(save_dir, f"simulation_data_step_{step_num}.h5")
        f_active = h5py.File(h5_path, "w")

        dsets = {}
        # for name in ["w", "w_1", "beta"]:
        for name in ["u", "w", "p", "omega"]:
            dsets[name] = f_active.create_dataset(
                name,
                shape=(0, M, N),
                maxshape=(None, M, N),
                dtype=np.float64,
                compression="gzip",
                chunks=(1, M, N)
            )
        return f_active, dsets

    h5_file, datasets = create_h5_file(time_step)

    # Euler
    # u = np.zeros((M, N))
    # u_ = np.zeros((M, N))
    # u_1 = np.zeros((M, N))
    # u_2 = np.zeros((M, N))
    #
    # w = np.zeros((M, N))
    # w_ = np.zeros((M, N))
    # w_1 = np.zeros((M, N))
    # w_2 = np.zeros((M, N))
    #
    # p = np.zeros((M, N))
    # Div = np.zeros((M, N))
    #
    # beta = np.zeros((M, N))
    # beta_ = np.zeros((M, N))

    # RK
    u_n = np.zeros((M, N))
    u_step1 = np.zeros((M, N))
    u_step2 = np.zeros((M, N))
    u_step3 = np.zeros((M, N))
    u_n1 = np.zeros((M, N))

    u_1 = np.zeros((M, N))
    u_2 = np.zeros((M, N))

    w_n = np.zeros((M, N))
    w_step1 = np.zeros((M, N))
    w_step2 = np.zeros((M, N))
    w_step3 = np.zeros((M, N))
    w_n1 = np.zeros((M, N))

    w_1 = np.zeros((M, N))
    w_2 = np.zeros((M, N))

    p = np.zeros((M, N))
    Div = np.zeros((M, N))

    beta_n = np.zeros((M, N))
    beta_step1 = np.zeros((M, N))
    beta_step2 = np.zeros((M, N))
    beta_step3 = np.zeros((M, N))
    beta_n1 = np.zeros((M, N))

    omega = np.zeros((M, N))

    da, A, ksp, pc, b, x_vec = Init_PETSc(dx, dz, M, N)

    # Euler
    # u_[:, :] = V_m

    # RK
    u_n[:, :] = V_m

    for time_step in range (0, time_step_max + 1):
        #Step_Euller(u, u_, u_1, u_2, w, w_, w_1, w_2, p, Div, beta, beta_, nu_1, zeta_1, V_m, V_m_1, g, A_12, rho_1, rho_2, x, y, D, Re, d, C_A, C_1D, C_2D, dx, dz, tau, da, A, ksp, pc, b, x_vec, M, N, time_step, Iter)
        Step_SSPRK3(u_n, u_step1, u_step2, u_step3, u_n1, u_1, u_2, w_n, w_step1, w_step2, w_step3, w_n1, w_1, w_2, p, Div,
                    beta_n, beta_step1, beta_step2, beta_step3, beta_n1, nu_1, zeta_1, V_m,
                    V_m_1, g, A_12, rho_1, rho_2, x, y, D, Re, d, C_A, C_1D, C_2D, dx, dz, tau, da, A, ksp, pc, b,
                    x_vec, M, N, time_step, Iter)

        if time_step % 5 == 0:

            if time_step % 100 == 0:
                # StepEnd(u, w, u_2, w_2, beta, Div, D, dx, dz, x, Re, M, N)
                StepEnd(u_n1, w_n1, u_2, w_2, beta_n1, Div, D, dx, dz, x, Re, M, N)
                # print(f"Крок {time_step}: max(u)={np.max(np.abs(u))}, max(w)={np.max(np.abs(w))}, max(p)={np.max(np.abs(p))}, max(beta)={np.max(beta)}, max(Div)={np.max(np.abs(Div))}")
                print(f"Крок {time_step}: max(u)={np.max(np.abs(u_n1))}, max(w)={np.max(np.abs(w_n1))}, max(p)={np.max(np.abs(p))}, max(beta)={np.max(beta_n1)}, max(Div)={np.max(np.abs(Div))}")

                if np.isnan(Div).any() or np.isinf(Div).any():
                    print(f"ВИБУХ ДИВЕРГЕНЦІЇ на кроці {time_step}!")
                    break

            omega = compute_vorticity(u_n1, w_n1, dx, dz, M, N)
            #for name, array in zip(["u", "w", "p", "u_1", "w_1", "u_2", "w_2", "beta"], [u, w, p, u_1, w_1, u_2, w_2, beta]):
            #for name, array in zip(["w", "w_1", "beta"], [w, w_1, beta]):
            # for name, array in zip(["w", "w_1", "beta"], [w_n1, w_1, beta_n1]):
            for name, array in zip(["u", "w", "p", "omega"], [u_n1, w_n1, p, omega]):
                dset = datasets[name]
                curr_len = dset.shape[0]
                dset.resize(curr_len + 1, axis=0)
                dset[curr_len, :, :] = array

            h5_file.flush()

        u_n[:, :] = u_n1[:, :]
        w_n[:, :] = w_n1[:, :]
        beta_n[:, :] = beta_n1[:, :]

        #time_step += 1

    h5_file.close()
    print("Розрахунок завершено! Усі файли успішно збережено в одну папку.")
