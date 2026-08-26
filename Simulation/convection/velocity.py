from numba import njit, prange
import numpy as np

from .helpers import *
from ..physics.limiters import *

# Velocity

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