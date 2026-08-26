from numba import njit
import numpy as np

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