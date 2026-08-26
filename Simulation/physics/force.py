from numba import njit, prange
import numpy as np

# Force
@njit
def gravity_force(beta, i, j, physical_param):
	return 0.5 * (beta[i, j] + beta[i, j + 1]) * physical_param.y * physical_param.g