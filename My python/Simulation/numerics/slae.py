from numba import njit, prange
import numpy as np

from .operators import divergence
from ..convection.beta import conv_beta_tvd

@njit
def div_beta_w2_simple(u, w, u2, w2, beta, dx, dz, i, j):
    wr_x = u2[i, j] - u[i, j]
    wr_x_m = u2[i-1, j] - u[i-1, j]
    wr_z = w2[i, j] - w[i, j]
    wr_z_m = w2[i, j-1] - w[i, j-1]

    return beta[i, j] * ((wr_x - wr_x_m)/dx + (wr_z - wr_z_m)/dz)

@njit
def build_rhs(u_, w_, u_2, w_2, beta, grid_param, simulate_param, physical_param):
    M = grid_param.M
    N = grid_param.N

    dx = grid_param.dx
    dz = grid_param.dz

    tau = simulate_param.tau

    rhs = np.zeros((M, N))
    y = physical_param.y

    for i in range(2, M - 2):
        for j in range(2, N - 2):
            div_beta_w2 = div_beta_w2_simple(u_, w_, u_2, w_2, beta, dx, dz, i, j)
            
            target_div = divergence(u_, w_, dx, dz, i, j) + y * div_beta_w2
            
            rhs[i, j] = target_div / tau
            
    return rhs