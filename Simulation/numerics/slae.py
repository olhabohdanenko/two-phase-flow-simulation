from numba import njit, prange
import numpy as np

from .operators import divergence
from ..convection.beta import conv_beta_tvd

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
            div_beta_w2 = -conv_beta_tvd(u_2, w_2, beta, dx, dz, i, j) + conv_beta_tvd(u_, w_, beta, dx, dz, i, j)
            
            target_div = divergence(u_, w_, dx, dz, i, j) - y * div_beta_w2
            
            rhs[i, j] = target_div / tau
            
    return rhs