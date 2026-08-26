from .steps import *
from ..boundary import *
from .operators import compute_velocity_1

def step_Euller(u, u_, u_1, u_2, w, w_, w_1, w_2, p, Div, beta, beta_, grid_param, simulate_param, physical_param, PETSc_param):
    apply_boundary_condition_velocity(u, w, u_, w_, grid_param, simulate_param)
    apply_boundary_condition(u, w, u_, w_, grid_param, simulate_param)

    step1(u, u_, w, w_, p, beta, grid_param, simulate_param, physical_param)

    step2(u, w, u_, w_, u_2, w_2, beta, p, grid_param, simulate_param, physical_param, PETSc_param)
    step3(u, u_, w, w_, p, grid_param, simulate_param)

    compute_velocity_1(u, w, u_1, w_1, u_2, w_2, beta, grid_param, physical_param)
    step4_semi_fully_implicit(u, w, u_1, w_1, u_2, w_2, beta, grid_param, simulate_param, physical_param)

    apply_boundary_condition_beta(beta, u_2, grid_param, simulate_param)
    if simulate_param.step > simulate_param.step_inj:
        apply_boundary_condition_inject(beta, u_2, grid_param, simulate_param)
    step5(u_2, w_2, beta, beta_, grid_param, simulate_param, physical_param)
    beta[:, :] = beta_[:, :]