from .steps import *
from ..boundary import *
from .operators import compute_velocity_1

def step1_RK(u_n, u_step1, u_1, u_2, w_n, w_step1, w_1, w_2, p, Div, beta_n, beta_step1, grid_param, simulate_param, physical_param, PETSc_param):
    apply_boundary_condition_velocity(u_n, w_n, u_step1, w_step1, grid_param, simulate_param)
    apply_boundary_condition(u_n, w_n, u_step1, w_step1, grid_param, simulate_param)

    step1(u_n, u_step1, w_n, w_step1, p, beta_n, grid_param, simulate_param, physical_param)

    step2(u_n, w_n, u_step1, w_step1, u_2, w_2, beta_n, p, grid_param, simulate_param, physical_param, PETSc_param)
    step3(u_step1, u_step1, w_step1, w_step1, p, grid_param, simulate_param)

    compute_velocity_1(u_step1, w_step1, u_1, w_1, u_2, w_2, beta_n, grid_param, physical_param)
    step4_semi_fully_implicit(u_step1, w_step1, u_1, w_1, u_2, w_2, beta_n, grid_param, simulate_param, physical_param)

    apply_boundary_condition_beta(beta_n, u_2, grid_param, simulate_param)
    if simulate_param.step > simulate_param.step_inj:
        apply_boundary_condition_inject(beta_n, u_2, grid_param, simulate_param)
    step5(u_2, w_2, beta_n, beta_step1, grid_param, simulate_param, physical_param)

def step2_RK(u_n, u_step1, u_step2, u_1, u_2, w_n, w_step1, w_step2, w_1, w_2, p, Div, beta_n, beta_step1, beta_step2, 
             grid_param, simulate_param, physical_param, PETSc_param):
    M = grid_param.M
    N = grid_param.N

    apply_boundary_condition_velocity(u_step1, w_step1, u_step2, w_step2, grid_param, simulate_param)
    apply_boundary_condition(u_step1, w_step1, u_step2, w_step2, grid_param, simulate_param)

    step1(u_step1, u_step2, w_step1, w_step2, p, beta_step1, grid_param, simulate_param, physical_param)

    step2(u_step1, w_step1, u_step2, w_step2, u_2, w_2, beta_step1, p, grid_param, simulate_param, physical_param, PETSc_param)
    step3(u_step2, u_step2, w_step2, w_step2, p, grid_param, simulate_param)

    compute_velocity_1(u_step2, w_step2, u_1, w_1, u_2, w_2, beta_step1, grid_param, physical_param)
    step4_semi_fully_implicit(u_step2, w_step2, u_1, w_1, u_2, w_2, beta_step1, grid_param, simulate_param, physical_param)

    apply_boundary_condition_beta(beta_step1, u_2, grid_param, simulate_param)
    if simulate_param.step > simulate_param.step_inj:
        apply_boundary_condition_inject(beta_step1, u_2, grid_param, simulate_param)

    step5(u_2, w_2, beta_step1, beta_step2, grid_param, simulate_param, physical_param)

    for i in prange(0, M):
        for j in range(0, N):
            u_step2[i, j] = 0.75 * u_n[i, j] + 0.25 * u_step2[i, j]
            w_step2[i, j] = 0.75 * w_n[i, j] + 0.25 * w_step2[i, j]

    for i in prange(0, M):
        for j in range(0, N):
            beta_step2[i, j] = 0.75 * beta_n[i, j] + 0.25 * beta_step2[i, j]

    step2(u_step1, w_step1, u_step2, w_step2, u_2, w_2, beta_step1, p, grid_param, simulate_param, physical_param, PETSc_param)
    step3(u_step2, u_step2, w_step2, w_step2, p, grid_param, simulate_param)

def step3_RK(u_n, u_step2, u_step3, u_n1, u_1, u_2, w_n, w_step2, w_step3, w_n1, w_1, w_2, p, Div, beta_n, beta_step2, beta_step3, beta_n1, 
             grid_param, simulate_param, physical_param, PETSc_param):
    M = grid_param.M
    N = grid_param.N

    apply_boundary_condition_velocity(u_step2, w_step2, u_step3, w_step3, grid_param, simulate_param)
    apply_boundary_condition(u_step2, w_step2, u_step3, w_step3, grid_param, simulate_param)

    step1(u_step2, u_step3, w_step2, w_step3, p, beta_step2, grid_param, simulate_param, physical_param)

    step2(u_step2, w_step2, u_step3, w_step3, u_2, w_2, beta_step2, p, grid_param, simulate_param, physical_param, PETSc_param)
    step3(u_step3, u_step3, w_step3, w_step3, p, grid_param, simulate_param)

    compute_velocity_1(u_step3, w_step3, u_1, w_1, u_2, w_2, beta_step2, grid_param, physical_param)
    step4_semi_fully_implicit(u_step3, w_step3, u_1, w_1, u_2, w_2, beta_step2, grid_param, simulate_param, physical_param)

    apply_boundary_condition_beta(beta_step2, u_2, grid_param, simulate_param)
    if simulate_param.step > simulate_param.step_inj:
        apply_boundary_condition_inject(beta_step2, u_2, grid_param, simulate_param)

    step5(u_2, w_2, beta_step2, beta_step3, grid_param, simulate_param, physical_param)

    for i in prange(0, M):
        for j in range(0, N):
            u_n1[i, j] = 1.0 / 3.0 * u_n[i, j] + 2.0 / 3.0 * u_step3[i, j]
            w_n1[i, j] = 1.0 / 3.0 * w_n[i, j] + 2.0 / 3.0 * w_step3[i, j]

    for i in prange(0, M):
        for j in range(0, N):
            beta_n1[i, j] = 1.0 / 3.0 * beta_n[i, j] + 2.0 / 3.0 * beta_step3[i, j]

    step2(u_step3, w_step3, u_n1, w_n1, u_2, w_2, beta_n1, p, grid_param, simulate_param, physical_param, PETSc_param)
    step3(u_n1, u_n1, w_n1, w_n1, p, grid_param, simulate_param)

    compute_velocity_1(u_n1, w_n1, u_1, w_1, u_2, w_2, beta_n1, grid_param, physical_param)
    step4_semi_fully_implicit(u_n1, w_n1, u_1, w_1, u_2, w_2, beta_n1, grid_param, simulate_param, physical_param)

def step_SSPRK3(u_n, u_step1, u_step2, u_step3, u_n1, u_1, u_2, w_n, w_step1, w_step2, w_step3, w_n1, w_1, w_2, p, Div,
                    beta_n, beta_step1, beta_step2, beta_step3, beta_n1,
                    grid_param, simulate_param, physical_param, PETSc_param):
    
    # step 1
    step1_RK(u_n, u_step1, u_1, u_2, w_n, w_step1, w_1, w_2, p, Div, beta_n, beta_step1, grid_param, simulate_param, physical_param, PETSc_param)

    #step 2
    step2_RK(u_n, u_step1, u_step2, u_1, u_2, w_n, w_step1, w_step2, w_1, w_2, p, Div, beta_n, beta_step1, beta_step2, grid_param, simulate_param, physical_param, PETSc_param)

    step3_RK(u_n, u_step2, u_step3, u_n1, u_1, u_2, w_n, w_step2, w_step3, w_n1, w_1, w_2, p, Div, beta_n, beta_step2, beta_step3, beta_n1, 
             grid_param, simulate_param, physical_param, PETSc_param)