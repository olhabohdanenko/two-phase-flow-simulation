import os
os.environ["NUMBA_NUM_THREADS"] = "4"

import numba

import numpy as np

import json
from datetime import datetime

from Simulation.params import GridParams, SimulateParams, PhysicalParams
from Simulation.solvers import *

from Simulation.io import *

from Simulation.numerics.ssp_rk3 import step_SSPRK3
from Simulation.numerics.euler import step_Euller
from Simulation.numerics.steps import step_end

from Simulation.utils import compute_vorticity
from Simulation.boundary import *


if __name__ == "__main__":
    M, N = 204, 38  # Кількість комірок по горизонталі/вертикалі (з 2-ма заграничними з кожного боку)
    dx, dz = 0.0025, 0.0025  # Кроки по сітці по горизонтальному/вертикальному напрямах (без урахування 4 заграничних)
    L, H, W = (M - 4) * dx, (N - 4) * dz, 0.16 # Довжина, висота, ширина резервуара

    nozzles = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45) # Координати центрів форсунок
    # nozzles = ()
    nozzle_r = 0.005 # Радіус однієї форсунки

    # Створення екземпляру класа сітка
    grid_param = GridParams(M, N, dx, dz)


    # Параметри симуляції
    tau = 1e-4  # Крок за часом

    Re = 2.0  # Сіткове число Рейнольдса
    Sh = 2.0 # Сіткове число Шервуда

    Q_x = 0.00575 # Витрати рідини в трубі
    Q_z = 0.00575 # Витрати рідини через форсунки

    V_x = Q_x / H / W  # Горизонтальна швидкість течії (одна комірка)
    V_x_end = (Q_x + Q_z) / H / W # Горизонтальна швидкість течії (одна комірка, з урахуванням рідини через форсунки)
    V_z = 0.0 # Вертикальна швидкість течії (одна комірка)
    if len(nozzles) > 0:
        V_z = Q_z / W / (2.0 * nozzle_r) / len(nozzles)  # вертикальна швидкість течії (одна комірка)

    Div_max = 1e+10 # Максимальна дивергенція

    # Налаштування вирішувача PETSc
    rtol = '1e-10' # Відносна похибка
    atol = '1e-12' # Абсолютна похибка
    max_iter = '2000' # Максимальна кількість ітерацій

    # Кількість кроків симуляції
    step = 0  # Поточна ітерація симуляції
    step_inj = 1000  # кількість ітерацій до запуску частинок
    step_max = 10000  # Максимальна кількість ітерацій симуляції

    step_create_file = 500000 # Кількість кроків між створенням .h5 файлів
    step_save = 20 # Інтервал збереження масивів даних (кроки)
    step_save_start = 0 # кількість кроків перед початком збереження масивів даних
    step_save_end = step_max # кількість кроків після яких не зберігаються дані (за замовчуванням - кількість кроків симуляції)

    step_check = 100 # Інтервал перевірки збіжності (кроки)

    # Створення екземпляру класа симуляції
    simulate_param = SimulateParams(tau, Re, Sh, V_x, V_z, step_max, step_inj, nozzles, nozzle_r)


    g = -9.82  # постійна всесвітнього тяжіння

    rho_1 = 1000.0  # Густина рідини
    rho_2 = 7800.0  # Густина твердої фази

    nu = 1.004e-4  # Кінематична в'язкість
    zeta = 1.004e-4  # Об'ємна в'язкість
     
    A_12 = 2.5  # В'язкість за Ейнштейном
    D = 1e-4  # Рознесення частинок

    C_1D = 18.0  # Лінійний опір (Стокса)
    C_2D = 0.0  # Квадратичний опір (Н'ютона)
    C_A = 0.5  # Приєднана маса

    d = 1e-3  # Усереднений діаметр частинок твердої фази

    x = rho_2 / rho_1
    y = 1.0 - x

    # Створення екземпляру класа фізичних параметрів симуляції
    physical_param = PhysicalParams(rho_1, rho_2, nu, zeta, A_12, D, C_1D, C_2D, C_A, d)


    # Створення екземпляру розв'язувача СЛАР для знаходження поля тиску за допомогою PETSc (матриця Лапласіана)
    PETSc_param = PETScSolver(M, N, dx, dz, rtol, atol, max_iter)


    # Створення папки симуляції
    project_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = os.path.join(project_dir, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)


    # Запис вхідних даних симуляції
    metadata = {"M": M, "N": N, "dx": dx, "dz": dz, "tau": tau, "timestamp": timestamp,
                "Q_x": Q_x, "Q_z": Q_z,
                "V_x_in": V_x, "V_x_out": V_x_end, "V_z": V_z,
                "nozzles": nozzles, "nozzle_r": nozzle_r,
                "rho_1": rho_1, "rho_2": rho_2,
                "nu": nu, "zeta": zeta,
                "A": A_12, "D": D, "Re": Re,
                "C_1D": C_1D, "C_2D": C_2D, "C_A": C_A, "d": d, "x": x, "y": y,
                "step_inject": step_inj, "step_max": step_max, "step_check": step_check,
                "step_save": step_save, "step_save_start": step_save_start, "step_save_end": step_save_end,
                "time discretization": "Euler"}
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    """
    СТВОРЕННЯ МАСИВІВ І ПОЧАТОК СИМУЛЯЦІЇ
    """
    # # Створення масивів для SSP_RK3
    # # Горизонтальні швидкості
    # u_n = np.zeros((M, N))
    # u_step1 = np.zeros((M, N))
    # u_step2 = np.zeros((M, N))
    # u_step3 = np.zeros((M, N))
    # u_n1 = np.zeros((M, N))

    # u_1 = np.zeros((M, N)) # Рідини
    # u_2 = np.zeros((M, N)) # Твердої фази

    # # Вертикальні швидкості
    # w_n = np.zeros((M, N))
    # w_step1 = np.zeros((M, N))
    # w_step2 = np.zeros((M, N))
    # w_step3 = np.zeros((M, N))
    # w_n1 = np.zeros((M, N))

    # w_1 = np.zeros((M, N)) # Рідини
    # w_2 = np.zeros((M, N)) # Твердої фази

    # p_n = np.zeros((M, N))
    # p_step1 = np.zeros((M, N))
    # p_step2 = np.zeros((M, N))
    # p_step3 = np.zeros((M, N))
    # p_n1 = np.zeros((M, N))

    # phi = np.zeros((M, N)) # Прирост тиску
    # Div = np.zeros((M, N)) # Дивергенція

    # # Об'ємна частка твердої фази в комірці
    # beta_n = np.zeros((M, N))
    # beta_step1 = np.zeros((M, N))
    # beta_step2 = np.zeros((M, N))
    # beta_step3 = np.zeros((M, N))
    # beta_n1 = np.zeros((M, N))

    # # На старому часовому кроці задаємо швидкість в каналі
    # # u_n[:, :] = V_x
    # beta_n[:, :] = 1e-3

    # # Кроки симуляції
    # field_names = ["p", "beta"]
    # with HDF5Writer(save_dir, shape_spatial=(M, N), field_names=field_names, step_create=step_create_file) as writer, \
    #         JSONLogger(save_dir, filename="simulation_log.json") as logger:
    #     for step in range(0, step_max + 1):

    #         step_SSPRK3(u_n, u_step1, u_step2, u_step3, u_n1, u_1, u_2, w_n, w_step1, w_step2, w_step3, w_n1, w_1, w_2, 
    #                     p_n, p_step1, p_step2, p_step3, p_n1, phi, Div,
    #                     beta_n, beta_step1, beta_step2, beta_step3, beta_n1, grid_param, simulate_param, physical_param, PETSc_param)

    #         if step % step_check == 0:
    #             step_end(u_n1, w_n1, u_2, w_2, beta_n1, Div, grid_param, simulate_param, physical_param)
    #             is_exploded = logger.log_step(step, u_n1, w_n1, p_n1, beta_n1, Div, Div_max)
                
    #             if is_exploded:
    #                 break

    #         if step_save_start <= step <= step_save_end and step % step_save == 0:
    #             writer.write_step(step, {
    #                 "p": p_n1,
    #                 "beta": beta_n1
    #             })

    #         u_n[:, :] = u_n1[:, :]
    #         w_n[:, :] = w_n1[:, :]
    #         p_n[:, :] = p_n1[:, :]
    #         beta_n[:, :] = beta_n1[:, :]

    #         simulate_param.step = step

    """
    СТВОРЕННЯ МАСИВІВ І ПОЧАТОК СИМУЛЯЦІЇ
    """

    # Створення масивів для Euler
    # Горизонтальні швидкості
    u_n = np.zeros((M, N))
    u_ = np.zeros((M, N))

    u_1 = np.zeros((M, N)) # Рідини
    u_2 = np.zeros((M, N)) # Твердої фази

    # Вертикальні швидкості
    w_n = np.zeros((M, N))
    w_ = np.zeros((M, N))

    w_1 = np.zeros((M, N)) # Рідини
    w_2 = np.zeros((M, N)) # Твердої фази

    p_n = np.zeros((M, N))

    phi = np.zeros((M, N)) # Прирост тиску
    Div = np.zeros((M, N)) # Дивергенція

    # Об'ємна частка твердої фази в комірці
    beta_n = np.zeros((M, N))
    beta_ = np.zeros((M, N))

    # На старому часовому кроці задаємо швидкість в каналі
    # u_n[:, :] = V_x
    # beta_n[:, :] = 1E-3

    # Кроки симуляції
    field_names = ["u", "w", "p", "beta"]
    with HDF5Writer(save_dir, shape_spatial=(M, N), field_names=field_names, step_create=step_create_file) as writer, \
            JSONLogger(save_dir, filename="simulation_log.json") as logger:
        for step in range(0, step_max + 1):
            step_Euller(u_n, u_, u_1, u_2, w_n, w_, w_1, w_2, p_n, phi, Div, beta_n, beta_, grid_param, simulate_param, physical_param, PETSc_param)

            if step % step_check == 0:
                step_end(u_n, w_n, u_2, w_2, beta_n, Div, grid_param, simulate_param, physical_param)
                is_exploded = logger.log_step(step, u_n, w_n, p_n, beta_n, Div, Div_max)
                
                if is_exploded:
                    break

            if step_save_start <= step <= step_save_end and step % step_save == 0:
                writer.write_step(step, {
                    "u": u_n,
                    "w": w_n,
                    "p": p_n,
                    "beta": beta_n
                })

            simulate_param.step = step

    # u_center = u_n[M//2, :]   # беремо рядок у центрі по x (M//2)
    # for j in range(0, N, 2):
    #     z = j * dz
    #     print(f"z={z:.4f}, u={u_center[j]:.6f}")

    # p_center = p_n[:, N//2]  # беремо центральний ряд
    # for i in range(0, M, 10):
    #     print(f"x={i*dx:.4f}, p={p_center[i]:.6f}")

    # i = M // 2
    # for j in range(0, N, 10):
    #     z = j * dz
    #     print(f"z = {z:.4f}, p = {p_n[i, j]:.6f}")

    # print("N =", N)
    # print("p біля j=0 (дно?):", p_n[M//2, 0], p_n[M//2, 1], p_n[M//2, 2])
    # print("p біля j=N-1 (верх?):", p_n[M//2, N-3], p_n[M//2, N-2], p_n[M//2, N-1])

    print("Розрахунок завершено! Усі файли успішно збережено в одну папку.")
