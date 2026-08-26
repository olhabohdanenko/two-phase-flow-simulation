import os
os.environ["NUMBA_NUM_THREADS"] = "2"

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


if __name__ == "__main__":
    # L = 0.5, H = 0.085, W = 0.16, width_nozzle = 0.01
    M, N = 204, 38  # Кількість комірок по горизонталі/вертикалі (з 2-ма заграничними з кожного боку)
    dx, dz = 0.0025, 0.0025  # Кроки по сітці по горизонтальному/вертикальному напрямах (без урахування 4 заграничних)

    # Створення екземпляру класа сітка
    grid_param = GridParams(M, N, dx, dz)

    # Параметри симуляції
    tau = 1e-3  # крок за часом
    Re = 100.0  # Сіткове число Рейнольдса

    # V_m = 0.3676  # горизонтальна швидкість течії (одна комірка)
    # V_m_1 = 0.4  # вертикальна швидкість течії (одна комірка)
    V_x = 0.37  # горизонтальна швидкість течії (одна комірка)
    V_z =  0.4  # вертикальна швидкість течії (одна комірка)

    step = 0  # поточна ітерація симуляції
    step_inj = 1000  # кількість ітерацій до запуску частинок
    step_max = 600000  # максимальна кількість ітерацій симуляції

    step_create_file = 30000 # кількість кроків між створенням .h5 файлів
    step_save = 100 # інтервал збереження масивів даних (кроки)
    step_save_start = 0 # кількість кроків перед початком збереження масивів даних
    step_save_end = step_max # кількість кроків після яких не зберігаються дані (за замовчуванням - кількість кроків симуляції)

    step_check = 100 # інтервал перевірки збіжності (кроки)

    nozzles = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45) # координати центрів форсунок
    nozzle_r = 0.005 # радіус однієї форсунки

    # Створення екземпляру класа симуляції
    simulate_param = SimulateParams(tau, Re, V_x, V_z, step_max, step_inj, nozzles, nozzle_r)

    g = -9.82  # постійна всесвітнього тяжіння

    rho_1 = 1000.0  # Густина рідини
    rho_2 = 2400.0  # Густина твердої фази

    nu = 1e-5  # Кінематична в'язкість
    zeta = 1e-5  # Об'ємна в'язкість
    A_12 = 2.5  # В'язкість за Ейнштейном
    D = 1e-4  # Рознесення частинок

    C_1D = 18.0  # Лінійний опір (Стокса)
    C_2D = 0.0  # Квадратичний опір (Н'ютона)
    C_A = 0.5  # Приєднана маса

    d = 5E-4  # Усереднений діаметр частинок твердої фази

    x = rho_2 / rho_1
    y = 1.0 - x

    # Створення екземпляру класа фізичних параметрів симуляції
    physical_param = PhysicalParams(rho_1, rho_2, nu, zeta, A_12, D, C_1D, C_2D, C_A, d)

    # Створення екземпляру розв'язувача СЛАР для знаходження поля тиску за допомогою PETSc (матриця Лапласіана)
    PETSc_param = PETScSolver(M, N, dx, dz)

    # Створення папки симуляції
    project_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = os.path.join(project_dir, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Запис вхідних даних симуляції
    metadata = {"M": M, "N": N, "dx": dx, "dz": dz, "tau": tau, "timestamp": timestamp,
                "V_x": V_x, "V_z": V_z,
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

    # Створення масивів для SSP_RK3
    # Горизонтальні швидкості
    u_n = np.zeros((M, N))
    u_step1 = np.zeros((M, N))
    # u_step2 = np.zeros((M, N))
    # u_step3 = np.zeros((M, N))
    # u_n1 = np.zeros((M, N))

    u_1 = np.zeros((M, N)) # Рідини
    u_2 = np.zeros((M, N)) # Твердої фази

    # Вертикальні швидкості
    w_n = np.zeros((M, N))
    w_step1 = np.zeros((M, N))
    # w_step2 = np.zeros((M, N))
    # w_step3 = np.zeros((M, N))
    # w_n1 = np.zeros((M, N))

    w_1 = np.zeros((M, N)) # Рідини
    w_2 = np.zeros((M, N)) # Твердої фази

    p = np.zeros((M, N)) # Тиск
    Div = np.zeros((M, N)) # Дивергенція

    # Об'ємна частка твердої фази в комірці
    beta_n = np.zeros((M, N))
    beta_step1 = np.zeros((M, N))
    # beta_step2 = np.zeros((M, N))
    # beta_step3 = np.zeros((M, N))
    # beta_n1 = np.zeros((M, N))

    # Вихровість
    # omega = np.zeros((M, N))

    # На старому часовому кроці задаємо швидкість в каналі
    u_step1[:, :] = V_x

    # Кроки симуляції
    field_names = ["u", "w", "p", "beta"]
    with HDF5Writer(save_dir, shape_spatial=(M, N), field_names=field_names, step_create=step_create_file) as writer, \
            JSONLogger(save_dir, filename="simulation_log.json") as logger:
        for step in range(0, step_max + 1):
            # step_SSPRK3(u_n, u_step1, u_step2, u_step3, u_n1, u_1, u_2, w_n, w_step1, w_step2, w_step3, w_n1, w_1, w_2, p, Div,
            #             beta_n, beta_step1, beta_step2, beta_step3, beta_n1, grid_param, simulate_param, physical_param, PETSc_param)

            step_Euller(u_n, u_step1, u_1, u_2, w_n, w_step1, w_1, w_2, p, Div, beta_n, beta_step1, grid_param, simulate_param, physical_param, PETSc_param)

            if step % step_check == 0:
                step_end(u_n, w_n, u_2, w_2, beta_n, Div, grid_param, simulate_param, physical_param)
                is_exploded = logger.log_step(step, u_n, w_n, p, beta_n, Div)
                
                if is_exploded:
                    break

            if step_save_start <= step <= step_save_end and step % step_save == 0:
                # omega = compute_vorticity(u_n1, w_n1, grid_param)

                writer.write_step(step, {
                    "u": u_n,
                    "w": w_n,
                    "p": p,
                    "beta": beta_n
                })

            # u_n[:, :] = u_n1[:, :]
            # w_n[:, :] = w_n1[:, :]
            # beta_n[:, :] = beta_n1[:, :]

            simulate_param.step = step

    print("Розрахунок завершено! Усі файли успішно збережено в одну папку.")
