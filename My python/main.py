import os
os.environ["NUMBA_NUM_THREADS"] = "4"

import numba

import numpy as np

import json
from datetime import datetime

from Simulation.params import GridParams, SimulateParams, PhysicalParams
from Simulation.solvers import *

from Simulation.io import *

from Simulation.numerics.euler import step_Euller

from Simulation.numerics.steps import step_end

from Simulation.boundary import *

from enum import Enum

class Nozzles(Enum):
    T = "True"
    F = "False"

if __name__ == "__main__":
    """
    ==============================================================
    1. ПАРАМЕТРИ СІТКИ
    ==============================================================
    """

    L, H, W = 0.5, 0.085, 0.16 # Довжина, висота, ширина резервуара
    M, N = 204, 38  # Кількість комірок по горизонталі/вертикалі (з 2-ма заграничними з кожного боку)
    inlet = 0.02 * W
    dx, dz = L / (M - 4), H / (N - 4)  # Кроки по сітці по горизонтальному/вертикальному напрямах (без урахування 4 заграничних)

    Nozzles_activate = Nozzles.T

    nozzles = ()
    if Nozzles_activate.value:
        nozzles = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45) # Координати центрів форсунок

    nozzle_r = 0.005 # Радіус кожної форсунки

    # Створення екземпляру класа сітка
    grid_param = GridParams(M, N, dx, dz)

    """
    ==============================================================
    2. ФІЗИЧНІ ПАРАМЕТРИ

    - постійна всесвітнього тяжіння

    - густини

    - в'язкості

    - опори/приєндана маса

    - усереднений діаметр частинок
    ==============================================================
    """

    g = -9.82  # постійна всесвітнього тяжіння

    rho_1 = 1000.0  # Густина рідини
    rho_2 = 7800.0  # Густина твердої фази

    nu = 1.004e-6  # Кінематична в'язкість
    zeta = 1.004e-6  # Об'ємна в'язкість
     
    A_12 = 2.5  # В'язкість за Ейнштейном
    D = 1e-4  # Рознесення частинок

    C_1D = 18.0  # Лінійний опір (Стокса)
    C_2D = 0.0  # Квадратичний опір (Н'ютона)
    C_A = 0.5  # Приєднана маса

    d = 2.5e-4  # Усереднений діаметр частинок твердої фази

    x = rho_2 / rho_1
    y = 1.0 - x

    # Створення екземпляру класа фізичних параметрів симуляції
    physical_param = PhysicalParams(rho_1, rho_2, nu, zeta, A_12, D, C_1D, C_2D, C_A, d)

    """
    ===============================================================
    3. ПАРАМЕТРИ СИМУЛЯЦІЇ

    - крок за часом
    - сіткові числа

    - витрати рідини
    - швидкості

    - об'ємна концентрація частинок яка подається
    - обмеження дивергенції

    - налаштування вирішувача PETSc для рівняння Пуассона для тиску
    - вибір часової дискретизації
    - налаштування кроків симуляції
    ===============================================================
    """

    tau = 1e-4  # Крок за часом

    Re = 2.0  # Сіткове число Рейнольдса
    Sh = 0.7 # Турбулентне число Шмідта

    h = 0.05 # Висота кінця жолоба для подачи частинок
    l = 0.5 # Довжина жолоба для подачі частинок

    angle = 30 # Кут напрямку жолоба для подачі частинок до резервуару
    theta = np.deg2rad(angle) # Кут в радіанах

    a = 9.81 * np.sin(theta) # Прискорення частинок враховуючи силу тяжіння і кут

    Q_x = 0.00575 # Витрати рідини в трубі
    Q_z = 0.00575 # Витрати рідини через форсунки

    V_x = Q_x / H / W  # Горизонтальна швидкість течії (одна комірка)
    V_x_end = (Q_x + Q_z) / H / W # Горизонтальна швидкість течії (одна комірка, з урахуванням рідини через форсунки)
    V_z = 0.0 # Вертикальна швидкість течії (одна комірка)
    if len(nozzles) > 0:
        V_z = Q_z / W / (2.0 * nozzle_r) / len(nozzles)  # Вертикальна швидкість течії (одна комірка)

    V_chute = np.sqrt(2 * a * l) # Швидкість кочення по жолобу для подачі частинок

    Vx_imp = V_chute * np.cos(theta) # Горизонтальна швидкість входу частинок в рідину
    Vz_imp = V_chute * np.sin(theta) # Вертикальна швидкість входу частинок в рідину

    if h > 0.0:
        disc = Vz_imp**2 + 2 * 9.81 * h
        t = (-Vz_imp + np.sqrt(disc)) / 9.81
        Vz_imp = Vz_imp - 9.81 * t

    V_imp = np.sqrt(Vx_imp**2 + Vz_imp**2) # Швидкість входження частинок в поток
    angle_deg = np.rad2deg(np.arctan2(Vz_imp, Vx_imp)) # Кут входження частинок в рідину

    m = 0.25 # Кг/с Маса частинок які подаються
    Vdot_p = m / rho_2
    particle_vol_conc = Vdot_p / (inlet * V_imp)

    Div_max = 1e+10 # Максимальна дивергенція

    # Налаштування вирішувача PETSc
    rtol = '1e-10' # Відносна похибка
    atol = '1e-12' # Абсолютна похибка
    max_iter = '2000' # Максимальна кількість ітерацій

    # Кількість кроків симуляції
    step = 0  # Поточна ітерація симуляції
    step_inj = 1000  # кількість ітерацій до запуску частинок
    step_max = 20000  # Максимальна кількість ітерацій симуляції

    step_create_file = 100000 # Кількість кроків між створенням .h5 файлів
    step_save = 50 # Інтервал збереження масивів даних (кроки)
    step_save_start = 0 # кількість кроків перед початком збереження масивів даних
    step_save_end = step_max # кількість кроків після яких не зберігаються дані (за замовчуванням - кількість кроків симуляції)

    step_check = 100 # Інтервал перевірки збіжності (кроки)

    # Створення екземпляру класа симуляції
    simulate_param = SimulateParams(tau, Re, Sh, V_x, V_z, Vx_imp, Vz_imp, particle_vol_conc, step_max, step_inj, nozzles, nozzle_r)

    pressure_solver = PressureSolverPETSc(M, N, dx, dz, rtol, atol, max_iter)

    """
    ==============================================================
    4. ПІДГОТОВКА СИМУЛЯЦІЇ

    - створення папки симуляції

    - логування вхідних даних
    ==============================================================
    """

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

    p_n = np.zeros((M, N)) # Тиск

    Div = np.zeros((M, N)) # Дивергенція

    # Об'ємна частка твердої фази в комірці
    beta_n = np.zeros((M, N))
    beta_ = np.zeros((M, N))

    # Кроки симуляції
    field_names = ["u", "w", "p", "beta"]
    with HDF5Writer(save_dir, shape_spatial=(M, N), field_names=field_names, step_create=step_create_file) as writer, \
            JSONLogger(save_dir, filename="simulation_log.json") as logger:
        for step in range(0, step_max + 1):
            step_Euller(u_n, u_, u_1, u_2, w_n, w_, w_1, w_2, p_n, Div, beta_n, beta_, grid_param, simulate_param, physical_param, pressure_solver, step_check)

            if step % step_check == 0:
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

    print("Розрахунок завершено! Усі файли успішно збережено в одну папку.")
