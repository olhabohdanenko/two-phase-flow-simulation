import numpy as np

# ============================================================
# 1. Швидкість осідання (Stokes)
# ============================================================
def compute_settling_velocity(rho_f, rho_p, nu, diameter):
    return (9.81 * (rho_p - rho_f) * diameter**2) / (18 * nu * rho_f)


def compute_sedimentation():
    rho_fluid = [1000]
    rho_particle = [2400, 7800]
    diameter = [5e-5, 1.5e-4, 2.5e-4, 5e-4]
    Nu = [1e-6, 1e-5, 1e-4]

    print("=" * 74)
    print("=== Швидкість осідання (формула Стокса) ===")
    print("=" * 74)

    print(f"{'№':>3} | {'Material':<10} | {'ρp':>6} | {'d [m]':>8} | {'ρf':>5} | {'ν [m²/s]':>10} | {'V_settle [m/s]':>14}")
    print("-" * 74)

    number = 0
    for rho_f in rho_fluid:
        for rho_p in rho_particle:
            for d in diameter:
                for nu in Nu:
                    V = compute_settling_velocity(rho_f, rho_p, nu, d)
                    material = "Abrasive" if rho_p == 2400 else "Metallic"
                    print(f"{number:>3} | {material:<10} | {rho_p:>6} | {d:>8.1e} | {rho_f:>5} | {nu:>10.1e} | {V:>14.4e}")
                    number += 1
    print()


# ============================================================
# 2. Швидкість входу частинок у рідину
# ============================================================
def particle_exit_velocity(L, theta_deg):
    theta = np.deg2rad(theta_deg)

    a = 9.81 * np.sin(theta)

    V_chute = np.sqrt(2 * a * L)

    Vx = V_chute * np.cos(theta)
    Vz = V_chute * np.sin(theta)

    return V_chute, Vx, Vz


def free_fall(Vz0, h):
    if h <= 0:
        return Vz0
    
    g = 9.81

    disc = Vz0**2 + 2 * g * h

    t = (-Vz0 + np.sqrt(disc)) / g

    Vz = Vz0 + g * t

    return Vz


def impact_velocity(Vx, Vz):
    V_imp = np.sqrt(Vx**2 + Vz**2)

    angle_deg = np.rad2deg(np.arctan2(Vz, Vx))

    return V_imp, angle_deg


def compute_particle_velocities():
    L_list = [0.5, 1.0, 1.5]
    h_list = [0.0, 0.05, 0.1]
    theta_list = [30, 45]

    print("=" * 80)
    print("=== Швидкість входу частинок у рідину (без тертя) ===")
    print("=" * 80)

    print(f"{'№':>3} | {'L':>5} | {'h':>5} | {'θ':>4} | {'V_chute':>8} | {'Vx':>8} | {'Vy':>8} | {'|V_imp|':>8} | {'Angle°':>7}")
    print("-" * 80)

    number = 0
    for L in L_list:
        for h in h_list:
            for theta in theta_list:
                V_chute, Vx, Vy0 = particle_exit_velocity(L, theta)
                Vy = free_fall(Vy0, h)
                V_imp, angle = impact_velocity(Vx, Vy)

                print(f"{number:>3} | {L:>5.1f} | {h:>5.2f} | {theta:>4} | {V_chute:>8.4f} | {Vx:>8.4f} | {Vy:>8.4f} | {V_imp:>8.4f} | {angle:>7.2f}")
                number += 1

    print()


# ============================================================
# 3. Об'ємна концентрація α і масова витрата
# ============================================================
def compute_concentration_and_massflow():
    rho_p = 7800.0                  # кг/м³
    A_inlet = 0.02 * 0.16           # 0.0032 м² (з topoSetDict)
    v_imp = 2.426                   # м/с

    print("=" * 47)
    print("=== Об'ємна концентрація α і масова витрата ===")
    print("=" * 47)
    print(f"Площа входу A_inlet = {A_inlet:.4f} м²")
    print(f"Швидкість входу v_imp = {v_imp:.3f} м/с")
    print(f"Густина частинок ρp = {rho_p} кг/м³\n")

    print(f"{'m [кг/с]':>10} | {'V̇_p [м³/с]':>13} | {'α':>10}")
    print("-" * 38)
    for m in [0.2, 0.5, 0.6, 1.0, 2.5, 5.0, 10.0]:
        Vdot_p = m / rho_p
        alpha = Vdot_p / (A_inlet * v_imp)

        print(f"{m:>10.1f} | {Vdot_p:>12.6f} | {alpha:>10.5f}")

    print("")
    print(f"{'α':>10} | {'V̇_p [м³/с]':>13} | {'m [кг/с]':>10}")
    print("-" * 38)
    for alpha in [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5]:
        Vdot_p = alpha * A_inlet * v_imp
        m = Vdot_p * rho_p
        print(f"{alpha:>10.3f} | {Vdot_p:>12.6f} | {m:>10.3f}")

    print()


# ============================================================
# Запуск
# ============================================================
if __name__ == "__main__":
    compute_sedimentation()
    compute_particle_velocities()
    compute_concentration_and_massflow()