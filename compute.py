from tabulate import tabulate
import numpy as np

def compute(rho_f, rho_p, nu, diameter):
    return (-9.81 * (rho_p - rho_f) * diameter**2) / (18 * nu * rho_f)

if __name__ == "__main__":
    rho_fluid = [1000]
    rho_particle = [2400, 7800]

    diameter = [5e-5, 2.5e-4, 5e-4]
    Nu = [1e-6, 1e-5, 1e-4]

    headers = [" № ", "Material", "Density_p", "Diameter p.", "Density_f", "Kin.fluid.visc.", "Settling V"]
    results_table = []

    number = 0

    for rho_f in rho_fluid:
        for rho_p in rho_particle:
            for d in diameter:
                for nu in Nu:
                    V = compute(rho_f, rho_p, nu, d)
                    material = "Abrasive" if rho_p == 2400 else "Metalic"
                    results_table.append([number, material, rho_p, d, rho_f, f"{nu:.2e}", f"{V:.2e}"])
                    number += 1

    print(tabulate(results_table, headers=headers, tablefmt="github"))


