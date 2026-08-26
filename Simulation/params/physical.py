from numba import float64, int64, types
from numba.experimental import jitclass

phys_spec = [
      ('g', float64),
      
      ('rho_1', float64),
      ('rho_2', float64),

      ('nu', float64),
      ('zeta', float64),
      ('A_12', float64),
      ('D', float64),

      ('C_1D', float64),
      ('C_2D', float64),
      ('C_A', float64),
      ('d', float64),

      ('x', float64),
      ('y', float64),
]

@jitclass(phys_spec)
class PhysicalParams:
      def __init__(self, rho_1, rho_2, nu, zeta, A_12, D, C_1D, C_2D, C_A, d):
            self.g = -9.82
            self.rho_1 = rho_1
            self.rho_2 = rho_2

            self.nu = nu
            self.zeta = zeta
            self.A_12 = A_12
            self.D = D

            self.C_1D = C_1D
            self.C_2D = C_2D
            self.C_A = C_A

            self.d = d

            self.x = rho_2 / rho_1
            self.y = 1.0 - self.x