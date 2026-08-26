from numba import float64, int64, types
from numba.experimental import jitclass

grid_spec = [
      ('M', int64),
      ('N', int64),

      ('dx', float64),
      ('dz', float64),
]

@jitclass(grid_spec)
class GridParams:
      def __init__(self, M, N, dx, dz):
            self.M = M
            self.N = N

            self.dx = dx
            self.dz = dz