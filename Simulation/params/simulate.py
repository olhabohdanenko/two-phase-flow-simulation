from numba import float64, int64, types
from numba.experimental import jitclass

import numpy as np

sim_spec = [
      ('tau', float64),
      ('Re', float64),

      ('V_x', float64),
      ('V_z', float64),

      ('step', int64),
      ('step_max', int64),
      ('step_inj', int64),

      ('nozzle_positions', float64[:]),
      ('nozzle_r', float64),
]

@jitclass(sim_spec)
class SimulateParams:
      def __init__(self, tau, Re, V_x, V_z, step_max, step_inj, nozzle_positions, nozzle_r):
            self.tau = tau
            self.Re = Re

            self.V_x = V_x
            self.V_z = V_z

            self.step = 0
            self.step_max = step_max
            self.step_inj = step_inj

            self.nozzle_positions = np.array(nozzle_positions, dtype=np.float64)
            self.nozzle_r = nozzle_r