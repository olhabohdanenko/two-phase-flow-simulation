import numpy as np
from petsc4py import PETSc

class PETScSolver:
    def __init__(self, M, N, dx, dz):
        self.M = M
        self.N = N

        self.dx = dx
        self.dz = dz
        
        self.da, self.A, self.ksp, self.pc, self.b, self.x_vec = self._init_petsc()

    def _init_petsc(self):
        da = PETSc.DMDA().create(dim=2, sizes=(self.M, self.N), dof=1, stencil_width=1)
        da.setUp()

        A = self.build_laplasian(da)

        b = da.createGlobalVec()
        x_vec = da.createGlobalVec()

        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        ksp.setType('fgmres')

        pc = ksp.getPC()
        pc.setType('hypre')
        pc.setHYPREType('boomeramg')

        # --------------- ГЛОБАЛЬНІ ОПЦІЇ ---------------
        opts = PETSc.Options()
        opts.setValue('-ksp_rtol', '1e-6')
        opts.setValue('-ksp_atol', '1e-8')
        opts.setValue('-ksp_max_it', '1500')
        # -----------------------------------------------

        ksp.setFromOptions()
        ksp.setUp()

        return da, A, ksp, pc, b, x_vec

    def solve(self, rhs, p):
        with self.da.getVecArray(self.b) as arr:
            arr[:, :] = 0.0
            arr[2:-2, 2:-2] = rhs[2:-2, 2:-2]


        self.ksp.solve(self.b, self.x_vec)

        with self.da.getVecArray(self.x_vec) as arr_x:
            p[:, :] = arr_x[:, :]

        return p

    def build_laplasian(self, da):
        A = da.createMatrix()
        A.setType('aij')
        (xs, xe), (ys, ye) = da.getRanges()

        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()

        dx2, dz2 = self.dx**2, self.dz**2

        for i in range(xs, xe):
            for j in range(ys, ye):
                row.i = i
                row.j = j

                if i == 0 or i == self.M - 1 or j == 0 or j == self.N - 1:
                    A.setValueStencil(row, row, 1.0)
                    continue

                # Права грань (Outlet) -> p = 0 (Діріхле)
                elif i == self.M - 2:
                    A.setValueStencil(row, row, 1.0)
                    continue
                # Ліва грань (Inlet) -> Нейман dp/dx = 0
                elif i == 1:
                    A.setValueStencil(row, row, -1.0 / dx2)
                    col.i = i + 1;
                    col.j = j
                    A.setValueStencil(row, col, 1.0 / dx2)
                    continue
                # Нижня грань -> Нейман dp/dz = 0
                elif j == 1:
                    A.setValueStencil(row, row, -1.0 / dz2)
                    col.i = i;
                    col.j = j + 1
                    A.setValueStencil(row, col, 1.0 / dz2)
                    continue
                # Верхня грань -> Нейман dp/dz = 0
                elif j == self.N - 2:
                    A.setValueStencil(row, row, -1.0 / dz2)
                    col.i = i;
                    col.j = j - 1
                    A.setValueStencil(row, col, 1.0 / dz2)
                    continue

                # 3. ВНУТРІШНІ ТОЧКИ - Лапласіан
                else:
                    A.setValueStencil(row, row, 2.0 / dx2 + 2.0 / dz2)
                    col.i = i - 1;
                    col.j = j;
                    A.setValueStencil(row, col, -1.0 / dx2)
                    col.i = i + 1;
                    col.j = j;
                    A.setValueStencil(row, col, -1.0 / dx2)
                    col.i = i;
                    col.j = j - 1;
                    A.setValueStencil(row, col, -1.0 / dz2)
                    col.i = i;
                    col.j = j + 1;
                    A.setValueStencil(row, col, -1.0 / dz2)

        A.assemblyBegin()
        A.assemblyEnd()
        return A  