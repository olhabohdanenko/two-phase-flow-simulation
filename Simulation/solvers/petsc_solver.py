import numpy as np
from petsc4py import PETSc

class PETScSolver:
    def __init__(self, M, N, dx, dz, rtol, atol, max_iter):
        self.M = M
        self.N = N

        self.dx = dx
        self.dz = dz
        
        self.da, self.A, self.ksp, self.pc, self.b, self.x_vec = self._init_petsc(rtol, atol, max_iter)

    def _init_petsc(self, rtol, atol, max_iter):
        da = PETSc.DMDA().create(dim=2, sizes=(self.M, self.N), dof=1, stencil_width=1)
        da.setUp()

        A = self.build_laplasian(da)

        b = da.createGlobalVec()
        x_vec = da.createGlobalVec()

        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        ksp.setType('fgmres')

        pc = ksp.getPC()
        # pc.setType('lu')
        pc.setType('hypre')
        pc.setHYPREType('boomeramg')

        # --------------- ГЛОБАЛЬНІ ОПЦІЇ ---------------
        opts = PETSc.Options()
        opts.setValue('-ksp_rtol', rtol)
        opts.setValue('-ksp_atol', atol)
        opts.setValue('-ksp_max_it', max_iter)
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

    def update(self, beta, y):
        self.A.zeroEntries()
        self.fill_operator(self.A, beta, y)
        self.A.assemblyBegin()
        self.A.assemblyEnd()
        

    def fill_operator(self, A, beta, y):
        (xs, xe), (ys, ye) = self.da.getRanges()

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
                    beta_w = 0.5 * (beta[i, j] + beta[i + 1, j])
                    beta_e = 0.5 * (beta[i - 1, j] + beta[i, j])
                    beta_n = 0.5 * (beta[i, j] + beta[i, j + 1])
                    beta_s = 0.5 * (beta[i, j - 1] + beta[i, j])

                    A.setValueStencil(row, row, (2.0 - y * (beta_w + beta_e)) / dx2 + (2.0 - y * (beta_n + beta_s))/ dz2)

                    col.i = i - 1;
                    col.j = j;
                    A.setValueStencil(row, col, -(1.0 - y * beta_e) / dx2)

                    col.i = i + 1;
                    col.j = j;
                    A.setValueStencil(row, col, -(1.0 - y * beta_w) / dx2)

                    col.i = i;
                    col.j = j - 1;
                    A.setValueStencil(row, col, -(1.0 - y * beta_s) / dz2)

                    col.i = i;
                    col.j = j + 1;
                    A.setValueStencil(row, col, -(1.0 - y * beta_n) / dz2)

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