import sys
import numpy as np
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_physics_constants():
	nu_1 = 1e-4
	return nu_1

class ImplicitCoupledSystem:
	def __init__(self, M, N, dx, dz, tau):
		self.M = M
		self.N = N
		self.dx = dx
		self.dz = dz
		self.tau = tau
		self.nu_1 = get_physics_constants()

		self.da = PETSc.DMDA().create(dim=2, sizes=[M, N], dof=3, stencil_width=1, stencil_type=PETSc.DMDA.StencilType.BOX)
		self.da.setUp()
		self.local_X = self.da.createLocalVec()

	def stencil(self, i, j, c):
		s = PETSc.Mat.Stencil()
		s.index = (i, j)
		s.field = c
		return s

	def formFunction(self, snes, X, F):
		F.zeroEntries()
		self.da.globalToLocal(X, self.local_X)
		x_arr = self.da.getVecArray(self.local_X)
		f_arr = self.da.getVecArray(F)

		(xs, xe), (ys, ye) = self.da.getRanges()
		dx, dz, tau = self.dx, self.dz, self.tau
		nu_0 = self.nu_1

		for i in range(xs, xe):
			for j in range(ys, ye):
				if i <= 1 or i >= self.M - 2 or j <= 1 or j >= self.N - 2:
					for c in range(3):
						f_arr[i, j, c] = x_arr[i, j, c]
					continue

				u_c = x_arr[i, j, 0]
				u_r, u_l = x_arr[i + 1, j, 0], x_arr[i - 1, j, 0]
				u_t, u_d = x_arr[i, j + 1, 0], x_arr[i, j - 1, 0]
				u_tr, u_tl = x_arr[i + 1, j + 1, 0], x_arr[i - 1, j + 1, 0]

				w_c = x_arr[i, j, 1]
				w_r, w_l = x_arr[i + 1, j, 1], x_arr[i - 1, j, 1]
				w_t, w_d = x_arr[i, j + 1, 1], x_arr[i, j - 1, 1]
				w_dr, w_dl = x_arr[i + 1, j - 1, 1], x_arr[i - 1, j - 1, 1]

				p_r, p_c_x = x_arr[i + 1, j, 2], x_arr[i, j, 2]
				p_t, p_c_z = x_arr[i, j + 1, 2], x_arr[i, j, 2]

				nu = nu_0

				# Рядок рівняння F_u

				term_time = u_c / tau # +
				term_grad_p = (p_r - p_c_x) / dx # +
				term_adv = u_c * (u_r - u_l) / (2.0 * dx) + 0.25 * (w_c + w_r + w_d + w_dr) * (u_t - u_d) / (2.0 * dz) # +

				term_diff_x = nu * (u_r - 2.0 * u_c + u_l) / (dx**2)# +
				term_diff_z = nu * (u_t - 2.0 * u_c + u_d) / (dz**2) # +

				f_arr[i, j, 0] = term_time + term_grad_p + term_adv - term_diff_x - term_diff_z

				# Рядок рівняння F_w

				term_time = w_c / tau # +
				term_grad_p = (p_t - p_c_z) / dz # +
				term_adv = w_c * (w_t - w_d) / (2.0 * dz) + 0.25 * (u_c + u_t + u_l + u_tl) * (w_r - w_l) / (2.0 * dx) # +

				term_diff_x = nu * (w_r - 2.0 * w_c + w_l) / (dx**2)# +
				term_diff_z = nu * (w_t - 2.0 * w_c + w_d) / (dz**2) # +

				f_arr[i, j, 1] = term_time + term_grad_p + term_adv - term_diff_x - term_diff_z

				# Рядок рівняння F_p

				term_grad_p = (u_c - u_l) / dx + (w_c - w_d) / dz # +

				f_arr[i, j, 2] = term_grad_p

		F.assemble()

	def formJacobian(self, snes, X, J, P):
		print("=== FORM JAC START ===")

		J.zeroEntries()
		self.da.globalToLocal(X, self.local_X)

		# Отримуємо багатовимірний доступ до поточного наближення
		x_arr = self.da.getVecArray(self.local_X)

		(xs, xe), (ys, ye) = self.da.getRanges()

		# Кешуємо константи для швидкості доступу
		dx, dz, tau = self.dx, self.dz, self.tau
		dx2, dz2 = dx ** 2, dz ** 2
		nu_0 = self.nu_1

		for i in range(xs, xe):
			for j in range(ys, ye):
				# Рядок рівняння F_u для вузла (i, j)
				row_Fu = self.stencil(i=i, j=j, c=0)
				row_Fw = self.stencil(i=i, j=j, c=1)
				row_Fp = self.stencil(i=i, j=j, c=2)

				# Пропускаємо границі
				if i <= 1 or i >= self.M - 2 or j <= 1 or j >= self.N - 2:
					# На границях Якобіан = одиничній матриці (якщо ГУ Діріхле)
					J.setValueStencil(row_Fu, row_Fu, 1.0, PETSc.InsertMode.ADD_VALUES)
					J.setValueStencil(row_Fw, row_Fw, 1.0, PETSc.InsertMode.ADD_VALUES)
					J.setValueStencil(row_Fp, row_Fp, 1.0, PETSc.InsertMode.ADD_VALUES)
					continue

				# Локальні значення змінних
				u_c = x_arr[i, j, 0]
				u_r, u_l = x_arr[i + 1, j, 0], x_arr[i - 1, j, 0]
				u_t, u_d = x_arr[i, j + 1, 0], x_arr[i, j - 1, 0]
				u_tr, u_tl = x_arr[i + 1, j + 1, 0], x_arr[i - 1, j + 1, 0]

				w_c = x_arr[i, j, 1]
				w_r, w_l = x_arr[i + 1, j, 1], x_arr[i - 1, j, 1]
				w_t, w_d = x_arr[i, j + 1, 1], x_arr[i, j - 1, 1]
				w_dr, w_dl = x_arr[i + 1, j - 1, 1], x_arr[i - 1, j - 1, 1]

				p_r, p_c_x = x_arr[i + 1, j, 2], x_arr[i, j, 2]
				p_t, p_c_z = x_arr[i, j + 1, 2], x_arr[i, j, 2]

				nu = nu_0

				# Рядок рівняння F_u

				# =================================================================
				# 1. ПОХІДНІ dF_u / d_u (c = 0)
				# =================================================================
				idx1 = idx2 = idx3 = idx4 = idx5 = idx6 = idx7 = idx8 = idx9 = 0.0

				# time +
				idx5 += 1.0 / tau # +

				# advection +
				idx5 += (u_r - u_l) / (2.0 * dx)  # center
				idx6 += u_c / (2.0 * dx)  # right
				idx4 += -u_c / (2.0 * dx)  # left

				idx8 += (w_c + w_r + w_d + w_dr) / (8.0 * dz)  # top
				idx2 += -(w_c + w_r + w_d + w_dr) / (8.0 * dz)  # down

				# diffusion +
				idx5 += 2.0 * nu * (1.0 / dx2 + 1.0 / dz2)

				idx6 += - nu / dx2

				idx4 += - nu / dx2

				idx8 += - nu / dz2

				idx2 += - nu / dz2

				# --- ВСТАВКА dF_u / d_u В МАТРИЦЮ (c=0) ---
				u_stencil_entries = [
					(i - 1, j - 1, idx1), (i, j - 1, idx2), (i + 1, j - 1, idx3),
					(i - 1, j, idx4), (i, j, idx5), (i + 1, j, idx6),
					(i - 1, j + 1, idx7), (i, j + 1, idx8), (i + 1, j + 1, idx9)
				]
				for (ni, nj, val) in u_stencil_entries:
					if val != 0.0:
						J.setValueStencil(row_Fu, self.stencil(i=ni, j=nj, c=0), val,
										PETSc.InsertMode.ADD_VALUES)

				# =================================================================
				# 2. ПОХІДНІ dF_u / d_w (c = 1)
				# =================================================================
				idx1 = idx2 = idx3 = idx4 = idx5 = idx6 = idx7 = idx8 = idx9 = 0.0

				# advection +
				idx5 += (u_t - u_d) / (8.0 * dz)  # center
				idx6 += (u_t - u_d) / (8.0 * dz)  # right
				idx2 += (u_t - u_d) / (8.0 * dz)  # down
				idx3 += (u_t - u_d) / (8.0 * dz)  # down right

				# --- ВСТАВКА dF_u / d_w В МАТРИЦЮ (c=1) ---
				w_stencil_entries = [
					(i - 1, j - 1, idx1), (i, j - 1, idx2), (i + 1, j - 1, idx3),
					(i - 1, j, idx4), (i, j, idx5), (i + 1, j, idx6),
					(i - 1, j + 1, idx7), (i, j + 1, idx8), (i + 1, j + 1, idx9),
				]
				for (ni, nj, val) in w_stencil_entries:
					if val != 0.0:
						J.setValueStencil(row_Fu, self.stencil(i=ni, j=nj, c=1), val,
										PETSc.InsertMode.ADD_VALUES)

				# =================================================================
				# 3. ПОХІДНІ dF_u / d_p (c = 2)
				# =================================================================
				idx1 = idx2 = idx3 = idx4 = idx5 = idx6 = idx7 = idx8 = idx9 = 0.0
				idx5_p = - 1.0 / dx  # center
				idx6_p = 1.0 / dx  # right

				J.setValueStencil(row_Fu, self.stencil(i=i, j=j, c=2), idx5_p, PETSc.InsertMode.ADD_VALUES)
				J.setValueStencil(row_Fu, self.stencil(i=i + 1, j=j, c=2), idx6_p, PETSc.InsertMode.ADD_VALUES)

				# Рядок рівняння F_w

				u_c = x_arr[i, j, 0]
				u_r, u_l = x_arr[i + 1, j, 0], x_arr[i - 1, j, 0]
				u_t, u_d = x_arr[i, j + 1, 0], x_arr[i, j - 1, 0]
				u_tr, u_tl = x_arr[i + 1, j + 1, 0], x_arr[i - 1, j + 1, 0]

				w_c = x_arr[i, j, 1]
				w_r, w_l = x_arr[i + 1, j, 1], x_arr[i - 1, j, 1]
				w_t, w_d = x_arr[i, j + 1, 1], x_arr[i, j - 1, 1]
				w_dr, w_dl = x_arr[i + 1, j - 1, 1], x_arr[i - 1, j - 1, 1]

				p_r, p_c_x = x_arr[i + 1, j, 2], x_arr[i, j, 2]
				p_t, p_c_z = x_arr[i, j + 1, 2], x_arr[i, j, 2]

				nu = nu_0

				# =================================================================
				# 1. ПОХІДНІ dF_w / d_u (c = 0)
				# =================================================================
				idx1 = idx2 = idx3 = idx4 = idx5 = idx6 = idx7 = idx8 = idx9 = 0.0

				# advection +
				idx5 += (w_r - w_l) / (8.0 * dx)  # center
				idx8 += (w_r - w_l) / (8.0 * dx)  # top
				idx4 += (w_r - w_l) / (8.0 * dx)  # left
				idx7 += (w_r - w_l) / (8.0 * dx)  # top left

				# --- ВСТАВКА dF_w / d_u В МАТРИЦЮ (c=1) ---
				u_stencil_entries = [
					(i - 1, j - 1, idx1), (i, j - 1, idx2), (i + 1, j - 1, idx3),
					(i - 1, j, idx4), (i, j, idx5), (i + 1, j, idx6),
					(i - 1, j + 1, idx7), (i, j + 1, idx8), (i + 1, j + 1, idx9)
				]
				for (ni, nj, val) in u_stencil_entries:
					if val != 0.0:
						J.setValueStencil(row_Fw, self.stencil(i=ni, j=nj, c=0), val,
										PETSc.InsertMode.ADD_VALUES)

				# =================================================================
				# 2. ПОХІДНІ dF_w / d_w (c = 1)
				# =================================================================
				idx1 = idx2 = idx3 = idx4 = idx5 = idx6 = idx7 = idx8 = idx9 = 0.0

				# time +
				idx5 += 1.0 / tau # +

				# advection +
				idx5 += (w_t - w_d) / (2.0 * dz)  # center
				idx8 += w_c / (2.0 * dz)  # top
				idx2 += -w_c / (2.0 * dz)  # down

				idx6 += (u_c + u_t + u_l + u_tl) / (8.0 * dx)  # right
				idx4 += -(u_c + u_t + u_l + u_tl) / (8.0 * dx)  # left

				# diffusion +
				idx5 += 2.0 * nu * (1.0 / dx2 + 1.0 / dz2)

				idx6 += - nu / dx2

				idx4 += - nu / dx2

				idx8 += - nu / dz2

				idx2 += - nu / dz2

				# --- ВСТАВКА dF_w / d_w В МАТРИЦЮ (c=1) ---
				w_stencil_entries = [
					(i - 1, j - 1, idx1), (i, j - 1, idx2), (i + 1, j - 1, idx3),
					(i - 1, j, idx4), (i, j, idx5), (i + 1, j, idx6),
					(i - 1, j + 1, idx7), (i, j + 1, idx8), (i + 1, j + 1, idx9),
				]
				for (ni, nj, val) in w_stencil_entries:
					if val != 0.0:
						J.setValueStencil(row_Fw, self.stencil(i=ni, j=nj, c=1), val,
										PETSc.InsertMode.ADD_VALUES)

				# =================================================================
				# 3. ПОХІДНІ dF_w / d_p (c = 2)
				# =================================================================
				idx1 = idx2 = idx3 = idx4 = idx5 = idx6 = idx7 = idx8 = idx9 = 0.0
				idx5_p = - 1.0 / dz  # center
				idx8_p = 1.0 / dz  # top

				J.setValueStencil(row_Fw, self.stencil(i=i, j=j, c=2), idx5_p, PETSc.InsertMode.ADD_VALUES)
				J.setValueStencil(row_Fw, self.stencil(i=i, j=j + 1, c=2), idx8_p, PETSc.InsertMode.ADD_VALUES)

				# Рядок рівняння F_p

				u_c = x_arr[i, j, 0]
				u_r, u_l = x_arr[i + 1, j, 0], x_arr[i - 1, j, 0]
				u_t, u_d = x_arr[i, j + 1, 0], x_arr[i, j - 1, 0]
				u_tr, u_tl = x_arr[i + 1, j + 1, 0], x_arr[i - 1, j + 1, 0]
				u_dr, u_dl = x_arr[i + 1, j - 1, 0], x_arr[i - 1, j - 1, 0]

				w_c = x_arr[i, j, 1]
				w_r, w_l = x_arr[i + 1, j, 1], x_arr[i - 1, j, 1]
				w_t, w_d = x_arr[i, j + 1, 1], x_arr[i, j - 1, 1]
				w_tr, w_tl = x_arr[i + 1, j + 1, 1], x_arr[i - 1, j + 1, 1]
				w_dr, w_dl = x_arr[i + 1, j - 1, 1], x_arr[i - 1, j - 1, 1]

				p_r, p_c_x = x_arr[i + 1, j, 2], x_arr[i, j, 2]
				p_t, p_c_z = x_arr[i, j + 1, 2], x_arr[i, j, 2]

				# =================================================================
				# 1. ПОХІДНІ dF_p / d_u (c = 0)
				# =================================================================
				idx1 = idx2 = idx3 = idx4 = idx5 = idx6 = idx7 = idx8 = idx9 = 0.0
				idx5_p = 1.0 / dx  # center
				idx4_p = - 1.0 / dx  # left

				J.setValueStencil(row_Fp, self.stencil(i=i, j=j, c=0), idx5_p, PETSc.InsertMode.ADD_VALUES)
				J.setValueStencil(row_Fp, self.stencil(i=i - 1, j=j, c=0), idx4_p, PETSc.InsertMode.ADD_VALUES)

				# =================================================================
				# 1. ПОХІДНІ dF_p / d_w (c = 1)
				# =================================================================
				idx1 = idx2 = idx3 = idx4 = idx5 = idx6 = idx7 = idx8 = idx9 = 0.0
				idx5_p = 1.0 / dz  # center
				idx2_p = - 1.0 / dz  # down

				J.setValueStencil(row_Fp, self.stencil(i=i, j=j, c=1), idx5_p, PETSc.InsertMode.ADD_VALUES)
				J.setValueStencil(row_Fp, self.stencil(i=i, j=j - 1, c=1), idx2_p, PETSc.InsertMode.ADD_VALUES)

		J.assemble()
		print("J norm =", J.norm())
		J.view()	

		if J != P:
			P.assemble()

def main():
	M, N = 7, 7
	dx, dz, tau = 0.01, 0.01, 1e-3

	system = ImplicitCoupledSystem(M, N, dx, dz, tau)

	x_vec = system.da.createGlobalVec()
	f_vec = system.da.createGlobalVec()

	J_mat = system.da.createMatrix()
	P_mat = system.da.createMatrix()

	x_vec.setRandom()
	x_vec.abs()
	x_vec.scale(0.1)

	snes = PETSc.SNES().create()
	snes.setDM(system.da)
	snes.setFunction(system.formFunction, f_vec)
	snes.setJacobian(system.formJacobian, J_mat, P_mat)
	
	# Увімкнення читання аргументів командного рядка (-snes_test_jacobian)
	snes.setFromOptions()

	# Перевірка через вбудований механізм PETSc
	snes.solve(None, x_vec)

	# system.formJacobian(snes, x_vec, J_mat, P_mat)

if __name__ == "__main__":
	main()