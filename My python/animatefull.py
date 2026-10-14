import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Simulation.utils import compute_vorticity_3d
from Simulation.params import GridParams

H5_FILE = "run_20261011_003146/simulation_data_step_0.h5"

def calculate_and_draw_particles(ax, u_grid, w_grid, grid_param, dt=1e-4, tau=1.0):
    eta = 0.001004     # Динамічна в'язкість середовища (Па·с)
    rho_p = 7800.0      # Щільність частинки (кг/м³)
    rho_f = 1000.0      # Щільність середовища (кг/м³)
    gravity = 9.81
    
    M, N = u_grid.shape
    dx, dz = grid_param.dx, grid_param.dz
    lenOfBox = getattr(grid_param, 'lx', dx * M)
    H = getattr(grid_param, 'lz', dz * N)

    lines = []

    d = 2.5e-4
    r = d / 2.0
    V = (4.0 / 3.0) * np.pi * (r ** 3)
    C = 6.0 * np.pi * r * eta
    M_p = V * rho_p

    x, y = 0.0, H
    vx, vy = 0.0, 0.0
    
    traj_x = [x]
    traj_y = [y]

    countIter = 50000

    # Обчислюємо траєкторію
    while 0 <= x <= lenOfBox and 0 <= y <= H and countIter > 0:
        middlePoint = int(1.0 / (r * tau)) if (r * tau) > 0 else 1
        if middlePoint < 1: 
            middlePoint = 1

        for _ in range(middlePoint):
            i = int(1 + (x / lenOfBox) * (M - 2))
            j = int(1 + (y / H) * (N - 2))

            i = max(0, min(i, M - 1))
            j = max(0, min(j, N - 1))

            ax_p = C * (u_grid[i, j] - vx) / M_p
            ay_p = ((rho_f - rho_p) * gravity * V + C * (w_grid[i, j] - vy)) / M_p

            vx += dt * ax_p
            x += dt * vx

            vy += dt * ay_p
            y += dt * vy

            traj_x.append(x)
            traj_y.append(y)

            countIter -= 1
            if not (0 <= x <= lenOfBox and 0 <= y <= H) or countIter <= 0:
                break

    # Малюємо траєкторію ОДИН РАЗ після завершення інтегрування
    grid_x = np.array(traj_x) / dx
    grid_y = np.array(traj_y) / dz

    line, = ax.plot(grid_x, grid_y, color='red', linewidth=1.5, alpha=0.8)
    lines.append(line)

    return lines


def create_animation(h5_path, gif_path="simulation_beta_2d4.gif", animation="beta", fps=15):
    if not os.path.exists(h5_path):
        print(f"Помилка: Файл {h5_path} не знайдено!")
        return

    with h5py.File(h5_path, "r") as f:
        # Перевірка наявності необхідних датасетів
        for req in [animation, "u", "w"]:
            if req not in f:
                print(f"Помилка: Датасет '{req}' відсутній у файлі!")
                return

        beta_dataset = f[animation]
        u_ds = f["u"]
        w_ds = f["w"]

        num_steps = beta_dataset.shape[0]
        grid_shape = beta_dataset.shape[1:]  # (M, N)
        M, N = grid_shape

        # Створення екземпляра GridParams для розрахунку частинок
        dx, dz = 0.0025, 0.0025
        grid_param = GridParams(M, N, dx, dz)

        print(f"Знайдено {num_steps} кроків. Розмір сітки: {M}x{N}")

        valid_indices = []
        global_min = float("inf")
        global_max = float("-inf")

        for i in range(num_steps):
            try:
                step_data = beta_dataset[i]
                if np.isnan(step_data).any() or np.isinf(step_data).any():
                    continue

                s_min, s_max = np.min(step_data), np.max(step_data)
                if s_min < global_min: global_min = s_min
                if s_max > global_max: global_max = s_max

                valid_indices.append(i)
            except OSError:
                break

        if not valid_indices:
            print("Не вдалося зчитати жодного коректного кадру!")
            return

        fig, ax = plt.subplots(figsize=(12, 2.3))

        first_frame = beta_dataset[valid_indices[0]]

        im = ax.imshow(
            first_frame.T,
            origin="lower",
            cmap="coolwarm",
            vmin=global_min,
            vmax=global_max,
            aspect="equal"
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(animation)
        title_text = ax.set_title(f"Step: {valid_indices[0]}")
        ax.set_xlabel("X (M)")
        ax.set_ylabel("Z (N)")

        # Масив для збереження намальованих ліній траєкторій
        particle_lines = []
        contours = [ax.contour(first_frame.T, colors='white', alpha=0.4, levels=8, origin="lower")]

        def update(frame_idx):
            nonlocal particle_lines
            actual_step = valid_indices[frame_idx]
            
            # Оновлюємо фонове скалярне поле
            frame_data = beta_dataset[actual_step]
            im.set_array(frame_data.T)

            # Видаляємо попередні лінії частинок
            for line in particle_lines:
                line.remove()
            particle_lines.clear()

            # Зчитуємо поля швидкостей для поточного кадру
            u_frame = u_ds[actual_step]
            w_frame = w_ds[actual_step]

            # Розраховуємо та малюємо траєкторії частинок
            particle_lines = calculate_and_draw_particles(ax, u_frame, w_frame, grid_param)

            if contours[0] is not None:
                if hasattr(contours[0], 'remove'):
                    contours[0].remove()
                elif hasattr(contours[0], 'collections'):
                    for c in contours[0].collections:
                        c.remove()

            contours[0] = ax.contour(frame_data.T, colors='white', alpha=0.4, levels=8, origin="lower")

            title_text.set_text(f"Step: {actual_step}")
            return [im, title_text] + particle_lines

        print(f"Генерація GIF ({gif_path})...")
        anim = FuncAnimation(
            fig,
            update,
            frames=len(valid_indices),
            interval=1000 // fps,
            blit=False
        )

        anim.save(gif_path, writer="pillow", fps=fps)
        plt.close(fig)
        print(f"Анімацію успішно збережено у {gif_path}!")


def create_vector_animation(h5_path, gif_path="simulation_vector_field.gif", bg_field="omega", fps=15, step=4):
    M, N = 204, 38
    dx, dz = 0.0025, 0.0025
    grid_param = GridParams(M, N, dx, dz)

    if not os.path.exists(h5_path):
        print(f"Помилка: Файл {h5_path} не знайдено!")
        return

    with h5py.File(h5_path, "r") as f:
        for req in ["u", "w"]:
            if req not in f:
                print(f"Помилка: Датасет '{req}' відсутній у файлі!")
                return

        u_ds = f["u"]
        w_ds = f["w"]
        u_arr = u_ds[:]
        w_arr = w_ds[:]
        bg_ds = compute_vorticity_3d(u_arr, w_arr, grid_param.dx, grid_param.dz)

        num_steps = min(u_ds.shape[0], w_ds.shape[0], bg_ds.shape[0])
        grid_shape = bg_ds.shape[1:]

        global_min = float("inf")
        global_max = float("-inf")
        valid_indices = []

        for i in range(num_steps):
            try:
                bg_step = bg_ds[i]
                u_step = u_ds[i]
                w_step = w_ds[i]

                if any(np.isnan(arr).any() or np.isinf(arr).any() for arr in [bg_step, u_step, w_step]):
                    continue

                s_min, s_max = np.min(bg_step), np.max(bg_step)
                if s_min < global_min: global_min = s_min
                if s_max > global_max: global_max = s_max

                valid_indices.append(i)

            except OSError:
                break

        if not valid_indices:
            print("Не вдалося зчитати жодного коректного кадру!")
            return

        fig, ax = plt.subplots(figsize=(12, 2.3))

        first_bg = bg_ds[valid_indices[0]]
        first_u = u_ds[valid_indices[0]]
        first_w = w_ds[valid_indices[0]]

        im = ax.imshow(
            first_bg.T,
            origin="lower",
            cmap="coolwarm" if bg_field != "omega" else "coolwarm",
            vmin=global_min,
            vmax=global_max,
            aspect="equal"
        )

        M_len, N_len = grid_shape
        X_coords, Z_coords = np.meshgrid(np.arange(M_len), np.arange(N_len))

        X_sub = X_coords[::step, ::step]
        Z_sub = Z_coords[::step, ::step]

        u_sub = first_u.T[::step, ::step]
        w_sub = first_w.T[::step, ::step]

        quiv = ax.quiver(
            X_sub, Z_sub, u_sub, w_sub,
            color='black',
            scale=None,
            pivot='mid',
            angles='xy',
            scale_units='xy',
            headwidth=3,
            headlength=4
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(bg_field)
        title_text = ax.set_title(f"Step: {valid_indices[0]}")
        ax.set_xlabel("X (M)")
        ax.set_ylabel("Z (N)")

        def update(frame_idx):
            actual_step = valid_indices[frame_idx]
            
            bg_data = bg_ds[actual_step]
            u_data = u_ds[actual_step]
            w_data = w_ds[actual_step]

            im.set_array(bg_data.T)

            u_sub_new = u_data.T[::step, ::step]
            w_sub_new = w_data.T[::step, ::step]
            quiv.set_UVC(u_sub_new, w_sub_new)

            title_text.set_text(f"Step: {actual_step}")
            return [im, quiv, title_text]

        print(f"Генерація GIF ({gif_path})...")
        anim = FuncAnimation(
            fig,
            update,
            frames=len(valid_indices),
            interval=1000 // fps,
            blit=False
        )

        anim.save(gif_path, writer="pillow", fps=fps)
        plt.close(fig)
        print(f"Векторну анімацію успішно збережено у {gif_path}!")

if __name__ == "__main__":
    save = "3"
    for var in ["beta"]:
        gif_path = f"simulation_{var}_2d{save}_contour.gif"
        create_animation(H5_FILE, gif_path, var, fps=15)

    create_vector_animation(H5_FILE, gif_path=f"simulation_vectors_{save}.gif", bg_field="omega", fps=15, step=4)
    # ffmpeg -framerate 15 -i animation.%004d.png output.gif
