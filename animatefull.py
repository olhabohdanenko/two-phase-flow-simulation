import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Simulation.utils import compute_vorticity_3d

H5_FILE = "run_20260914_144529/simulation_data_step_0.h5"
from Simulation.params import GridParams


def create_animation(h5_path, gif_path="simulation_beta_2d4.gif", animation = "beta", fps=15):

    if not os.path.exists(h5_path):
        print(f"Помилка: Файл {h5_path} не знайдено!")
        return

    with h5py.File(h5_path, "r") as f:
        if animation not in f:
            print("Помилка: Датасет '{animation}' відсутній у файлі!")
            return

        beta_dataset = f[animation]
        num_steps = beta_dataset.shape[0]
        grid_shape = beta_dataset.shape[1:]  # (M, N)

        print(f"Знайдено {num_steps} кроків. Розмір сітки: {grid_shape[0]}x{grid_shape[1]}")

        print("Сканування меж значень (min/max)...")
        global_min = float("inf")
        global_max = float("-inf")
        valid_indices = []

        for i in range(num_steps):
            try:
                step_data = beta_dataset[i]

                if np.isnan(step_data).any() or np.isinf(step_data).any():
                    print(f"Попередження: Крок {i} містить NaN/Inf. Пропускаємо.")
                    continue

                s_min = np.min(step_data)
                s_max = np.max(step_data)

                if s_min < global_min:
                    global_min = s_min
                if s_max > global_max:
                    global_max = s_max

                valid_indices.append(i)

            except OSError as e:
                print(f"Зауваження: Помилка читання кроку {i} ({e}). Зупинка сканування.")
                break

        if not valid_indices:
            print("Не вдалося зчитати жодного коректного кадру!")
            return

        print(f"Успішно зчитано {len(valid_indices)} кадрів. Діапазон {animation}: [{global_min:.4f}, {global_max:.4f}]")

        fig, ax = plt.subplots(figsize=(12, 2.3))

        first_frame = beta_dataset[valid_indices[0]]

        im = ax.imshow(
            first_frame.T,  # Транспонуємо (N, M)
            origin="lower",
            cmap="viridis",
            vmin=global_min,
            vmax=global_max,
            aspect="equal"
        )

        contours = [ax.contour(first_frame.T, colors='white', alpha=0.4, levels=8, origin="lower")]

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(animation)
        title_text = ax.set_title(f"Step: {valid_indices[0]}")
        ax.set_xlabel("X (M)")
        ax.set_ylabel("Z (N)")

        def update(frame_idx):
            actual_step = valid_indices[frame_idx]
            frame_data = beta_dataset[actual_step]
            
            im.set_array(frame_data.T)
            
            if contours[0] is not None:
                if hasattr(contours[0], 'remove'):
                    contours[0].remove()
                elif hasattr(contours[0], 'collections'):
                    for c in contours[0].collections:
                        c.remove()

            contours[0] = ax.contour(frame_data.T, colors='white', alpha=0.4, levels=8, origin="lower")

            title_text.set_text(f"Step: {actual_step}")
            return [im, title_text]


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
    """
    bg_field: назва фонового скалярного поля (наприклад, 'omega' або 'u')
    step: крок прорідження сітки для стрілочок (що більше значення, тим рідші стрілки)
    """
    M, N = 204, 38  # Кількість комірок по горизонталі/вертикалі (з 2-ма заграничними з кожного боку)
    dx, dz = 0.0025, 0.0025  # Кроки по сітці по горизонтальному/вертикальному напрямах (без урахування 4 заграничних)

    # Створення екземпляру класа сітка
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
        grid_shape = bg_ds.shape[1:]  # (M, N)

        print(f"Знайдено {num_steps} кроків. Розмір сітки: {grid_shape[0]}x{grid_shape[1]}")

        print(f"Сканування меж значений для {bg_field}...")
        global_min = float("inf")
        global_max = float("-inf")
        valid_indices = []

        for i in range(num_steps):
            try:
                bg_step = bg_ds[i]
                u_step = u_ds[i]
                w_step = w_ds[i]

                if any(np.isnan(arr).any() or np.isinf(arr).any() for arr in [bg_step, u_step, w_step]):
                    print(f"Попередження: Крок {i} містить NaN/Inf. Пропускаємо.")
                    continue

                s_min, s_max = np.min(bg_step), np.max(bg_step)
                if s_min < global_min: global_min = s_min
                if s_max > global_max: global_max = s_max

                valid_indices.append(i)

            except OSError as e:
                print(f"Зауваження: Помилка читання кроку {i} ({e}). Зупинка сканування.")
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
            cmap="coolwarm" if bg_field != "omega" else "viridis",
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
    save = "2"
    for var in ["beta"]:
        gif_path = f"simulation_{var}_2d{save}_contour.gif"
        create_animation(H5_FILE, gif_path, var, fps=15)

    create_vector_animation(H5_FILE, gif_path=f"simulation_vectors_{save}.gif", bg_field="omega", fps=15, step=4)
