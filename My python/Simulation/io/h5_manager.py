import os
import h5py

import numpy as np

class HDF5Writer:
    def __init__(self, save_dir, shape_spatial, field_names=None, step_create=10000, compression="gzip"):
        self.save_dir = save_dir
        self.M, self.N = shape_spatial
        self.field_names = field_names if field_names else ["u", "w", "omega"]
        self.step_create = step_create
        self.compression = compression
        
        self.file = None
        self.datasets = {}
        self.current_step_create = -1

    def __enter__(self):
        return self

    def _open_new_file(self, step_num):
        if self.file is not None:
            self.file.close()

        self.current_step_create = step_num
        h5_path = os.path.join(self.save_dir, f"simulation_data_step_{step_num}.h5")
        self.file = h5py.File(h5_path, "w")
        self.datasets = {}

        for name in self.field_names:
            self.datasets[name] = self.file.create_dataset(
                name,
                shape=(0, self.M, self.N),
                maxshape=(None, self.M, self.N),
                dtype=np.float64,
                compression=self.compression,
                chunks=(1, self.M, self.N)
            )

    def write_step(self, step_num, data_dict):
        target_file_step = (step_num // self.step_create) * self.step_create
        
        if self.file is None or target_file_step != self.current_step_create:
            self._open_new_file(target_file_step)

        for name, array in data_dict.items():
            if name in self.datasets:
                dset = self.datasets[name]
                curr_len = dset.shape[0]
                dset.resize(curr_len + 1, axis=0)
                dset[curr_len, :, :] = array

        self.file.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file is not None:
            self.file.close()
            self.file = None