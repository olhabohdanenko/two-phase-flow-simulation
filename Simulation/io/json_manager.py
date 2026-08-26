import os
import json

import numpy as np

class JSONLogger:
    def __init__(self, save_dir, filename="simulation_log.json"):
        self.filepath = os.path.join(save_dir, filename)
        self.logs = []

    def __enter__(self):
        return self

    def log_step(self, step, u_n1, w_n1, p, beta_n1, Div):
        max_u = float(np.max(np.abs(u_n1)))
        max_w = float(np.max(np.abs(w_n1)))
        max_p = float(np.max(np.abs(p)))
        max_beta = float(np.max(beta_n1))
        max_div = float(np.max(np.abs(Div)))
        
        is_exploded = bool(np.isnan(Div).any() or np.isinf(Div).any())

        log_entry = {
            "step": int(step),
            "max_u": max_u,
            "max_w": max_w,
            "max_p": max_p,
            "max_beta": max_beta,
            "max_div": max_div,
            "status": "EXPLODED" if is_exploded else "OK"
        }
        
        self.logs.append(log_entry)

        print(f"Крок {step}: max(u)={max_u:.6e}, max(w)={max_w:.6e}, max(p)={max_p:.6e}, max(beta)={max_beta:.6e}, max(Div)={max_div:.6e}")
        
        if is_exploded:
            print(f"ВИБУХ ДИВЕРГЕНЦІЇ на кроці {step}!")

        return is_exploded

    def _flush(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=4, ensure_ascii=False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._flush()