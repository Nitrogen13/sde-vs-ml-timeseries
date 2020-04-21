import numpy as np
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from OrnsteinUhlenbeck.n_linear_ou import OrnsteinUhlenbeckModel
from utils import mle_ou

N = 1
t = np.arange(0, 100, 0.01)

param_grid = {'kappa': [0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0],
              'theta_a': [0.000001, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.01, 0.1],
              "theta_b": [0.005, 0.01, 0.02, 0.05, 0.1, 0.5],
              "sigma": [0.5, 2.0]}

# param_grid = {'kappa': [0.01],
#               'theta_a': [0.000001],
#               "theta_b": [0.5],
#               "sigma": [0.2, 0.3]}

grid = ParameterGrid(param_grid)

import pandas as pd

df = pd.DataFrame(columns=["kappa", "theta_a", "theta_b", "sigma", "error_kappa", "error_theta_a", "error_theta_b", "error_sigma"])

cls = OrnsteinUhlenbeckModel
for q in tqdm(grid):
    errors = {"error_kappa": 0, "error_theta_a": 0, "error_theta_b": 0, "error_sigma": 0}
    for i in range(N):

        kappa = q["kappa"]
        theta_a = q["theta_a"]
        theta_b = q["theta_b"]
        sigma = q["sigma"]

        x = cls(0.0, kappa, theta_a, theta_b, sigma).path(t)

        result = mle_ou(cls, t, x)
        print(result.status)
        for param in param_grid.keys():
            errors[f"error_{param}"] += abs(q[param] - result.params[param].value) / q[param] * 100
    for param in errors.keys():
        errors[param] = errors[param]/N  # mean percent error

    format_float = lambda x: '{:.5f}'.format(x)
    df = df.append({**q, **{k: format_float(v) for k, v in errors.items()}}, ignore_index=True)


df.to_csv("percent_results2.csv")
print("DONE")

