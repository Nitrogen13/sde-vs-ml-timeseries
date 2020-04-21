import math

import numpy as np
import scipy.stats
from lmfit import minimize, Parameters


def mle_ou(cls, t, s):
    """Maximum-likelihood estimator for standard OU"""

    def log_likelihood(q):
        """Calculates log likelihood of a standard OU path"""
        K = q["kappa"]
        theta_a = q["theta_a"]
        theta_b = q["theta_b"]
        sigma = q["sigma"]
        dt = np.diff(t)
        mu: np.array = cls.mean(
            s[:-1], t[:-1], dt, K, theta_a, theta_b
        )
        sigma = cls.std(dt, K, sigma)
        return -np.sum(scipy.stats.norm.logpdf(s[1:], loc=mu, scale=sigma))

    params = Parameters()
    params.add_many(
        ("kappa", 0.5, True, 1e-6, None),
        ("theta_a", 0.0001, True, 0.0, None),
        ("theta_b", np.mean(s), True, 0.0, None),
        ("sigma", est_sigma_quadratic_variation(t, s), True, 1e-8, None),
    )

    result = minimize(
        log_likelihood,
        params,
        method="lbfgsb",
        options={"maxiter": 50000, "disp": False},
    )
    return result


def standard_brownian_motion_path(t, x0=0.0):
    """ Simulates a sample path"""
    assert len(t) > 1
    x = scipy.stats.norm.rvs(size=len(t))
    x[0] = x0
    dt = np.diff(t)
    x[1:] = x[1:] * np.sqrt(dt)
    return np.cumsum(x)


def gaussian_path(x0, t, loc_fun, scale_fun, params):
    assert len(t) > 1
    x = scipy.stats.norm.rvs(size=len(t))
    x[0] = x0
    dt = np.diff(t)
    scale = scale_fun(dt, params)
    x[1:] = x[1:] * scale
    for i in range(1, len(x)):
        x[i] += loc_fun(x[i - 1], dt[i - 1], params)
    return x


def est_sigma_quadratic_variation(t, x):
    """ Estimate sigma using quadratic variation"""
    assert len(t) == len(x)
    return math.sqrt(quadratic_variation(x) / (t[-1] - t[0]))


def quadratic_variation(x):
    assert len(x) > 1
    dx = np.diff(x)
    return np.sum(dx * dx)
