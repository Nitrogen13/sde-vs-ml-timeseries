from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import scipy.optimize
import scipy.stats
from lmfit import minimize, Parameters

from utils import est_sigma_quadratic_variation

np.random.seed(42)

class VasicekModel:
    def __init__(self, s_0: float, kappa: float, theta: float, sigma: float):
        assert kappa >= 0
        assert sigma >= 0
        self.s_0 = s_0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def path(self, t: np.array):
        """ Simulates a sample path"""
        assert len(t) > 1
        dt: np.array = np.diff(t)
        s = np.random.normal(loc=0, scale=1.0, size=len(t))  # Wiener process increments
        s[0] = self.s_0
        scale = self.std(dt, self.kappa, self.sigma)
        s[1:] = s[1:] * scale
        for i in range(1, len(s)):
            s[i] += self.mean(s[i - 1], dt[i - 1], self.kappa, self.theta)
        return s

    @staticmethod
    def mean(
        s_t: Union[np.array, float],
        dt: Union[np.array, float],
        kappa: float,
        theta: float,
    ) -> float:
        """ Mean function of an OU process"""
        return s_t * np.exp(-kappa * dt) + (1.0 - np.exp(-kappa * dt)) * theta

    @staticmethod
    def variance(t: np.array, kappa: float, sigma: float):
        """ Variance function of an OU process"""
        return (sigma ** 2) * (1.0 - np.exp(-2.0 * kappa * t)) / (2 * kappa)

    @staticmethod
    def std(t: np.array, kappa: float, sigma: float):
        return np.sqrt(VasicekModel.variance(t, kappa, sigma))


def mle_ou(t, s):
    """Maximum-likelihood estimator for standard OU"""

    def log_likelihood(q):
        """Calculates log likelihood of a standard OU path"""
        K = q["kappa"]
        theta = q["theta"]
        sigma = q["sigma"]
        dt = np.diff(t)
        mu = VasicekModel.mean(s[:-1], dt, K, theta)
        sigma = VasicekModel.std(dt, K, sigma)
        return -np.sum(scipy.stats.norm.logpdf(s[1:], loc=mu, scale=sigma))

    params = Parameters()
    params.add_many(
        ("kappa", 0.5, True, 1e-6, None),
        ("theta", np.mean(s), True, 0.0, None),
        ("sigma", est_sigma_quadratic_variation(t, s), True, 1e-8, None),
    )

    result = minimize(
        log_likelihood,
        params,
        method="L-BFGS-B",
        options={"maxiter": 500, "disp": False},
    )
    return result


# simulate a path of the OU process on a given grid t, starting with x_0 = 0.8
# q = (S_0, init value
# time, timeline
# kappa, mean reversion speed
# theta, mean reversion level
# sigma) volatility

t = np.arange(0, 100, 0.01)
q = (0.0, 0.02, 0.05, 0.5)
ou = VasicekModel(*q)

x = ou.path(t)

plt.plot(t, x)
plt.show()

result = mle_ou(t, x)
print(result.status)
print(result.params)
