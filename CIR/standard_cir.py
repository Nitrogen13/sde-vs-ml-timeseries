from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from lmfit import minimize, Parameters
from scipy.special import ive

from utils import est_sigma_quadratic_variation


class StationaryCIRModel:
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
        s = [self.s_0]
        for t in range(len(dt)):
            ds = (
                    self.kappa * (self.theta - s[-1]) * dt[0]
                    + self.sigma * np.sqrt(dt[0]) * np.sqrt(s[-1]) * np.random.normal()
            )
            if s[t] + ds >= 0:
                s.append(s[t] + ds)
            else:
                s.append(0.0000000001)
        return s
    #
    # @staticmethod
    # def mean(
    #         s_t: Union[np.array, float],
    #         dt: Union[np.array, float],
    #         kappa: float,
    #         theta: float,
    # ) -> float:
    #     """ Mean function of an OU process"""
    #     return s_t * np.exp(-kappa * dt) + (1.0 - np.exp(-kappa * dt)) * theta
    #
    # @staticmethod
    # def variance(t: np.array, kappa: float, sigma: float):
    #     """ Variance function of an OU process"""
    #     return (sigma ** 2) * (1.0 - np.exp(-2.0 * kappa * t)) / (2 * kappa)

    # @staticmethod
    # def std(t: np.array, kappa: float, sigma: float):
    #     return np.sqrt(StationaryCIRModel.variance(t, kappa, sigma))


def mle_ou(t, s):
    """Maximum-likelihood estimator for standard CIR"""

    def log_likelihood(q):
        """Calculates log likelihood of a standard CIR path"""
        K = q["kappa"]
        theta = q["theta"]
        sigma = q["sigma"]
        dt = np.diff(t)[0]
        # print(K)
        # print(theta)
        # print(sigma)

        c = (2 * K) / ((sigma ** 2) * (1 - np.exp(-K * dt)))
        l = ((2 * K * theta) / sigma ** 2) - 1
        p_dist = []
        for i in range(len(s) - 1):
            u = c * s[i] * np.exp(-K * dt) or 1e-16
            print(u, c, s[i], np.exp(-K * dt), K)
            v = c * s[i + 1]
            print(v)
            p = np.log(c) - c * (s[i] * np.exp(-K * dt) + s[i + 1]) + (l / 2) * np.log(
                 s[i + 1] / s[i] * np.exp(-K * dt)) + np.log(ive(l, 2 * np.sqrt(u * v)) or 1e-16)
            print(s[i])
            print(u)
            print(v)
            print(ive(l, 2 * np.sqrt(u * v)) or 1e-16)
            print(p)
            print("+======================+")
            p_dist.append(p)
        return - np.sum(p_dist)

    params = Parameters()
    params.add_many(
        ("kappa", 0.5, True, 1e-6, 20.0),
        ("theta", np.mean(s), True, 1e-6, None),
        ("sigma", est_sigma_quadratic_variation(t, s), True, 1e-8, None),
    )

    result = minimize(
        log_likelihood,
        params,
        method="L-BFGS-B",
        options={"maxiter": 5000, "disp": False},
    )
    return result


# simulate a path of the CIR process on a given grid t, starting with x_0 = 0.8
# q = (S_0, init value
# time, timeline
# kappa, mean reversion speed
# theta, mean reversion level
# sigma) volatility

t = np.arange(0, 10, 0.01)
q = (0.05, 5, 0.05, 0.5)
cir = StationaryCIRModel(*q)

x = cir.path(t)

plt.plot(t, x)
plt.show()

result = mle_ou(t, x)
print(result.status)
print(result.params)
