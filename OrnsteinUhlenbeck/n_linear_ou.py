"""
OU (Ornstein-Uhlenbeck) process
dX = -K(X-theta) dt + sigma dB
"""
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from utils import mle_ou

class OrnsteinUhlenbeckModel:
    def __init__(
        self, s_0: float, kappa: float, theta_a: float, theta_b: float, sigma: float
    ):
        assert kappa >= 0
        assert sigma >= 0
        self.s_0 = s_0
        self.kappa = kappa
        self.theta_a = theta_a
        self.theta_b = theta_b
        self.sigma = sigma

    def path(self, t):
        """ Simulates a sample path"""
        assert len(t) > 1
        dt: np.array = np.diff(t)
        s = [self.s_0]
        for t in range(len(dt)):
            ds = (
                self.kappa
                * (self.theta(t * dt[0], self.theta_a, self.theta_b) - s[t])
                * dt[0]
                + self.sigma * np.sqrt(dt[0]) * np.random.normal()
            )
            s.append(s[t] + ds)
        return s

    @staticmethod
    def theta(t: Union[np.array, float], a: float, b: float):
        if not isinstance(t, float):
            t = np.cumsum(t)
        return a * t + b

    @staticmethod
    def mean(
        s_t: Union[np.array, float],
        t: Union[np.array, float],
        dt: Union[np.array, float],
        kappa: float,
        theta_a: float,
        theta_b: float,
    ) -> float:
        """ Mean function of an OU process"""
        return (
            s_t * np.exp(-kappa * dt)
            + OrnsteinUhlenbeckModel.theta(t, theta_a, theta_b)
            + (
                np.exp(-kappa * dt)
                * (theta_a - kappa * OrnsteinUhlenbeckModel.theta(t, theta_a, theta_b))
                - theta_a
            )
            / kappa
        )

    @staticmethod
    def variance(t: np.array, kappa: float, sigma: float):
        """ Variance function of an OU process"""
        return (sigma ** 2) * (1.0 - np.exp(-2.0 * kappa * t)) / (2 * kappa)

    @staticmethod
    def std(t: np.array, kappa: float, sigma: float):
        return np.sqrt(OrnsteinUhlenbeckModel.variance(t, kappa, sigma))


# simulate a path of the OU process on a given grid t, starting with x_0 = 0.8
# q = (S_0, init value
# time, timeline
# kappa, mean reversion speed
# theta_a, mean reversion level
# theta_b,
# sigma) volatility

# t = np.arange(0, 100, 0.01)
# # q = (0.0, 20.0, 0.1, 0.5, 1.0)
# q = (0.0, 0.1, 0.000001, 0.5, 2.0)
# ou = OrnsteinUhlenbeckModel(*q)
# x = ou.path(t)
#
# plt.plot(t, x)
# plt.show()
#
# result = mle_ou(OrnsteinUhlenbeckModel, t, x)
# print(result.status)
# print(result.params)
