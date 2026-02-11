import numpy as np
from typing import Dict, List, Any
from scipy.optimize import brentq

class LakeEnvironment:
    """
    Shared lake environment with nonlinear stochastic dynamics
    and endogenous eutrophication threshold.
    """

    def __init__(self, uncertain_params: Dict[str, float], constants: Dict[str, float]):
        self.mu_eps = uncertain_params["mu"]        # mean of epsilon_t
        self.var_eps = uncertain_params["sigma"]**2 # variance of epsilon_t
        self.b = uncertain_params["b"]
        self.q = uncertain_params["q"]

        self.T = constants["T"]

        # Convert to underlying normal parameters
        self._compute_lognormal_parameters()

        self.X_crit = self._compute_critical_threshold()

    def _compute_lognormal_parameters(self):
        """
        Convert (mean, variance) of lognormal distribution
        to parameters of underlying normal distribution.
        """
        self.sigma_N_sq = np.log(1.0 + self.var_eps / self.mu_eps**2)
        self.sigma_N = np.sqrt(self.sigma_N_sq)
        self.mu_N = np.log(self.mu_eps) - 0.5 * self.sigma_N_sq

    def sample_natural_pollution(self) -> float:
        """
        Sample epsilon_t from LogNormal(mu, sigma^2),
        where mu and sigma^2 are the mean and variance
        of the lognormal distribution.
        """
        return np.random.lognormal(mean=self.mu_N, sigma=self.sigma_N)

    def _threshold_equation(self, X: float) -> float:
        """
        Equation whose root defines X_crit:
            X^q / (1 + X^q) - b X = 0
        """
        return (X**self.q) / (1 + X**self.q) - self.b * X

    def _compute_critical_threshold(self) -> float:
        """
        Compute X_crit using Brent's method over [0.01, 1.5].
        """

        lower, upper = 0.01, 1.5

        f_lower = self._threshold_equation(lower)
        f_upper = self._threshold_equation(upper)

        if f_lower * f_upper > 0:
            raise ValueError(
                "No root for X_crit found in [0.01, 1.5] "
                f"for b={self.b}, q={self.q}"
            )

        return brentq(self._threshold_equation, lower, upper)

    def step(self, X_t: float, a_c: float, r_r: float, epsilon_t: float) -> float:
        net_anthropogenic = a_c - r_r
        recycling = X_t**self.q / (1 + X_t**self.q)
        removal = self.b * X_t
        X_next = X_t + net_anthropogenic + recycling - removal + epsilon_t
        return max(X_next, 0.0)


class LocalCommunity:
    """
    Local community controlling pollution emissions.
    """

    def __init__(self, uncertain_params: Dict[str, float], constants: Dict[str, float]):
        self.alpha = constants["alpha"]
        self.delta = uncertain_params["delta"]

        self.economic_benefit = 0.0
        self.failed = False

    def reset(self):
        self.economic_benefit = 0.0
        self.failed = False

    def observe_and_act(self, t: int, X_t: float, decisions: Dict[str, List[float]]) -> float:
        return decisions["a_community"][t]

    def update_objectives(self, t: int, a_c: float):
        self.economic_benefit += self.alpha * a_c * (self.delta ** t)

    def check_failure(self, X_t: float, X_crit: float):
        if X_t >= X_crit:
            self.failed = True


class EnvironmentalRegulator:
    """
    Environmental regulator controlling abatement.
    """

    def __init__(self):
        self.total_pollution = 0.0
        self.regulation_cost = 0.0
        self.failed = False

    def reset(self):
        self.total_pollution = 0.0
        self.regulation_cost = 0.0
        self.failed = False

    def observe_and_act(self, t: int, X_t: float, decisions: Dict[str, List[float]]) -> float:
        return decisions["r_regulator"][t]

    def update_objectives(self, X_t: float, r_r: float):
        self.total_pollution += X_t
        self.regulation_cost += r_r**2  # convex cost

    def check_failure(self, X_t: float, X_crit: float):
        if X_t >= X_crit:
            self.failed = True


def simulate_lake_model(
    decisions: Dict[str, List[float]],
    uncertain_params: Dict[str, float],
    constants: Dict[str, float],
) -> Dict[str, Any]:
    """
    Unified interface for the composed lake model.

    Parameters
    ----------
    decisions:
        {
            "a_community": [a_0, ..., a_T],
            "r_regulator": [r_0, ..., r_T]
        }

    uncertain_params:
        {
            "mu": float,
            "sigma": float,
            "b": float,
            "q": float,
            "delta": float
        }

    constants:
        {
            "T": int,
            "alpha": float
        }
    """

    env = LakeEnvironment(uncertain_params, constants)
    community = LocalCommunity(uncertain_params, constants)
    regulator = EnvironmentalRegulator()

    community.reset()
    regulator.reset()

    X_t = 0.0
    T = constants["T"]

    for t in range(T):
        a_c = community.observe_and_act(t, X_t, decisions)
        r_r = regulator.observe_and_act(t, X_t, decisions)

        epsilon_t = env.sample_natural_pollution()
        X_t = env.step(X_t, a_c, r_r, epsilon_t)

        community.update_objectives(t, a_c)
        regulator.update_objectives(X_t, r_r)

        community.check_failure(X_t, env.X_crit)
        regulator.check_failure(X_t, env.X_crit)

        if X_t >= env.X_crit:
            break  # absorbing eutrophic state

    return {
        "community": {
            "economic_benefit": community.economic_benefit,
            "eutrophication_occurred": community.failed,
        },
        "regulator": {
            "total_pollution": regulator.total_pollution,
            "regulation_cost": regulator.regulation_cost,
            "eutrophication_occurred": regulator.failed,
        },
        "environment": {
            "X_crit": env.X_crit,
            "final_pollution": X_t,
        },
    }
