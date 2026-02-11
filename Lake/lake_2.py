import math
import random
from typing import Dict, List


# ============================================================
# Common Environment
# ============================================================

import math
import random
from typing import Dict, List
from scipy.optimize import brentq


# ============================================================
# Common Environment (REFINED)
# ============================================================

import math
import random
from scipy.optimize import brentq


class LakeEnvironment:
    """
    Shared physical environment: lake pollution dynamics,
    stochastic forcing, and eutrophication threshold.
    """

    def __init__(self, uncertain_params, constant_params):
        self.mu = uncertain_params["mu"]          # mean of lognormal
        self.sigma2 = uncertain_params["sigma"]   # variance of lognormal
        self.b = uncertain_params["b"]
        self.q = uncertain_params["q"]
        self.delta = uncertain_params["delta"]

        self.T = constant_params["T"]

        # Initial state
        self.X = 0.0
        self.E = 0

        # Convert lognormal moments to normal parameters
        self._compute_lognormal_parameters()

        # Compute critical threshold
        self.X_crit = self._compute_xcrit()

    # --------------------------------------------------
    # Lognormal parameter conversion (REFINED)
    # --------------------------------------------------

    def _compute_lognormal_parameters(self):
        """
        Convert lognormal mean and variance to
        underlying normal parameters.
        """
        variance = self.sigma2
        mean = self.mu

        self.sigma_N2 = math.log(1.0 + variance / (mean ** 2))
        self.sigma_N = math.sqrt(self.sigma_N2)
        self.mu_N = math.log(mean) - 0.5 * self.sigma_N2

    def draw_natural_pollution(self):
        """
        Correct lognormal sampling using underlying
        normal distribution parameters.
        """
        return random.lognormvariate(self.mu_N, self.sigma_N)

    # --------------------------------------------------
    # X_crit computation (unchanged)
    # --------------------------------------------------

    def _threshold_equation(self, X):
        return (X ** self.q) / (1.0 + X ** self.q) - self.b * X

    def _compute_xcrit(self):
        lower, upper = 0.01, 1.5
        return brentq(self._threshold_equation, lower, upper)

    # --------------------------------------------------
    # State transition
    # --------------------------------------------------

    def step(self, a_community, r_regulator):
        if self.E == 1:
            return self.X, self.E

        epsilon = self.draw_natural_pollution()
        A = a_community - r_regulator

        self.X = (
            self.X
            + A
            + (self.X ** self.q) / (1.0 + self.X ** self.q)
            - self.b * self.X
            + epsilon
        )

        if self.X >= self.X_crit:
            self.E = 1

        return self.X, self.E



# ============================================================
# Perspective: Local Community
# ============================================================

class LocalCommunity:
    """
    Local community perspective: economic benefits and risk exposure.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.economic_benefit = 0.0
        self.tipping_occurred = False

    def record_step(self, a_t: float, delta: float, t: int, eutrophic: bool):
        if not eutrophic:
            self.economic_benefit += self.alpha * a_t * (delta ** t)
        if eutrophic:
            self.tipping_occurred = True

    def objectives(self):
        return {
            "community_economic_benefit": self.economic_benefit,
            "community_tipping_risk": int(self.tipping_occurred),
        }


# ============================================================
# Perspective: Environmental Regulator
# ============================================================

class EnvironmentalRegulator:
    """
    Environmental regulator perspective: safety, cost, disruption.
    """

    def __init__(self):
        self.regulatory_cost = 0.0
        self.economic_disruption = 0.0
        self.tipping_occurred = False

    @staticmethod
    def cost_function(r):
        return r ** 2  # convex cost

    def record_step(
        self, a_t: float, r_t: float, delta: float, t: int, eutrophic: bool
    ):
        self.regulatory_cost += self.cost_function(r_t) * (delta ** t)
        self.economic_disruption += (a_t - r_t) ** 2
        if eutrophic:
            self.tipping_occurred = True

    def objectives(self):
        return {
            "regulator_tipping_risk": int(self.tipping_occurred),
            "regulator_cost": self.regulatory_cost,
            "regulator_disruption": self.economic_disruption,
        }


# ============================================================
# Unified Simulation Interface
# ============================================================

def simulate_lake_model(
    decision_variables: Dict[str, List[float]],
    uncertain_parameters: Dict[str, float],
    constant_parameters: Dict[str, float],
) -> Dict[str, float]:
    """
    Unified interface function.

    Inputs
    ------
    decision_variables:
        {
            "a_community": [a_0, a_1, ..., a_T],
            "r_regulator": [r_0, r_1, ..., r_T]
        }

    uncertain_parameters:
        {
            "mu": ...,
            "sigma": ...,
            "b": ...,
            "q": ...,
            "delta": ...
        }

    constant_parameters:
        {
            "T": ...,
            "alpha": ...
        }

    Returns
    -------
    Dictionary of objective values for each perspective.
    """

    T = constant_parameters["T"]

    env = LakeEnvironment(uncertain_parameters, constant_parameters)
    community = LocalCommunity(alpha=constant_parameters["alpha"])
    regulator = EnvironmentalRegulator()

    a_series = decision_variables["a_community"]
    r_series = decision_variables["r_regulator"]

    for t in range(T):
        a_t = a_series[t]
        r_t = r_series[t]

        X, E = env.step(a_t, r_t)

        community.record_step(
            a_t=a_t,
            delta=env.delta,
            t=t,
            eutrophic=(E == 1),
        )

        regulator.record_step(
            a_t=a_t,
            r_t=r_t,
            delta=env.delta,
            t=t,
            eutrophic=(E == 1),
        )

        if E == 1:
            break  # irreversible regime

    # Combine objectives (no aggregation)
    objectives = {}
    objectives.update(community.objectives())
    objectives.update(regulator.objectives())

    return objectives
