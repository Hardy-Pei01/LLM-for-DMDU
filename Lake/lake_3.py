import numpy as np
from scipy.optimize import brentq


# ============================================================
# Common Environment
# ============================================================

import numpy as np
from scipy.optimize import brentq


class LakeEnvironment:
    """
    Shared biophysical lake environment.
    """

    def __init__(self, uncertain_params, horizon):
        """
        uncertain_params: dict of deeply uncertain parameters
            {
              mu_eps: mean of epsilon_t,
              sigma_eps: standard deviation of epsilon_t,
              b, q
            }
        horizon: int
        """
        self.b = uncertain_params["b"]
        self.q = uncertain_params["q"]

        # Mean and std of the *lognormal variable*
        self.mu_eps = uncertain_params["mu_eps"]
        self.sigma_eps = uncertain_params["sigma_eps"]

        self.T = horizon

        # Compute critical threshold endogenously
        self.X_crit = self.compute_X_crit()

        # State variable
        self.X = np.zeros(self.T + 1)

        # Precompute normal-space parameters
        self._compute_lognormal_parameters()

    def _compute_lognormal_parameters(self):
        """
        Convert lognormal mean/variance to normal parameters.
        """
        mean = self.mu_eps
        var = self.sigma_eps ** 2

        self.sigma_ln = np.sqrt(np.log(1.0 + var / mean**2))
        self.mu_ln = np.log(mean) - 0.5 * self.sigma_ln**2

    def compute_X_crit(self):
        """
        Compute X_crit using Brent's root-finding method.
        """

        def balance_function(X):
            recycling = (X ** self.q) / (1.0 + X ** self.q)
            removal = self.b * X
            return recycling - removal

        return brentq(balance_function, 0.01, 1.5)

    def natural_pollution(self):
        """
        Draw epsilon_t with correct lognormal statistics.
        """
        return np.random.lognormal(
            mean=self.mu_ln,
            sigma=self.sigma_ln
        )

    def step(self, t, A_t):
        X_t = self.X[t]
        epsilon_t = self.natural_pollution()

        recycling = (X_t ** self.q) / (1.0 + X_t ** self.q)
        removal = self.b * X_t

        self.X[t + 1] = X_t + A_t + recycling - removal + epsilon_t


# ============================================================
# Local Community Perspective
# ============================================================

class LocalCommunity:
    """
    Local community decision-maker.
    """

    def __init__(self, uncertain_params, constant_params):
        """
        uncertain_params: {delta_C}
        constant_params: {alpha}
        """
        self.alpha = constant_params["alpha"]
        self.delta = uncertain_params["delta_C"]

        self.economic_benefit = 0.0
        self.eutrophication_occurred = False

    def decide(self, t, X_t, a_R_t, decisions):
        return decisions["a_C"][t]

    def update_objectives(self, t, A_t, X_t, X_crit):
        self.economic_benefit += self.alpha * A_t * (self.delta ** t)

        if X_t >= X_crit:
            self.eutrophication_occurred = True

    def evaluate_objectives(self):
        return {
            "C1_economic_benefit": self.economic_benefit,
            "C2_eutrophication": float(self.eutrophication_occurred)
        }


# ============================================================
# Environmental Regulator Perspective
# ============================================================

class EnvironmentalRegulator:
    """
    Environmental regulator decision-maker.
    """

    def __init__(self, uncertain_params):
        """
        uncertain_params: {delta_R}
        """
        self.delta = uncertain_params["delta_R"]

        self.eutrophication_occurred = False
        self.restriction_cost = 0.0
        self.pollution_stock_cost = 0.0

    def decide(self, t, X_t, a_C_t, decisions):
        return decisions["a_R"][t]

    def update_objectives(self, t, X_t, a_C_t, a_R_t, X_crit):
        if X_t >= X_crit:
            self.eutrophication_occurred = True

        self.restriction_cost += (a_R_t - a_C_t) ** 2
        self.pollution_stock_cost += X_t * (self.delta ** t)

    def evaluate_objectives(self):
        return {
            "R1_eutrophication": float(self.eutrophication_occurred),
            "R2_restriction_cost": self.restriction_cost,
            "R3_pollution_stock": self.pollution_stock_cost
        }


# ============================================================
# Unified Simulation Interface
# ============================================================

def simulate_lake_governance(decisions, uncertain_params, constant_params):
    """
    Unified model interface.

    Inputs
    ------
    decisions : dict
        {
            "a_C": array-like,
            "a_R": array-like
        }

    uncertain_params : dict
        {
            "mu": float,
            "sigma": float,
            "b": float,
            "q": float,
            "delta_C": float,
            "delta_R": float
        }

    constant_params : dict
        {
            "alpha": float,
            "T": int
        }

    Returns
    -------
    dict
        Objective values for all perspectives.
    """

    T = constant_params["T"]

    # Initialize environment
    env = LakeEnvironment(
        uncertain_params=uncertain_params,
        horizon=T
    )

    # Initialize stakeholders
    community = LocalCommunity(
        uncertain_params=uncertain_params,
        constant_params=constant_params
    )

    regulator = EnvironmentalRegulator(
        uncertain_params=uncertain_params
    )

    # Simulation loop
    for t in range(T):
        X_t = env.X[t]

        a_C_t = community.decide(t, X_t, decisions["a_R"][t], decisions)
        a_R_t = regulator.decide(t, X_t, a_C_t, decisions)

        A_t = min(a_C_t, a_R_t)

        env.step(t, A_t)

        community.update_objectives(t, A_t, X_t, env.X_crit)
        regulator.update_objectives(t, X_t, a_C_t, a_R_t, env.X_crit)

    results = {}
    results.update(community.evaluate_objectives())
    results.update(regulator.evaluate_objectives())

    return results
