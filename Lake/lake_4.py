import numpy as np
from typing import Dict, Any, List
from scipy.optimize import brentq

def compute_x_crit(
    b: float,
    q: float,
    x_min: float = 0.01,
    x_max: float = 1.5
) -> float:
    """
    Compute the critical pollution threshold X_crit using Brent's method.

    X_crit solves:
        X^q / (1 + X^q) = b X

    The root is searched for in the interval [0.01, 1.5].
    """

    def balance_function(x: float) -> float:
        recycling = (x ** q) / (1.0 + x ** q)
        removal = b * x
        return recycling - removal

    # Brent's method requires opposite signs at interval endpoints
    f_min = balance_function(x_min)
    f_max = balance_function(x_max)

    if f_min * f_max > 0:
        raise ValueError(
            "No root found for X_crit in [0.01, 1.5]. "
            "Check parameter values b and q."
        )

    x_crit = brentq(balance_function, x_min, x_max)
    return x_crit


class LakeEnvironment:
    """
    Shared biophysical environment.
    """

    def __init__(self, params):
        """
        params must include:
            b      : natural removal rate
            q      : recycling nonlinearity
            mu     : mean of lognormal natural pollution (real space)
            sigma  : standard deviation of lognormal natural pollution (real space)
            T      : time horizon
            X0     : initial pollution level
        """
        self.b = params["b"]
        self.q = params["q"]
        self.mu = params["mu"]
        self.sigma = params["sigma"]
        self.T = params["T"]
        self.X0 = params.get("X0", 0.0)

        # --- convert real-space mean/variance to log-space parameters ---
        self._log_sigma_sq = np.log(1.0 + (self.sigma ** 2) / (self.mu ** 2))
        self._log_sigma = np.sqrt(self._log_sigma_sq)
        self._log_mu = np.log(self.mu) - 0.5 * self._log_sigma_sq

    def recycling(self, X):
        return (X ** self.q) / (1.0 + X ** self.q)

    def draw_natural_pollution(self):
        """
        Draw epsilon_t ~ LogNormal(mean=mu, var=sigma^2) in real space.
        """
        return np.random.lognormal(
            mean=self._log_mu,
            sigma=self._log_sigma
        )

    def step(self, X_t, a_t, r_t, epsilon_t):
        X_next = (
            X_t
            + a_t
            - r_t
            + self.recycling(X_t)
            - self.b * X_t
            + epsilon_t
        )
        return max(X_next, 0.0)


class LocalCommunity:
    """
    Community module: controls emissions a_t and evaluates economic objectives.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        params must include:
            alpha  : benefit-to-pollution ratio
            delta  : discount factor
        """
        self.alpha = params["alpha"]
        self.delta = params["delta"]

    def economic_benefit(self, a_t: float, t: int) -> float:
        return self.alpha * a_t * (self.delta ** t)

    def evaluate_objectives(
        self,
        emissions: List[float],
        states: List[float],
        x_crit: float
    ) -> Dict[str, float]:
        """
        Returns community objectives.
        """
        economic_value = sum(
            self.economic_benefit(a_t, t)
            for t, a_t in enumerate(emissions)
        )

        safe = all(X < x_crit for X in states)

        return {
            "community_economic_benefit": economic_value,
            "community_avoids_eutrophication": float(safe)
        }


class EnvironmentalRegulator:
    """
    Regulator module: controls remediation r_t and evaluates environmental objectives.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        params must include:
            cost_coeff : coefficient for remediation cost
        """
        self.cost_coeff = params["cost_coeff"]

    def remediation_cost(self, r_t: float) -> float:
        return self.cost_coeff * (r_t ** 2)

    def evaluate_objectives(
        self,
        remediations: List[float],
        states: List[float],
        x_crit: float
    ) -> Dict[str, float]:
        """
        Returns regulator objectives.
        """
        avoids_eutrophication = all(X < x_crit for X in states)

        pollution_stock = sum(states)

        total_cost = sum(self.remediation_cost(r_t) for r_t in remediations)

        return {
            "regulator_avoids_eutrophication": float(avoids_eutrophication),
            "regulator_total_pollution": pollution_stock,
            "regulator_total_cost": total_cost
        }


def simulate_lake_model(
    decisions: Dict[str, List[float]],
    uncertain_params: Dict[str, Any],
    constant_params: Dict[str, Any]
) -> Dict[str, float]:
    """
    Unified interface function.
    """

    # --- Initialize environment ---
    env = LakeEnvironment({
        "b": uncertain_params["b"],
        "q": uncertain_params["q"],
        "mu": uncertain_params["mu"],
        "sigma": uncertain_params["sigma"],
        "T": constant_params["T"],
        "X0": constant_params.get("X0", 0.0),
    })

    # --- Initialize stakeholders ---
    community = LocalCommunity({
        "alpha": constant_params["alpha"],
        "delta": uncertain_params["delta"],
    })

    regulator = EnvironmentalRegulator({
        "cost_coeff": constant_params["cost_coeff"]
    })

    # --- Compute endogenous tipping point ---
    x_crit = compute_x_crit(
        b=env.b,
        q=env.q,
        x_min=0.01,
        x_max=1.5
    )

    # --- Simulate dynamics ---
    X = env.X0
    states = [X]

    for t in range(env.T):
        a_t = decisions["a"][t]
        r_t = decisions["r"][t]
        epsilon_t = env.draw_natural_pollution()

        X = env.step(X, a_t, r_t, epsilon_t)
        states.append(X)

        if X >= x_crit:
            break  # irreversible eutrophication

    # --- Evaluate objectives ---
    results = {}
    results.update(
        community.evaluate_objectives(decisions["a"], states, x_crit)
    )
    results.update(
        regulator.evaluate_objectives(decisions["r"], states, x_crit)
    )

    return results
