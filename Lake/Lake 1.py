import numpy as np
from scipy.optimize import brentq

# ============================================================
# Shared Environment Model
# ============================================================

class LakeEnvironment:
    """
    Shared lake environment with pollution dynamics.
    """

    def __init__(self, params):
        """
        params: dictionary containing uncertain parameters and constants
        keys: 'b', 'q', 'mu', 'sigma', 'alpha', 'delta', 'T'
        """
        self.b = params["b"]
        self.q = params["q"]
        self.mu = params["mu"]
        self.sigma = params["sigma"]
        self.alpha = params["alpha"]
        self.delta = params["delta"]
        self.T = params["T"]

        # Derive X_crit dynamically from b and q within [0.01, 1.5]
        self.X_crit = self._compute_Xcrit()

        # Convert given lognormal mean/variance (μ, σ²) to normal parameters
        self._set_lognormal_parameters()

    # -----------------------------------------------------------------
    # Compute Xcrit from implicit condition: (X^q)/(1 + X^q) = b * X
    # -----------------------------------------------------------------
    def _compute_Xcrit(self):
        def f(X):
            return (X ** self.q) / (1 + X ** self.q) - self.b * X

        try:
            return brentq(f, 0.01, 1.5)  # constrained range
        except ValueError:
            # If no root found in range, use midpoint as fallback
            return 0.75

    # -----------------------------------------------------------------
    # Convert lognormal mean/variance to normal parameters
    # -----------------------------------------------------------------
    def _set_lognormal_parameters(self):
        mu, sigma = self.mu, self.sigma
        s = np.sqrt(np.log(1 + (sigma ** 2) / (mu ** 2)))
        m = np.log(mu ** 2 / np.sqrt(mu ** 2 + sigma ** 2))
        self.ln_mean = m
        self.ln_sigma = s

    # -----------------------------------------------------------------
    # One-step state transition
    # -----------------------------------------------------------------
    def step(self, X_t, a_E, u_C, a_S):
        """
        Compute next-period pollution level:
        X_{t+1} = X_t + (a_E - u_C + a_S) + (X_t^q)/(1 + X_t^q) - b*X_t + ε_t
        """
        epsilon_t = np.random.lognormal(mean=self.ln_mean, sigma=self.ln_sigma)
        X_next = X_t + (a_E - u_C + a_S) + (X_t ** self.q) / (1 + X_t ** self.q) - self.b * X_t + epsilon_t
        return X_next


# ============================================================
# Perspective 1: Economic Planner
# ============================================================

class EconomicPlanner:
    def __init__(self, env):
        self.env = env

    def objective(self, a_E_series):
        """Maximize discounted economic benefit."""
        T = self.env.T
        alpha, delta = self.env.alpha, self.env.delta
        return np.sum([alpha * a_E_series[t] * (delta ** t) for t in range(T)])


# ============================================================
# Perspective 2: Environmental Regulator
# ============================================================

class EnvironmentalRegulator:
    def __init__(self, env):
        self.env = env

    def objectives(self, X_series):
        """
        1. Minimize expected pollution
        2. Minimize eutrophication risk
        """
        mean_pollution = np.mean(X_series)
        risk = np.mean(np.array(X_series) >= self.env.X_crit)
        return {"pollution": mean_pollution, "risk": risk}


# ============================================================
# Perspective 3: Community / Public Health
# ============================================================

class Community:
    def __init__(self, env):
        self.env = env

    def objectives(self, a_E_series, X_series):
        """
        Multi-objective:
          1. Economic well-being
          2. Environmental quality
          3. Equity (variance)
        """
        alpha, delta = self.env.alpha, self.env.delta
        T = self.env.T

        f1 = np.sum([alpha * a_E_series[t] * (delta ** t) for t in range(T)])
        f2 = -np.sum([X_series[t] * (delta ** t) for t in range(T)])
        f3 = -np.var(X_series)

        return {"economic": f1, "environment": f2, "equity": f3}


# ============================================================
# Perspective 4: Global / Scientific (Robustness)
# ============================================================

class GlobalScientist:
    def __init__(self, env):
        self.env = env

    def objectives(self, a_E_series, X_series, param_set_list):
        """
        Evaluate robustness under deep uncertainty.
        Each param_set in param_set_list is a dict with {b, q, mu, sigma, delta}.
        """
        alpha = self.env.alpha
        T = self.env.T

        # Expected performance
        expected_perf = np.mean([
            np.sum([alpha * a_E_series[t] * (p["delta"] ** t) for t in range(T)])
            for p in param_set_list
        ])

        # Worst-case (robustness)
        worst_case = np.min([
            np.sum([alpha * a_E_series[t] * (p["delta"] ** t) for t in range(T)])
            for p in param_set_list
        ])

        # Sustainability: average probability of eutrophication across uncertain sets
        risks = []
        for p in param_set_list:
            b, q = p["b"], p["q"]

            def f(X):
                return (X ** q) / (1 + X ** q) - b * X

            try:
                Xcrit_p = brentq(f, 0.01, 1.5)
            except ValueError:
                Xcrit_p = 0.75  # fallback
            risks.append(np.mean(np.array(X_series) >= Xcrit_p))

        sustainability = -np.max(risks)

        return {
            "expected": expected_perf,
            "robustness": worst_case,
            "sustainability": sustainability
        }


# ============================================================
# Unified Simulation Interface
# ============================================================

def simulate_lake_system(control_dict, param_dict):
    """
    Unified simulation interface.

    Inputs:
    - control_dict: dictionary with control time series
        keys: 'a_E', 'a_R', 'u_C', 'a_S'
    - param_dict: dictionary with uncertain parameter values
        keys: 'b', 'q', 'mu', 'sigma', 'alpha', 'delta', 'T'

    Returns:
    - dict: objectives for each perspective
    """

    # Initialize environment
    env = LakeEnvironment(param_dict)

    # Unpack controls
    a_E = np.array(control_dict["a_E"])
    a_R = np.array(control_dict["a_R"])
    u_C = np.array(control_dict["u_C"])
    a_S = np.array(control_dict["a_S"])

    # Enforce regulatory cap
    a_E = np.minimum(a_E, a_R)

    # Simulate dynamics
    X_series = [0.0]
    X_t = 0.0
    for t in range(env.T):
        X_next = env.step(X_t, a_E[t], u_C[t], a_S[t])
        X_series.append(X_next)
        X_t = X_next

    # Compute objectives for all perspectives
    econ = EconomicPlanner(env).objective(a_E)
    reg = EnvironmentalRegulator(env).objectives(X_series)
    comm = Community(env).objectives(a_E, X_series)

    # Example uncertainty sets for scientist perspective
    theta_list = [
        {"b": param_dict["b"] * 0.9, "q": param_dict["q"], "mu": param_dict["mu"], "sigma": param_dict["sigma"], "delta": param_dict["delta"]},
        {"b": param_dict["b"], "q": param_dict["q"] * 1.1, "mu": param_dict["mu"], "sigma": param_dict["sigma"], "delta": param_dict["delta"]},
        {"b": param_dict["b"], "q": param_dict["q"], "mu": param_dict["mu"] * 1.1, "sigma": param_dict["sigma"], "delta": param_dict["delta"]}
    ]
    sci = GlobalScientist(env).objectives(a_E, X_series, theta_list)

    # Combine and return results
    return {
        "EconomicPlanner": {"economic_benefit": econ},
        "Regulator": reg,
        "Community": comm,
        "Scientist": sci,
        "StateTrajectory": X_series
    }


# ============================================================
# Example Call to the Unified Simulation Interface
# ============================================================

if __name__ == "__main__":

    # -----------------------------
    # Example uncertain parameters
    # -----------------------------
    param_dict = {
        "b": 0.4,        # natural removal rate
        "q": 2.0,        # recycling rate
        "mu": 0.02,      # mean natural inflow (lognormal mean)
        "sigma": 0.01,   # std of natural inflow (lognormal std)
        "alpha": 1.0,    # benefit-to-pollution ratio
        "delta": 0.95,   # discount factor
        "T": 20          # simulation horizon (years)
    }

    # -----------------------------
    # Example control variable time series
    # -----------------------------
    T = param_dict["T"]
    control_dict = {
        "a_E": np.full(T, 0.1),   # Economic emissions
        "a_R": np.full(T, 0.15),  # Regulatory cap (must be >= a_E)
        "u_C": np.full(T, 0.02),  # Community mitigation
        "a_S": np.zeros(T)        # Global interventions (none)
    }

    # -----------------------------
    # Run the unified simulation
    # -----------------------------
    results = simulate_lake_system(control_dict, param_dict)

    # -----------------------------
    # Display results
    # -----------------------------
    print("\n=== Simulation Results ===")
    print(f"Final lake pollution: {results['StateTrajectory'][-1]:.4f}")
    print("\nEconomic Planner Objective:")
    print(results["EconomicPlanner"])

    print("\nEnvironmental Regulator Objectives:")
    for k, v in results["Regulator"].items():
        print(f"  {k}: {v:.4f}")

    print("\nCommunity Objectives:")
    for k, v in results["Community"].items():
        print(f"  {k}: {v:.4f}")

    print("\nGlobal/Scientific Objectives:")
    for k, v in results["Scientist"].items():
        print(f"  {k}: {v:.4f}")
