import numpy as np
from scipy.optimize import root_scalar

# ============================================================
# 1. Common Environment Class
# ============================================================

class LakeEnvironment:
    """
    Shared lake environment with stochastic inflows and nonlinear pollution dynamics.
    """

    def __init__(self, params):
        """
        Parameters:
            params (dict): global and uncertain parameters.
                Required keys:
                - b, q, mu, sigma, delta, T, X0
        """
        self.b = params["b"]
        self.q = params["q"]
        self.mu = params["mu"]
        self.sigma = params["sigma"]
        self.delta = params["delta"]
        self.T = params["T"]
        self.X0 = params.get("X0", 0.0)

        # Compute critical pollution threshold X_crit(b, q)
        self.X_crit = self.compute_Xcrit()

    # ------------------------------------------------------------
    # Root-finding for X_crit(b, q)
    # ------------------------------------------------------------
    def compute_Xcrit(self):
        """
        Solve f(X) = X^q/(1 + X^q) - bX = 0 for X in [0.01, 1.5].
        Uses robust root-finding (brentq method).
        """
        def f(X):
            return (X**self.q) / (1 + X**self.q) - self.b * X

        try:
            sol = root_scalar(f, bracket=[0.01, 1.5], method="brentq")
            if sol.converged:
                return sol.root
            else:
                # fallback: use midpoint if solver fails
                return 0.5 * (0.01 + 1.5)
        except ValueError:
            # fallback: default midpoint if no valid bracket
            return 0.5 * (0.01 + 1.5)

    # ------------------------------------------------------------
    # Corrected lognormal stochastic inflow
    # ------------------------------------------------------------
    def stochastic_inflow(self):
        """
        Draw stochastic inflow ε_t from LogNormal with real-space mean=μ, std=σ.
        Converts μ, σ to log-space parameters.
        """
        shape = np.sqrt(np.log(1 + (self.sigma / self.mu) ** 2))
        scale = self.mu / np.sqrt(1 + (self.sigma / self.mu) ** 2)
        return np.random.lognormal(mean=np.log(scale), sigma=shape)

    # ------------------------------------------------------------
    def transition(self, X_t, a_L, a_R, epsilon_t):
        """Compute next pollution level X_{t+1}."""
        return X_t + a_L + a_R + (X_t**self.q) / (1 + X_t**self.q) - self.b * X_t + epsilon_t


# ============================================================
# 2. Local Community Class
# ============================================================

class LocalCommunity:
    """Local community perspective: maximize economic benefit, minimize pollution damage."""

    def __init__(self, alpha_L, gamma_L):
        self.alpha_L = alpha_L
        self.gamma_L = gamma_L

    def objectives(self, emissions, states, delta):
        T = len(emissions)
        discounts = np.array([delta**t for t in range(T)])
        f1 = np.sum(self.alpha_L * np.array(emissions) * discounts)
        f2 = -np.sum(self.gamma_L * np.array(states) * discounts)
        return f1, f2


# ============================================================
# 3. Environmental Regulator Class
# ============================================================

class EnvironmentalRegulator:
    """Regulator perspective: protect ecosystem and minimize policy cost."""

    def __init__(self, c_R):
        self.c_R = c_R

    def objectives(self, actions, states, X_crit, delta):
        T = len(actions)
        discounts = np.array([delta**t for t in range(T)])
        penalty = np.maximum(np.array(states) - X_crit, 0.0)
        f1 = -np.sum((penalty**2) * discounts)  # ecological protection
        f2 = -np.sum(self.c_R * np.array(actions) ** 2 * discounts)  # policy cost
        return f1, f2


# ============================================================
# 4. Unified Simulation Interface
# ============================================================

def simulate_lake_problem(controls, params, random_seed=None):
    """
    Unified simulation of the multi-perspective lake problem.

    Parameters:
        controls (dict): Control time series.
            - 'local': list/array of local emissions a_t^(L)
            - 'regulator': list/array of regulator actions a_t^(R)
        params (dict): Model parameters (including uncertainties).
            Required keys:
                b, q, mu, sigma, delta, T, X0, alpha_L, gamma_L, c_R
        random_seed (int, optional): Seed for reproducibility.

    Returns:
        dict: {
            'local': {'economic_benefit': f1_L, 'environment_disutility': f2_L},
            'regulator': {'ecological_protection': f1_R, 'policy_cost': f2_R},
            'trajectory': {'X': X_series, 'Xcrit': X_crit}
        }
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Initialize shared environment
    env = LakeEnvironment(params)

    # Initialize agents
    local = LocalCommunity(params["alpha_L"], params["gamma_L"])
    regulator = EnvironmentalRegulator(params["c_R"])

    # Unpack controls
    a_L_series = np.array(controls["local"])
    a_R_series = np.array(controls["regulator"])
    T = params["T"]

    # Simulate pollution trajectory
    X_series = [env.X0]
    for t in range(T):
        epsilon_t = env.stochastic_inflow()
        X_next = env.transition(X_series[-1], a_L_series[t], a_R_series[t], epsilon_t)
        X_series.append(X_next)

    # Compute objectives
    f1_L, f2_L = local.objectives(a_L_series, X_series[:-1], env.delta)
    f1_R, f2_R = regulator.objectives(a_R_series, X_series[:-1], env.X_crit, env.delta)

    return {
        "local": {
            "economic_benefit": f1_L,
            "environment_disutility": f2_L
        },
        "regulator": {
            "ecological_protection": f1_R,
            "policy_cost": f2_R
        },
        "trajectory": {
            "X": X_series,
            "Xcrit": env.X_crit
        }
    }


# ============================================================
# Example usage (not executed):
# ============================================================
controls = {
    'local': [0.03] * 100,
    'regulator': [0] * 100
}
params = {
    'b': 0.42, 'q': 2.0, 'mu': 0.02, 'sigma': 0.0017,
    'delta': 0.98, 'T': 100, 'X0': 0.0,
    'alpha_L': 0.4, 'gamma_L': 0, 'c_R': 0
}