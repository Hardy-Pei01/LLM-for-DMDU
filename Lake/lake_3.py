import numpy as np
from scipy.optimize import root_scalar


# ============================================================
# 1. Shared Environment Model
# ============================================================

class LakeEnvironment:
    """
    Shared environmental system representing lake pollution dynamics.
    """

    def __init__(self, params):
        """
        Initialize environment parameters.
        params: dict containing 'b', 'q', 'mu', 'sigma', and 'T' (time horizon)
        """
        self.b = params.get('b', 0.4)
        self.q = params.get('q', 2.0)
        self.mu = params.get('mu', 0.05)  # mean of lognormal (linear domain)
        self.sigma = params.get('sigma', 0.02)  # std dev of lognormal (linear domain)
        self.T = params.get('T', 50)

        # Initialize pollution state
        self.X = [0.0]  # initial pollution level

    # --------------------------------------------------------
    # Compute the critical threshold within [0.01, 1.5]
    # --------------------------------------------------------
    def critical_threshold(self):
        """
        Compute the endogenous critical pollution threshold X_crit(b, q)
        defined implicitly by X^q / (1 + X^q) = b * X.
        The solution is constrained to [0.01, 1.5].
        """

        def f(x):
            return (x ** self.q) / (1 + x ** self.q) - self.b * x

        sol = root_scalar(f, bracket=[0.01, 1.5], method='brentq')
        if sol.converged:
            return sol.root
        else:
            raise RuntimeError("Failed to find X_crit in [0.01, 1.5].")

    # --------------------------------------------------------
    # Sample stochastic natural inflow ε_t from lognormal(μ, σ)
    # --------------------------------------------------------
    def stochastic_inflow(self, rng=np.random.default_rng()):
        """
        Sample ε_t ~ LogNormal(μ, σ^2), where μ and σ are the mean and std dev
        in the linear domain. The corresponding log-space parameters are derived.
        """
        mu_log = np.log(self.mu ** 2 / np.sqrt(self.sigma ** 2 + self.mu ** 2))
        sigma_log = np.sqrt(np.log(1 + (self.sigma ** 2 / self.mu ** 2)))
        return rng.lognormal(mean=mu_log, sigma=sigma_log)

    # --------------------------------------------------------
    # Pollution transition dynamics
    # --------------------------------------------------------
    def transition(self, a_L, a_R, epsilon_t):
        """
        Compute the next pollution state.
        """
        X_t = self.X[-1]
        X_next = X_t + a_L + a_R + (X_t ** self.q) / (1 + X_t ** self.q) - self.b * X_t + epsilon_t
        self.X.append(X_next)
        return X_next

    def reset(self):
        """Reset pollution to initial state."""
        self.X = [0.0]


# ============================================================
# 2. Local Community Model
# ============================================================

class LocalCommunity:
    """
    Local community perspective: seeks economic benefit and avoids penalties.
    """

    def __init__(self, params):
        self.alpha = params.get('alpha_L', 1.0)
        self.delta = params.get('delta_L', 0.95)
        self.penalty_scale = params.get('penalty_scale_L', 10.0)

    def economic_benefit(self, a_L_series):
        """Compute discounted economic benefit."""
        return sum(self.alpha * a * (self.delta ** t) for t, a in enumerate(a_L_series))

    def environmental_penalty(self, X_series, X_crit):
        """Compute discounted environmental penalty."""
        penalties = [(max(0, X - X_crit)) ** 2 for X in X_series]
        return -sum(self.penalty_scale * p * (self.delta ** t) for t, p in enumerate(penalties))

    def evaluate_objectives(self, a_L_series, X_series, X_crit):
        """Return both objective function values."""
        return {
            "f_econ_L": self.economic_benefit(a_L_series),
            "f_env_L": self.environmental_penalty(X_series, X_crit)
        }


# ============================================================
# 3. Environmental Regulator Model
# ============================================================

class Regulator:
    """
    Environmental regulator perspective: protect environment & sustain welfare.
    """

    def __init__(self, params):
        self.beta = params.get('beta_R', 0.5)
        self.delta = params.get('delta_R', 0.95)
        self.env_weight = params.get('env_weight_R', 20.0)

    def environmental_protection(self, X_series, X_crit):
        """Penalty for pollution exceeding threshold."""
        penalties = [(max(0, X - X_crit)) ** 2 for X in X_series]
        return -sum(self.env_weight * p * (self.delta ** t) for t, p in enumerate(penalties))

    def economic_welfare(self, a_L_series, a_R_series):
        """Social-economic welfare tied to activities."""
        welfare = [self.beta * (a_L + 0.5 * a_R) for a_L, a_R in zip(a_L_series, a_R_series)]
        return sum(w * (self.delta ** t) for t, w in enumerate(welfare))

    def evaluate_objectives(self, a_L_series, a_R_series, X_series, X_crit):
        """Return both objective function values."""
        return {
            "f_env_R": self.environmental_protection(X_series, X_crit),
            "f_econ_R": self.economic_welfare(a_L_series, a_R_series)
        }


# ============================================================
# 4. Unified Simulation Interface
# ============================================================

def simulate_lake_model(control_dict, uncertainty_dict, seed=None):
    """
    Simulate the unified lake problem dynamics under given controls and uncertainties.

    Inputs:
    -------
    control_dict : dict
        {
            "a_L": [list or array of local community emissions over time],
            "a_R": [list or array of regulator interventions over time]
        }

    uncertainty_dict : dict
        {
            "b": float, "q": float, "mu": float, "sigma": float,
            "delta_L": float, "delta_R": float,
            "alpha_L": float, "beta_R": float,
            "T": int
        }

    Returns:
    --------
    results : dict
        {
            "f_econ_L": ..., "f_env_L": ...,
            "f_env_R": ..., "f_econ_R": ...
        }
    """
    # Instantiate shared environment
    env = LakeEnvironment(uncertainty_dict)
    X_crit = env.critical_threshold()
    rng = np.random.default_rng(seed)

    # Instantiate stakeholder modules
    local = LocalCommunity(uncertainty_dict)
    regulator = Regulator(uncertainty_dict)

    # Controls
    a_L_series = control_dict.get("a_L", [0.0] * env.T)
    a_R_series = control_dict.get("a_R", [0.0] * env.T)

    # Simulation loop
    for t in range(env.T):
        eps_t = env.stochastic_inflow(rng)
        env.transition(a_L=a_L_series[t], a_R=a_R_series[t], epsilon_t=eps_t)

    X_series = env.X

    # Evaluate objectives
    local_results = local.evaluate_objectives(a_L_series, X_series, X_crit)
    regulator_results = regulator.evaluate_objectives(a_L_series, a_R_series, X_series, X_crit)

    # Combine results
    return X_series

# ============================================================
# Example (not executed)
# ============================================================
control_inputs = {
    "a_L": [0.03] * 100,
    "a_R": [0] * 100
}
uncertain_params = {
    "b": 0.42, "q": 2.0, "mu": 0.02, "sigma": 0.0017,
    "delta_L": 0.98, "delta_R": 0,
    "alpha_L": 0.4, "beta_R": 0,
    "T": 100
}