import numpy as np
from scipy.optimize import root_scalar

# ============================================================
# === Shared Environment =====================================
# ============================================================

class LakeEnvironment:
    """
    Shared environmental system for the Lake Problem.
    Handles pollution dynamics, natural stochastic inflow, and
    endogenous eutrophication threshold.
    """

    def __init__(self, b, q, mu, sigma, X0=0.0, T=50, seed=None):
        self.b = b
        self.q = q
        self.mu = mu           # desired lognormal mean
        self.sigma = sigma     # desired lognormal std (variance = sigma^2)
        self.T = T
        self.X0 = X0
        self.rng = np.random.default_rng(seed)
        self.Xcrit = self.compute_Xcrit()  # within [0.01, 1.5]

    # --------------------------------------------------------
    # Compute Xcrit using root-finding in [0.01, 1.5]
    # --------------------------------------------------------
    def compute_Xcrit(self):
        """
        Compute Xcrit in [0.01, 1.5] such that
        f(X) = X^q / (1 + X^q) - b * X = 0.
        Returns root or np.nan if not found.
        """
        def f(X):
            return (X**self.q) / (1 + X**self.q) - self.b * X

        try:
            sol = root_scalar(f, bracket=[0.01, 1.5], method="bisect")
            if sol.converged:
                return sol.root
            else:
                return np.nan
        except ValueError:
            # if no sign change in bracket
            return np.nan

    # --------------------------------------------------------
    # Draw lognormal natural inflow with correct parameterization
    # --------------------------------------------------------
    def draw_lognormal(self):
        """
        Draw epsilon_t ~ LogNormal(mean=mu, std=sigma)
        using proper conversion from lognormal mean/std
        to underlying normal parameters.
        """
        # Avoid invalid values
        mu = max(self.mu, 1e-12)
        sigma = max(self.sigma, 1e-12)

        s2 = np.log(1 + (sigma**2) / (mu**2))
        s = np.sqrt(s2)
        m = np.log(mu) - 0.5 * s2

        return self.rng.lognormal(mean=m, sigma=s)

    # --------------------------------------------------------
    # One-step transition
    # --------------------------------------------------------
    def step(self, X_t, a_C, a_R):
        eps_t = self.draw_lognormal()
        X_next = X_t + a_C + a_R + (X_t**self.q) / (1 + X_t**self.q) - self.b * X_t + eps_t
        return X_next, eps_t

    # --------------------------------------------------------
    # Full simulation
    # --------------------------------------------------------
    def simulate(self, a_C_series, a_R_series):
        X = np.zeros(self.T + 1)
        eps = np.zeros(self.T)
        X[0] = self.X0

        for t in range(self.T):
            X[t + 1], eps[t] = self.step(X[t], a_C_series[t], a_R_series[t])

        return X, eps


# ============================================================
# === Local Community ========================================
# ============================================================

class LocalCommunity:
    """
    Local community perspective: economic benefit vs. eutrophication risk.
    """
    def __init__(self, alpha_C, delta_C):
        self.alpha = alpha_C
        self.delta = delta_C

    def economic_benefit(self, a_C_series):
        T = len(a_C_series)
        return np.sum([self.alpha * a_C_series[t] * (self.delta**t) for t in range(T)])

    def environmental_safety(self, X_series, Xcrit):
        exceed = np.sum(X_series > Xcrit)
        return -exceed


# ============================================================
# === Environmental Regulator ================================
# ============================================================

class EnvironmentalRegulator:
    """
    Environmental regulator perspective: preservation, welfare, and policy cost.
    """
    def __init__(self, alpha_R, delta_R, c_R):
        self.alpha = alpha_R
        self.delta = delta_R
        self.c_R = c_R

    def environmental_preservation(self, X_series, Xcrit):
        exceed_amount = np.maximum(0, X_series - Xcrit)
        return -np.sum(exceed_amount)

    def economic_welfare(self, a_C_series):
        T = len(a_C_series)
        return np.sum([self.alpha * a_C_series[t] * (self.delta**t) for t in range(T)])

    def policy_cost(self, a_R_series):
        T = len(a_R_series)
        return -np.sum([self.c_R * (a_R_series[t]**2) * (self.delta**t) for t in range(T)])


# ============================================================
# === Unified Interface Function =============================
# ============================================================

def simulate_lake_problem(controls, uncertainties, T=100, seed=None):
    """
    Unified multi-perspective Lake Problem simulator.

    Parameters
    ----------
    controls : dict
        'a_C' : np.ndarray, community emissions (length T)
        'a_R' : np.ndarray, regulator interventions (length T)

    uncertainties : dict
        {
          'b': float,
          'q': float,
          'mu': float,
          'sigma': float,
          'alpha_C': float,
          'delta_C': float,
          'alpha_R': float,
          'delta_R': float,
          'c_R': float
        }

    Returns
    -------
    results : dict
        {
          'community': {
              'economic_benefit': float,
              'environmental_safety': float
          },
          'regulator': {
              'environmental_preservation': float,
              'economic_welfare': float,
              'policy_cost': float
          },
          'state': {
              'X': np.ndarray,
              'epsilon': np.ndarray,
              'Xcrit': float
          }
        }
    """

    # Unpack uncertainties
    b = uncertainties["b"]
    q = uncertainties["q"]
    mu = uncertainties["mu"]
    sigma = uncertainties["sigma"]

    alpha_C = uncertainties["alpha_C"]
    delta_C = uncertainties["delta_C"]
    alpha_R = uncertainties["alpha_R"]
    delta_R = uncertainties["delta_R"]
    c_R = uncertainties["c_R"]

    # Initialize shared environment
    env = LakeEnvironment(b=b, q=q, mu=mu, sigma=sigma, T=T, seed=seed)

    # Initialize stakeholders
    community = LocalCommunity(alpha_C, delta_C)
    regulator = EnvironmentalRegulator(alpha_R, delta_R, c_R)

    # Extract controls
    a_C_series = np.asarray(controls["a_C"])
    a_R_series = np.asarray(controls["a_R"])

    # Simulate shared environment
    X_series, eps_series = env.simulate(a_C_series, a_R_series)

    # Evaluate objectives
    results = {
        "community": {
            "economic_benefit": community.economic_benefit(a_C_series),
            "environmental_safety": community.environmental_safety(X_series, env.Xcrit)
        },
        "regulator": {
            "environmental_preservation": regulator.environmental_preservation(X_series, env.Xcrit),
            "economic_welfare": regulator.economic_welfare(a_C_series),
            "policy_cost": regulator.policy_cost(a_R_series)
        },
        "state": {
            "X": X_series,
            "epsilon": eps_series,
            "Xcrit": env.Xcrit
        }
    }

    return results


controls = {
    "a_C": np.full(100, 0.03),
    "a_R": np.full(100, 0)
}

uncertainties = {
    "b": 0.42,
    "q": 2.0,
    "mu": 0.02,
    "sigma": 0.0017,
    "alpha_C": 0.4,
    "delta_C": 0.98,
    "alpha_R": 0,
    "delta_R": 0,
    "c_R": 0
}