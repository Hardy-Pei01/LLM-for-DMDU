import numpy as np
from scipy.optimize import brentq

# ============================================================
# Common Environment: Lake Dynamics
# ============================================================

class LakeEnvironment:
    def __init__(self, params):
        """
        Shared lake environment with stochastic pollution inflow.

        params: dict containing uncertain and constant parameters
            Required keys:
                ['alpha', 'T', 'b', 'q', 'mu', 'sigma', 'delta',
                 'X_safe', 'beta', 'c_R', 'c_S']
        """
        self.params = params
        self.T = params['T']
        self.b = params['b']
        self.q = params['q']
        self.mu = params['mu']
        self.sigma = params['sigma']
        self.X_safe = params.get('X_safe', 0.5)
        self.X_crit = self._compute_Xcrit()
        self._set_lognormal_parameters()

    # --------------------------------------------------------
    # 1. Compute critical threshold X_crit by root finding
    # --------------------------------------------------------
    def _compute_Xcrit(self):
        """Find X_crit such that (X^q)/(1 + X^q) - bX = 0."""
        b, q = self.b, self.q
        def f(X):
            return (X**q) / (1 + X**q) - b * X
        # Root in the plausible range [0.01, 1.5]
        try:
            X_crit = brentq(f, 0.01, 1.5)
        except ValueError:
            # Fallback if root not found
            X_crit = 1.0
        return X_crit

    # --------------------------------------------------------
    # 2. Prepare parameters for correct lognormal generation
    # --------------------------------------------------------
    def _set_lognormal_parameters(self):
        """
        Convert mean (μ) and variance (σ²) in real space to
        meanlog (μ') and stdlog (σ') in log space for NumPy.
        """
        mu, sigma = self.mu, self.sigma
        if mu <= 0:
            mu = 1e-6  # ensure positivity
        self.meanlog = np.log(mu**2 / np.sqrt(sigma**2 + mu**2))
        self.stdlog = np.sqrt(np.log(1 + (sigma**2 / mu**2)))

    def natural_inflow(self):
        """Generate a stochastic natural inflow ε_t ~ LogNormal(μ, σ²)."""
        return np.random.lognormal(mean=self.meanlog, sigma=self.stdlog)

    # --------------------------------------------------------
    # 3. Lake dynamics
    # --------------------------------------------------------
    def step(self, X_t, emissions, mitigations, epsilon_t):
        """
        Advance lake pollution state by one timestep.

        X_t        : current pollution level
        emissions  : total emissions (a_E + a_C)
        mitigations: total mitigation (m_R + m_S)
        epsilon_t  : stochastic natural inflow
        """
        b, q = self.b, self.q
        X_next = X_t + (emissions - mitigations) + (X_t**q) / (1 + X_t**q) - b * X_t + epsilon_t
        return max(X_next, 0.0)


# ============================================================
# Stakeholder Classes
# ============================================================

class EconomicActor:
    """Industry/Agriculture perspective."""
    def __init__(self, alpha, delta):
        self.alpha = alpha
        self.delta = delta

    def objective(self, a_E):
        """Economic objective: discounted benefit from emissions."""
        T = len(a_E)
        return np.sum([self.alpha * a_E[t] * (self.delta ** t) for t in range(T)])


class Regulator:
    """Environmental regulator perspective."""
    def __init__(self, c_R=0.0):
        self.c_R = c_R

    def objectives(self, X, m_R, X_crit):
        """Return pollution-related objectives for the regulator."""
        f_pollution = np.sum(X)
        f_risk = np.max(np.maximum(X - X_crit, 0))
        f_cost = np.sum(self.c_R * np.square(m_R))
        return {"pollution": f_pollution, "risk": f_risk, "cost": f_cost}


class Community:
    """Local community perspective."""
    def __init__(self, alpha, beta, delta):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def objectives(self, a_C, X, X_safe):
        """Return welfare and quality objectives for the community."""
        T = len(a_C)
        welfare = np.sum([(self.alpha * a_C[t] - self.beta * X[t]) * (self.delta ** t)
                          for t in range(T)])
        quality = np.sum(np.maximum(X - X_safe, 0))
        return {"welfare": welfare, "quality": quality}


class SustainabilityAdvocate:
    """Future generations / long-term sustainability perspective."""
    def __init__(self, c_S=0.0):
        self.c_S = c_S

    def objectives(self, X, m_S, X_crit):
        """Return risk, resilience, and cost objectives."""
        f_risk = float(np.any(np.array(X) >= X_crit))  # 1 if eutrophication occurs
        f_resilience = -np.mean(X[-10:]) if len(X) > 10 else -X[-1]
        f_cost = np.sum(self.c_S * np.square(m_S))
        return {"risk": f_risk, "resilience": f_resilience, "cost": f_cost}


# ============================================================
# Unified Simulation Interface
# ============================================================

def simulate_lake_system(controls, uncertainties, seed=None):
    """
    Unified simulation interface for the multi-perspective lake model.

    Parameters
    ----------
    controls : dict
        {
            'a_E': [float],  # emissions by Economic actor
            'm_R': [float],  # mitigation by Regulator
            'a_C': [float],  # emissions by Community
            'm_S': [float],  # mitigation by Sustainability advocate
        }

    uncertainties : dict
        {
            'alpha', 'T', 'b', 'q', 'mu', 'sigma',
            'delta', 'X_safe', 'beta', 'c_R', 'c_S'
        }

    Returns
    -------
    dict
        Objective values for each perspective.
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize environment and compute X_crit internally
    env = LakeEnvironment(uncertainties)

    econ = EconomicActor(alpha=uncertainties['alpha'], delta=uncertainties['delta'])
    reg = Regulator(c_R=uncertainties['c_R'])
    comm = Community(alpha=uncertainties['alpha'], beta=uncertainties['beta'],
                     delta=uncertainties['delta'])
    sust = SustainabilityAdvocate(c_S=uncertainties['c_S'])

    # Retrieve control trajectories
    a_E = np.array(controls['a_E'])
    m_R = np.array(controls['m_R'])
    a_C = np.array(controls['a_C'])
    m_S = np.array(controls['m_S'])
    T = uncertainties['T']

    X = np.zeros(T + 1)

    # Simulate lake dynamics
    for t in range(T):
        eps_t = env.natural_inflow()
        emissions = a_E[t] + a_C[t]
        mitigations = m_R[t] + m_S[t]
        X[t + 1] = env.step(X[t], emissions, mitigations, eps_t)

    # Compute objectives
    f_E = {"economic": econ.objective(a_E)}
    f_R = reg.objectives(X, m_R, env.X_crit)
    f_C = comm.objectives(a_C, X, env.X_safe)
    f_S = sust.objectives(X, m_S, env.X_crit)

    return {
        "Economic": f_E,
        "Regulator": f_R,
        "Community": f_C,
        "Sustainability": f_S,
        "X_crit": env.X_crit  # include for reference
    }

# ============================================================
# Example Usage (for illustration; not executed)
# ============================================================
"""
controls = {
    'a_E': [0.5] * 100,
    'm_R': [0.2] * 100,
    'a_C': [0.3] * 100,
    'm_S': [0.1] * 100
}

uncertainties = {
    'alpha': 1.0,
    'T': 100,
    'b': 0.4,
    'q': 2.0,
    'mu': 0.5,
    'sigma': 0.2,
    'delta': 0.95,
    'X_safe': 0.5,
    'beta': 0.3,
    'c_R': 0.1,
    'c_S': 0.1
}

results = simulate_lake_system(controls, uncertainties, seed=42)
print(results)
"""
