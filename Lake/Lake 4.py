# ================================================================
# Refined Unified Modular Lake Problem Model
# ------------------------------------------------
# - Corrects X_crit root finding within [0.01, 1.5]
# - Corrects lognormal sampling for mean μ and variance σ²
# ================================================================

import numpy as np
from scipy.optimize import brentq

# ------------------------------------------------
# Shared Environment Class
# ------------------------------------------------
class LakeEnvironment:
    def __init__(self, params, T=100, seed=None):
        """
        params: dictionary of uncertain parameters
            keys: ['mu', 'sigma', 'b', 'q', 'delta']
        T: simulation horizon
        seed: random seed for reproducibility
        """
        self.mu = params.get('mu', 0.02)
        self.sigma = params.get('sigma', 0.01)
        self.b = params.get('b', 0.42)
        self.q = params.get('q', 2.0)
        self.delta = params.get('delta', 0.98)
        self.T = T
        self.X = np.zeros(T + 1)

        # initialize random inflow generator (correct lognormal mean/var)
        rng = np.random.default_rng(seed)
        self.eps = self._generate_lognormal_inflow(rng)

    def _generate_lognormal_inflow(self, rng):
        """
        Generate lognormal inflows with specified mean (mu)
        and variance (sigma^2) in the lognormal space.
        """
        mu, sigma = self.mu, self.sigma
        # convert to normal parameters ψ, φ
        phi = np.sqrt(np.log(1 + (sigma**2 / mu**2)))
        psi = np.log(mu) - 0.5 * phi**2
        return rng.lognormal(mean=psi, sigma=phi, size=self.T)

    def X_crit(self):
        """Compute X_crit in the bounded range [0.01, 1.5]."""
        def f(X):
            return (X**self.q) / (1 + X**self.q) - self.b * X

        try:
            Xc = brentq(f, 0.01, 1.5)
        except ValueError:
            # fallback if no root in range
            Xc = 1.5
        return Xc

    def update(self, emissions, mitigation, t):
        """Update lake pollution given emissions, mitigation, and random inflow."""
        Xt = self.X[t]
        next_X = Xt + emissions - mitigation + (Xt**self.q) / (1 + Xt**self.q) - self.b * Xt + self.eps[t]
        self.X[t + 1] = max(next_X, 0.0)
        return self.X[t + 1]


# ------------------------------------------------
# Stakeholder Base Class
# ------------------------------------------------
class Stakeholder:
    def __init__(self, name, delta):
        self.name = name
        self.delta = delta
        self.objectives = {}

    def discount(self, t):
        return self.delta ** t


# ------------------------------------------------
# Economic Developer
# ------------------------------------------------
class EconomicDeveloper(Stakeholder):
    def __init__(self, alpha_E, delta):
        super().__init__("Economic Developer", delta)
        self.alpha_E = alpha_E

    def effective_emission(self, a_E, u_E):
        return min(a_E, u_E)

    def objective(self, emissions):
        val = sum(self.alpha_E * e * self.discount(t) for t, e in enumerate(emissions))
        self.objectives["economic_benefit"] = val
        return self.objectives


# ------------------------------------------------
# Environmental Regulator
# ------------------------------------------------
class EnvironmentalRegulator(Stakeholder):
    def __init__(self, alpha_R, delta):
        super().__init__("Environmental Regulator", delta)
        self.alpha_R = alpha_R

    def objectives_fn(self, emissions, X, Xcrit):
        econ_support = sum(self.alpha_R * sum(e) * self.discount(t) for t, e in enumerate(emissions))
        env_protect = -np.mean(X >= Xcrit)
        self.objectives = {
            "economic_support": econ_support,
            "environmental_protection": env_protect,
        }
        return self.objectives


# ------------------------------------------------
# Local Community
# ------------------------------------------------
class LocalCommunity(Stakeholder):
    def __init__(self, alpha_H, beta_H, delta):
        super().__init__("Local Community", delta)
        self.alpha_H = alpha_H
        self.beta_H = beta_H

    def objectives_fn(self, industrial_emissions, X):
        econ_welfare = sum(self.alpha_H * e * self.discount(t) for t, e in enumerate(industrial_emissions))
        env_quality = -sum(self.beta_H * x * self.discount(t) for t, x in enumerate(X))
        self.objectives = {
            "economic_wellbeing": econ_welfare,
            "environmental_quality": env_quality,
        }
        return self.objectives


# ------------------------------------------------
# Sustainability Advocate (Mitigation / Offset)
# ------------------------------------------------
class SustainabilityAdvocate(Stakeholder):
    def __init__(self, gamma_S, delta):
        super().__init__("Sustainability Advocate", delta)
        self.gamma_S = gamma_S

    def effective_mitigation(self, a_S, c_S):
        return min(a_S, c_S)

    def objectives_fn(self, X, mitigation, Xcrit):
        preserve = -np.mean(X >= Xcrit)
        mitigation_cost = -sum(self.gamma_S * m * self.discount(t) for t, m in enumerate(mitigation))
        self.objectives = {
            "ecological_preservation": preserve,
            "mitigation_cost": mitigation_cost,
        }
        return self.objectives


# ------------------------------------------------
# Unified Simulation Interface
# ------------------------------------------------
def simulate_lake_system(controls, params, T=100, seed=None):
    """
    Unified interface to simulate the lake problem.

    Inputs:
        controls: dict of control variables
            {
              'a_E': [...],      # emissions by developer
              'a_H': [...],      # emissions by households
              'a_S': [...],      # mitigation by sustainability advocate
              'u_R_E': [...],    # regulator cap (industry)
              'u_R_H': [...],    # regulator cap (households)
              'c_S': [...]       # mitigation capacity
            }

        params: dict of uncertain parameters
            {
              'mu', 'sigma', 'b', 'q', 'delta',
              'alpha_E', 'alpha_R', 'alpha_H', 'beta_H', 'gamma_S'
            }

        T: time horizon
        seed: RNG seed for reproducibility

    Returns:
        dict with objective values for each perspective
    """

    # --- initialize environment ---
    env = LakeEnvironment(params, T=T, seed=seed)
    Xcrit = env.X_crit()

    # --- initialize stakeholders ---
    dev = EconomicDeveloper(params['alpha_E'], env.delta)
    reg = EnvironmentalRegulator(params['alpha_R'], env.delta)
    com = LocalCommunity(params['alpha_H'], params['beta_H'], env.delta)
    sus = SustainabilityAdvocate(params['gamma_S'], env.delta)

    # --- utility function to vectorize controls ---
    def arr(key):
        val = controls.get(key, 0)
        return np.array(val if isinstance(val, (list, np.ndarray)) else [val] * T)

    # retrieve control arrays
    a_E, a_H, a_S = arr('a_E'), arr('a_H'), arr('a_S')
    u_R_E, u_R_H = arr('u_R_E'), arr('u_R_H')
    c_S = arr('c_S')

    # storage for effective emissions and mitigation
    e_E, e_H, m_S = [], [], []

    # --- simulate dynamics ---
    for t in range(T):
        eE = dev.effective_emission(a_E[t], u_R_E[t])
        eH = min(a_H[t], u_R_H[t])
        mS = sus.effective_mitigation(a_S[t], c_S[t])

        env.update(eE + eH, mS, t)

        e_E.append(eE)
        e_H.append(eH)
        m_S.append(mS)

    # --- compute objectives for each stakeholder ---
    obj_dev = dev.objective(e_E)
    obj_reg = reg.objectives_fn(list(zip(e_E, e_H)), env.X, Xcrit)
    obj_com = com.objectives_fn(e_E, env.X)
    obj_sus = sus.objectives_fn(env.X, m_S, Xcrit)

    # --- compile results ---
    return {
        "EconomicDeveloper": obj_dev,
        "EnvironmentalRegulator": obj_reg,
        "LocalCommunity": obj_com,
        "SustainabilityAdvocate": obj_sus,
        "X_crit": Xcrit,
    }

# ================================================================
# Example (not executed):
# results = simulate_lake_system(control_dict, param_dict, T=100)
# ================================================================
