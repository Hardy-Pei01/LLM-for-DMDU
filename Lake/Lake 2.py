import numpy as np
from scipy.optimize import brentq

# ============================================================
# 1. Shared Lake Environment
# ============================================================

class LakeEnvironment:
    """
    Represents the shared lake environment and its pollution dynamics.
    Handles stochastic inflow and endogenous eutrophication threshold.
    """

    def __init__(self, b, q, mu, sigma, delta, alpha_params, eta_S, gamma_C, X0=0.0, T=50):
        self.b = b                # natural removal rate
        self.q = q                # recycling rate
        self.mu = mu              # mean of actual lognormal inflow
        self.sigma = sigma        # std dev of actual lognormal inflow
        self.delta = delta        # discount rate
        self.alpha = alpha_params # dict of benefit coefficients
        self.eta_S = eta_S        # social mitigation efficiency
        self.gamma_C = gamma_C    # community scaling factor
        self.X0 = X0
        self.T = T

    # ---------------------------------------------------------
    # Compute endogenous eutrophication threshold
    # ---------------------------------------------------------
    def x_crit(self):
        """
        Compute X_crit(b,q) within [0.01, 1.5], where X^q/(1+X^q) = bX.
        """
        f = lambda X: (X ** self.q) / (1 + X ** self.q) - self.b * X
        try:
            return brentq(f, 0.01, 1.5)
        except ValueError:
            # If no root in range, clamp to nearest boundary
            return 0.01 if f(0.01) * f(1.5) > 0 else 1.5

    # ---------------------------------------------------------
    # Generate stochastic natural inflow
    # ---------------------------------------------------------
    def natural_inflow(self):
        """
        Draw stochastic inflow ε_t from a lognormal distribution
        with mean=self.mu and std=self.sigma (in real space).
        """
        # Convert desired mean (μ) and std (σ) to log-space parameters
        # Let m, s be mean and std of underlying normal distribution:
        # μ = exp(m + s^2 / 2),  σ^2 = (exp(s^2) - 1) * exp(2m + s^2)
        variance = self.sigma ** 2
        s2 = np.log(1 + variance / (self.mu ** 2))
        s = np.sqrt(s2)
        m = np.log(self.mu) - 0.5 * s2
        return np.random.lognormal(mean=m, sigma=s)

    # ---------------------------------------------------------
    # One-step state transition
    # ---------------------------------------------------------
    def step(self, X_t, controls):
        """
        One-step pollution update:
        controls = dict with keys ['a_E', 'r', 's', 'c']
        """
        a_E = controls.get('a_E', 0.0)
        r = controls.get('r', 0.0)
        s = controls.get('s', 0.0)
        c = controls.get('c', 0.0)

        inflow_stochastic = self.natural_inflow()

        # Net anthropogenic pollution inflow
        a_net = a_E * (1 - r) + self.gamma_C * c - self.eta_S * s

        # Pollution dynamics
        X_next = X_t + a_net + (X_t ** self.q) / (1 + X_t ** self.q) - self.b * X_t + inflow_stochastic
        return max(X_next, 0.0)  # ensure nonnegative pollution


# ============================================================
# 2. Perspective Classes
# ============================================================

class EconomicPlanner:
    """Industry / economic growth perspective."""
    def __init__(self, alpha_E):
        self.alpha_E = alpha_E

    def objective(self, controls, delta):
        a_E = np.array(controls['a_E'])
        T = len(a_E)
        return np.sum(self.alpha_E * a_E * (delta ** np.arange(T)))


class EnvironmentalRegulator:
    """Environmental protection perspective."""
    def ecological_loss(self, X, Xcrit):
        return np.sum(np.maximum(0, X - Xcrit) ** 2)

    def reliability(self, X, Xcrit):
        return np.mean(X < Xcrit)


class SocialPlanner:
    """Integrated welfare perspective (multi-objective)."""
    def __init__(self, alpha_S):
        self.alpha_S = alpha_S

    def economic_benefit(self, a_E, r, delta):
        eff = np.array(a_E) * (1 - np.array(r))
        T = len(a_E)
        return np.sum(self.alpha_S * eff * (delta ** np.arange(T)))

    def environmental_safety(self, X, Xcrit):
        return -np.sum(np.maximum(0, X - Xcrit) ** 2)

    def equity(self, X):
        return -np.var(X)


class LocalCommunity:
    """Local adaptive perspective."""
    def __init__(self, alpha_C, lambda_C):
        self.alpha_C = alpha_C
        self.lambda_C = lambda_C

    def livelihood(self, c, delta):
        c = np.array(c)
        T = len(c)
        return np.sum(self.alpha_C * (1 - np.abs(c)) * (delta ** np.arange(T))) - self.lambda_C * np.var(c)

    def eutrophication_risk(self, X, Xcrit):
        return np.maximum(0, X[-1] - Xcrit)


# ============================================================
# 3. Unified Simulation Interface
# ============================================================

def simulate_lake_model(control_dict, param_dict):
    """
    Simulate the unified lake model with given controls and uncertain parameters.

    Inputs:
        control_dict: dict of control time series:
            {'a_E': [...], 'r': [...], 's': [...], 'c': [...]}
        param_dict: dict of uncertain parameters and constants:
            {'b','q','mu','sigma','delta',
             'alpha_E','alpha_S','alpha_C',
             'eta_S','gamma_C','lambda_C','T'}

    Returns:
        results: dict of all objective values and final pollution trajectory.
    """

    # Unpack parameters
    b = param_dict['b']
    q = param_dict['q']
    mu = param_dict['mu']
    sigma = param_dict['sigma']
    delta = param_dict['delta']
    alpha_E = param_dict['alpha_E']
    alpha_S = param_dict['alpha_S']
    alpha_C = param_dict['alpha_C']
    eta_S = param_dict['eta_S']
    gamma_C = param_dict['gamma_C']
    lambda_C = param_dict['lambda_C']
    T = param_dict['T']

    alpha_params = {'E': alpha_E, 'S': alpha_S, 'C': alpha_C}

    # Initialize environment and perspectives
    env = LakeEnvironment(b, q, mu, sigma, delta, alpha_params, eta_S, gamma_C, X0=0.0, T=T)
    econ = EconomicPlanner(alpha_E)
    regulator = EnvironmentalRegulator()
    social = SocialPlanner(alpha_S)
    community = LocalCommunity(alpha_C, lambda_C)

    # Compute critical threshold (bounded)
    Xcrit = env.x_crit()

    # Simulate pollution trajectory
    X = np.zeros(T + 1)
    for t in range(T):
        controls_t = {k: control_dict[k][t] for k in ['a_E', 'r', 's', 'c']}
        X[t + 1] = env.step(X[t], controls_t)

    # Compute objective values
    f_E = econ.objective(control_dict, delta)
    f_R_eco = regulator.ecological_loss(X, Xcrit)
    f_R_rel = regulator.reliability(X, Xcrit)
    f_S_econ = social.economic_benefit(control_dict['a_E'], control_dict['r'], delta)
    f_S_env = social.environmental_safety(X, Xcrit)
    f_S_equity = social.equity(X)
    f_C_life = community.livelihood(control_dict['c'], delta)
    f_C_risk = community.eutrophication_risk(X, Xcrit)

    # Return results
    results = {
        'EconomicPlanner': {'EconomicBenefit': f_E},
        'EnvironmentalRegulator': {'EcologicalLoss': f_R_eco,
                                   'Reliability': f_R_rel},
        'SocialPlanner': {'EconomicBenefit': f_S_econ,
                          'EnvironmentalSafety': f_S_env,
                          'Equity': f_S_equity},
        'LocalCommunity': {'Livelihood': f_C_life,
                           'EutrophicationRisk': f_C_risk},
        'FinalPollution': X[-1],
        'Trajectory': X
    }

    return results
