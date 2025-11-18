import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# Shared Environment: Lake System
# ============================================================

class LakeEnvironment:
    """
    Shared stochastic lake environment.
    Evolves pollution stock X_t according to both agents' actions and natural inflows.
    """

    def __init__(self, params):
        """
        params: dict containing global/environment parameters.
            {
                'b': natural removal rate,
                'q': natural recycling rate,
                'mu': mean of lognormal inflow (lognormal mean),
                'sigma': std. dev. of lognormal inflow (lognormal std),
                'T': time horizon,
                'X0': initial pollution,
            }
        """
        self.b = params["b"]
        self.q = params["q"]
        self.mu = params["mu"]
        self.sigma = params["sigma"]
        self.T = params["T"]
        self.X0 = params.get("X0", 0.0)

        # Convert lognormal (mean, std) → underlying normal (mean, std)
        self.norm_mean, self.norm_std = self._lognormal_to_normal(self.mu, self.sigma)

        # Compute critical threshold within [0.01, 1.5]
        self.X_crit = self.compute_critical_threshold()

    def _lognormal_to_normal(self, mu, sigma):
        """
        Convert lognormal mean (mu) and std (sigma)
        into parameters of the underlying normal distribution.
        """
        if mu <= 0:
            raise ValueError("Mean (mu) of lognormal inflow must be positive.")
        s = np.sqrt(np.log(1 + (sigma ** 2) / (mu ** 2)))
        m = np.log(mu) - 0.5 * s**2
        return m, s

    def compute_critical_threshold(self):
        """
        Computes X_crit by solving (x^q)/(1+x^q) = b*x within [0.01, 1.5].
        """
        b, q = self.b, self.q
        def f(x): return (x ** q) / (1 + x ** q) - b * x
        # Check signs before root finding
        f_low, f_high = f(0.01), f(1.5)
        if f_low * f_high > 0:
            # No root in range — fallback to boundary that minimizes |f(x)|
            grid = np.linspace(0.01, 1.5, 200)
            vals = [abs(f(xx)) for xx in grid]
            return grid[np.argmin(vals)]
        else:
            return brentq(f, 0.01, 1.5)

    def sample_inflow(self):
        """
        Draws a stochastic natural inflow ε_t from lognormal(mean=μ, std=σ).
        Uses derived normal parameters (mean, std).
        """
        return np.random.lognormal(self.norm_mean, self.norm_std)

    def step(self, X_t, a_L, a_R):
        """
        Advances the lake state one period forward.
        X_{t+1} = X_t + a_L + a_R + (X_t^q)/(1+X_t^q) - b*X_t + ε_t
        """
        epsilon_t = self.sample_inflow()
        X_next = X_t + a_L + a_R + (X_t ** self.q) / (1 + X_t ** self.q) - self.b * X_t + epsilon_t
        return X_next, epsilon_t


# ============================================================
# Perspective 1: Local Community
# ============================================================

class LocalCommunity:
    """
    Local community agent with control over anthropogenic emissions.
    """
    def __init__(self, params):
        """
        params: dict containing community-specific parameters
            {
                'alpha': benefit-to-pollution ratio,
                'beta': penalty weight for environmental loss,
                'delta': discount factor,
                'X_safe': safe pollution level,
            }
        """
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.delta = params["delta"]
        self.X_safe = params["X_safe"]

    def objectives(self, X_series, A_L_series):
        """
        Compute economic and environmental objectives for the local community.
        """
        T = len(A_L_series)
        f_econ = [self.alpha * A_L_series[t] * (self.delta ** t) for t in range(T)]
        f_env = -sum(self.beta * max(0, X_series[t] - self.X_safe) for t in range(T))
        return {"economic": f_econ, "environmental": f_env}


# ============================================================
# Perspective 2: Environmental Regulator
# ============================================================

class Regulator:
    """
    Regulator agent controlling policy interventions (e.g., emissions offsets).
    """
    def __init__(self, params):
        """
        params: dict containing regulator-specific parameters
            {
                'gamma': penalty for exceeding X_crit,
                'eta': weight on economic activity,
                'delta': discount factor,
            }
        """
        self.gamma = params["gamma"]
        self.eta = params["eta"]
        self.delta = params["delta"]

    def objectives(self, X_series, A_L_series, X_crit):
        """
        Compute environmental and economic objectives for the regulator.
        """
        T = len(A_L_series)
        f_env = [self.gamma * max(0, X_series[t] - X_crit) for t in range(T)]
        f_econ = sum(self.eta * A_L_series[t] * (self.delta ** t) for t in range(T))
        return {"environmental": f_env, "economic": f_econ}


# ============================================================
# Unified Multi-Agent Lake Problem Simulation
# ============================================================

def simulate_lake_problem(control_dict, uncertainty_dict):
    """
    Interface function to simulate the unified multi-agent lake model.

    Inputs:
    -------
    control_dict: dict of control time series for each stakeholder
        {
            'a_L': [a_L0, a_L1, ..., a_LT-1],
            'a_R': [a_R0, a_R1, ..., a_RT-1]
        }

    uncertainty_dict: dict of uncertain/global parameter values
        {
            'b': ..., 'q': ..., 'mu': ..., 'sigma': ..., 'T': ..., 'X0': ...,
            'alpha_L': ..., 'beta_L': ..., 'delta_L': ..., 'X_safe': ...,
            'gamma_R': ..., 'eta_R': ..., 'delta_R': ...
        }

    Output:
    -------
    dict containing objective function values for each perspective:
        {
            'LocalCommunity': {'economic': ..., 'environmental': ...},
            'Regulator': {'environmental': ..., 'economic': ...},
            'Environment': {'X_series': [...], 'X_crit': ...}
        }
    """

    # -----------------------------
    # 1. Initialize Environment
    # -----------------------------
    env_params = {
        "b": uncertainty_dict["b"],
        "q": uncertainty_dict["q"],
        "mu": uncertainty_dict["mu"],
        "sigma": uncertainty_dict["sigma"],
        "T": uncertainty_dict["T"],
        "X0": uncertainty_dict.get("X0", 0.0),
    }
    env = LakeEnvironment(env_params)

    # -----------------------------
    # 2. Initialize Agents
    # -----------------------------
    community_params = {
        "alpha": uncertainty_dict["alpha_L"],
        "beta": uncertainty_dict["beta_L"],
        "delta": uncertainty_dict["delta_L"],
        "X_safe": uncertainty_dict["X_safe"],
    }
    regulator_params = {
        "gamma": uncertainty_dict["gamma_R"],
        "eta": uncertainty_dict["eta_R"],
        "delta": uncertainty_dict["delta_R"],
    }
    community = LocalCommunity(community_params)
    regulator = Regulator(regulator_params)

    # -----------------------------
    # 3. Simulate Dynamics
    # -----------------------------
    T = env.T
    X = np.zeros(T + 1)
    X[0] = env.X0
    A_L = np.array(control_dict["a_L"])
    A_R = np.array(control_dict["a_R"])

    for t in range(T):
        X[t + 1], _ = env.step(X[t], A_L[t], A_R[t])

    # -----------------------------
    # 4. Evaluate Objectives
    # -----------------------------
    results_L = community.objectives(X[:-1], A_L)
    results_R = regulator.objectives(X[:-1], A_L, env.X_crit)

    # -----------------------------
    # 5. Return Structured Results
    # -----------------------------
    return {
        "LocalCommunity": results_L,
        "Regulator": results_R,
        "Environment": {
            "X_series": X.tolist(),
            "X_crit": env.X_crit,
        },
    }


a_L = [0.0306, 0.0485, 0.0373, 0.0375, 0.0477, 0.0466, 0.038, 0.0318, 0.0442, 0.0381, 0.0362, 0.0279, 0.0433,
       0.0394, 0.0358, 0.0409, 0.0407, 0.036, 0.0413, 0.0375, 0.0328, 0.0417, 0.0447, 0.0324, 0.0397, 0.0402,
       0.0389, 0.046, 0.0378, 0.0345, 0.0355, 0.0385, 0.036, 0.0389, 0.0328, 0.0364, 0.0453, 0.0392, 0.0386,
       0.0386, 0.0375, 0.0456, 0.0354, 0.0315, 0.0479, 0.0373, 0.0302, 0.0444, 0.0316, 0.0454, 0.0381, 0.0442,
       0.0426, 0.0474, 0.0373, 0.0383, 0.0385, 0.0376, 0.0385, 0.0388, 0.0399, 0.0398, 0.0391, 0.0371, 0.0388,
       0.0418, 0.0499, 0.0418, 0.0508, 0.0352, 0.0449, 0.043, 0.0344, 0.0434, 0.053, 0.0355, 0.0444, 0.0375,
       0.0443, 0.0342, 0.0257, 0.0328, 0.0409, 0.0302, 0.041, 0.041, 0.0346, 0.047, 0.0398, 0.038, 0.038,
       0.0429, 0.0396, 0.0403, 0.0397, 0.0406, 0.0342, 0.0421, 0.043, 0.0341]
# a_L = [0.03]*100
a_R = [-0.01]*100

control_inputs = {
    'a_L': a_L,
    'a_R': a_R
}
uncertain_params = {
    'b': 0.42, 'q': 2.0, 'mu': 0.02, 'sigma': 0.0017, 'T': 100, 'X0': 0.0,
    'alpha_L': 0.4, 'beta_L': 1, 'delta_L': 0.98, 'X_safe': 0,
    'gamma_R': 1, 'eta_R': 0, 'delta_R': 0
}

pollution = []
community_benefit = []
eutrophication = []
for i in range(40):
    results = simulate_lake_problem(control_inputs, uncertain_params)
    pollution.append(results["Environment"]["X_series"])
    community_benefit.append(results["LocalCommunity"]["economic"])
    eutrophication.append(results["Regulator"]["environmental"])


def envelope_diagram(value_list, y_label, title):
    value_df = pd.DataFrame(value_list)
    mean_list = value_df.mean()
    std_list = value_df.std()
    time_steps = [i for i in range(0, len(value_list[0]))]

    plt.plot(time_steps, mean_list, label='Mean', color='red')

    # 5. Plot the standard deviation as a shaded region (confidence interval)
    # You can choose to plot 1 standard deviation, 2 standard deviations, etc.
    plt.fill_between(time_steps, mean_list - std_list, mean_list + std_list,
                     color='blue', alpha=0.2, label='Standard deviation')

    # 6. Customize the plot
    plt.xlabel('Time steps', size=14)
    plt.ylabel(y_label, size=14)
    plt.title(title, size=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()

envelope_diagram(pollution, 'Current lake pollution', 'Average Time Series of Current Lake Pollution')
# envelope_diagram(community_benefit, 'Current economic benefit', 'Average Time Series of Current Economic Benefit')
# envelope_diagram(eutrophication, 'Current eutrophication loss', 'Average Time Series of Current Eutrophication Loss')