import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# 1. SHARED MARKET ENVIRONMENT
# ============================================================

class MarketEnvironment:
    """
    Shared environment representing the day-ahead electricity market.
    Stores stochastic demand, conventional bids, renewable uncertainty,
    and market-clearing logic.
    """

    def __init__(self, params, uncertainties):
        """
        params: dict of fixed parameters (means, variances, time horizon, etc.)
        uncertainties: dict of uncertain variables (e.g., realized demand, conventional bids)
        """
        self.params = params
        self.uncertainties = uncertainties
        self.state = {}

    def sample_demand(self, t):
        mu_D, sigma_D = self.params["mu_D"], self.params["sigma_D"]
        return np.random.normal(mu_D, sigma_D)

    def sample_conventional_bids(self):
        """Generate random bids from normal distributions for conventional producers."""
        N_c = self.params["N_c"]
        bids = []
        for i in range(N_c):
            b_i = np.random.normal(self.params["mu_b"][i], self.params["sigma_b"][i])
            p_i = np.random.normal(self.params["mu_p"][i], self.params["sigma_p"][i])
            bids.append((b_i, p_i))
        return bids

    def market_clearing_price(self, D_t, conv_bids, b_t, p_t):
        """
        Solve for clearing price c_t by sorting offers until cumulative supply meets demand.
        """
        offers = conv_bids + [(b_t, p_t)]
        offers.sort(key=lambda x: x[1])  # sort by price ascending
        supply = 0
        for b_i, p_i in offers:
            supply += b_i
            if supply >= D_t:
                return p_i  # clearing price when supply meets demand
        return offers[-1][1]  # fallback: highest offer price

    def update_state(self, t, D_t, conv_bids, c_t, q_u, q_o):
        """Update global state after clearing."""
        self.state[t] = {
            "D_t": D_t,
            "conv_bids": conv_bids,
            "c_t": c_t,
            "q_u": q_u,
            "q_o": q_o,
        }


# ============================================================
# 2. RENEWABLE PRODUCER (Agent R)
# ============================================================

class RenewableProducer:
    """
    Renewable producer deciding quantity and price bids under deep uncertainty.
    """

    def __init__(self, env: MarketEnvironment, controls):
        self.env = env
        self.controls = controls  # dict with {t: (b_t, p_t)}
        self.results = {}

    def profit(self, b_t, p_t, c_t, P_t, q_u, q_o):
        """Hourly profit given realized generation and penalties."""
        if p_t > c_t:
            return 0.0  # bid not accepted
        revenue = c_t * b_t
        penalty_under = q_u * max(b_t - P_t, 0)
        penalty_over = q_o * max(P_t - b_t, 0)
        return revenue - penalty_under - penalty_over

    def evaluate(self, t, c_t, q_u, q_o, P_range):
        """
        Compute expected and worst-case profit for hour t.
        Expected profit: over price uncertainty.
        Worst-case profit: min(profit at P_min, profit at P_max).
        """
        b_t, p_t = self.controls[t]
        P_min, P_max = P_range

        # Simulate expected profit over stochastic prices (simple Monte Carlo)
        # sim_prices = np.random.normal(c_t, 0.05 * c_t, 100)
        expected_profit = self.profit(b_t, p_t, c_t, np.mean(P_range), q_u, q_o)

        # Compute profits at both uncertainty extremes
        profit_low = self.profit(b_t, p_t, c_t, P_min, q_u, q_o)
        profit_high = self.profit(b_t, p_t, c_t, P_max, q_u, q_o)
        worst_profit = min(profit_low, profit_high)

        self.results[t] = {
            "expected_profit": expected_profit,
            "worst_profit": worst_profit,
        }
        return expected_profit, worst_profit


# ============================================================
# 3. SYSTEM REGULATOR (Agent S)
# ============================================================

class SystemRegulator:
    """
    Regulator controlling penalty parameters (q_u, q_o) and evaluating system-level outcomes.
    """

    def __init__(self, env: MarketEnvironment, controls):
        self.env = env
        self.controls = controls  # dict of { "q_u": val, "q_o": val }
        self.results = {}

    def compute_imbalance(self, D_t, conv_bids, c_t, b_t, p_t):
        """Compute system imbalance after clearing."""
        accepted_conv = sum(b_i for b_i, p_i in conv_bids if p_i <= c_t)
        accepted_ren = b_t if p_t <= c_t else 0.0
        return D_t - (accepted_conv + accepted_ren)

    def compute_objectives(self, t, D_t, c_t, conv_bids, b_t, p_t):
        """Compute system-level metrics: imbalance, cost, and renewable share."""
        imbalance = abs(self.compute_imbalance(D_t, conv_bids, c_t, b_t, p_t))
        cost = c_t * D_t
        renewable_accepted = b_t if p_t <= c_t else 0.0
        self.results[t] = {
            "imbalance": imbalance,
            "cost": cost,
            "renewable_share": renewable_accepted,
        }
        return imbalance, cost, renewable_accepted


# ============================================================
# 4. UNIFIED SIMULATION FUNCTION (Interface)
# ============================================================

def simulate_market(controls, uncertainties):
    """
    Interface function to simulate the unified model.

    Parameters
    ----------
    controls : dict
        {
          "renewable": {t: (b_t, p_t)},   # Renewable producer bids
          "regulator": {"q_u": ..., "q_o": ...}  # Regulator penalties
        }

    uncertainties : dict
        {
          "P_range": {t: (P_min, P_max)},   # Renewable generation uncertainty ranges
          "D_realizations": list or None,   # Optional pre-specified demand draws
        }

    Returns
    -------
    results : dict
        {
          "Renewable": {"expected_profit": ..., "worst_profit": ...},
          "Regulator": {"avg_imbalance": ..., "avg_cost": ..., "avg_renewable_share": ...}
        }
    """
    # --- Initialize Environment ---
    params = {
        "T": 24,
        "N_c": 3,
        "mu_D": 1000,
        "sigma_D": 100,
        "mu_b": [400, 350, 300],
        "sigma_b": [0, 0, 0],
        "mu_p": [45, 50, 60],
        "sigma_p": [5, 5, 5],
    }
    env = MarketEnvironment(params, uncertainties)

    # --- Initialize Agents ---
    RP = RenewableProducer(env, controls["renewable"])
    SR = SystemRegulator(env, controls["regulator"])
    q_u, q_o = SR.controls["q_u"], SR.controls["q_o"]
    clearing_price = []

    # --- Simulation loop ---
    results_R, results_S = [], []
    for t in range(1, params["T"] + 1):
        D_t = (uncertainties.get("D_realizations", [None])[t - 1]
               if uncertainties.get("D_realizations") else env.sample_demand(t))
        conv_bids = env.sample_conventional_bids()
        b_t, p_t = RP.controls[t]

        # Market clearing
        c_t = env.market_clearing_price(D_t, conv_bids, b_t, p_t)
        env.update_state(t, D_t, conv_bids, c_t, q_u, q_o)

        # Renewable evaluation
        P_range = uncertainties["P_range"][t]
        exp_profit, worst_profit = RP.evaluate(t, c_t, q_u, q_o, P_range)
        results_R.append((exp_profit, worst_profit))

        # Regulator evaluation
        imbalance, cost, renew_share = SR.compute_objectives(t, D_t, c_t, conv_bids, b_t, p_t)
        results_S.append((imbalance, cost, renew_share))
        clearing_price.append(c_t)

    # --- Aggregate results ---
    RP_out = {
        "expected_profit": [r[0] for r in results_R],
        "worst_profit": [r[1] for r in results_R],
    }
    SR_out = {
        "avg_imbalance": np.mean([r[0] for r in results_S]),
        "avg_cost": np.mean([r[1] for r in results_S]),
        "avg_renewable_share": [r[2] for r in results_S],
    }

    return {"clearing_price": clearing_price, "Renewable": RP_out, "Regulator": SR_out}


controls = {
    "renewable": {t: (220, 50) for t in range(1, 25)},
    "regulator": {"q_u": 100, "q_o": 50},
}

t_lis = [np.random.uniform(200, 200) for i in range(24)]
uncertainties = {
    "P_range": {t: (t_lis[t-1], t_lis[t-1]) for t in range(1, 25)},
}


clearing_price = []
renewable_benefit = []
renewable_accept = []
for i in range(40):
    results = simulate_market(controls, uncertainties)
    clearing_price.append(results["clearing_price"])
    renewable_benefit.append(results["Renewable"]["expected_profit"])
    renewable_accept.append(results["Regulator"]["avg_renewable_share"])


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

# print(renewable_benefit)
# envelope_diagram(clearing_price, 'Current clearing price', 'Average Time Series of Current Clearing Price')
envelope_diagram(renewable_benefit, 'Current renewable profit', 'Average Time Series of Current Renewable Profit')
# envelope_diagram(renewable_accept, 'Current accepted renewable energy', 'Average Time Series of Current Accepted Renewable Energy')