import numpy as np

# ===============================================================
# 1. Shared Environment
# ===============================================================

class MarketEnvironment:
    """Shared environment including parameters and uncertainties."""
    def __init__(self, params, uncertainties):
        self.params = params
        self.uncertainties = uncertainties
        self.num_hours = 24

    def get_demand(self, t):
        mu, sigma = self.params["mu_D"], self.params["sigma_D"]
        return (self.uncertainties.get("D_t", [None]*24)[t]
                or np.random.normal(mu, sigma))

    def get_conventional_bids(self, t):
        bids = []
        for j in range(3):
            mu_b, sig_b = self.params["mu_b"][j], self.params["sigma_b"][j]
            mu_p, sig_p = self.params["mu_p"][j], self.params["sigma_p"][j]
            b = (self.uncertainties.get("b_t^j", [[None]*3]*24)[t][j]
                 if "b_t^j" in self.uncertainties else np.random.normal(mu_b, sig_b))
            p = (self.uncertainties.get("p_t^j", [[None]*3]*24)[t][j]
                 if "p_t^j" in self.uncertainties else np.random.normal(mu_p, sig_p))
            bids.append((b, p))
        return bids

    def get_generation(self, t):
        g_min, g_max = self.params["g_bounds"]
        return (self.uncertainties.get("g_t", [None]*24)[t]
                or np.random.uniform(g_min, g_max))


# ===============================================================
# 2. System Operator (Market Clearing)
# ===============================================================

class SystemOperator:
    def clear_market(self, bids, demand):
        """Merit-order clearing."""
        bids_sorted = sorted(enumerate(bids), key=lambda x: x[1][1])
        supply = 0.0
        clearing_price = None
        accepted = [0.0] * len(bids)

        for idx, (b, p) in [(i, bid) for i, bid in bids_sorted]:
            if supply + b < demand:
                supply += b
                accepted[idx] = b
            else:
                accepted[idx] = max(0, demand - supply)
                clearing_price = p
                break

        if clearing_price is None:  # all bids accepted
            clearing_price = max(p for (_, p) in bids)

        return clearing_price, accepted


# ===============================================================
# 3. Renewable Producer
# ===============================================================

class RenewableProducer:
    def __init__(self, env):
        self.env = env
        self.controls = {"b_r": np.zeros(24), "p_r": np.zeros(24)}

    def set_controls(self, controls):
        self.controls["b_r"] = np.array(controls.get("b_r", np.zeros(24)))
        self.controls["p_r"] = np.array(controls.get("p_r", np.zeros(24)))

    def hourly_profit(self, t, price, accepted, generation):
        """Profit in hour t only."""
        c_minus, c_plus = self.env.params["c_minus"], self.env.params["c_plus"]
        tau = self.env.params.get("tau", np.zeros(24))[t]
        b_t, p_t = self.controls["b_r"][t], self.controls["p_r"][t]
        a_t = 1.0 if accepted > 0 else 0.0
        g_t = generation
        return a_t * ((p_t + tau) * b_t
                      - c_minus * max(b_t - g_t, 0)
                      - c_plus * max(g_t - b_t, 0))

    def hourly_imbalance(self, t, generation):
        b_t = self.controls["b_r"][t]
        return abs(b_t - generation)


# ===============================================================
# 4. Conventional Producers
# ===============================================================

class ConventionalProducers:
    def __init__(self, env):
        self.env = env

    def expected_profit_metrics(self, clearing_price, accepted):
        results = []
        for j in range(3):
            mu_p = self.env.params["mu_p"][j]
            mu_b = self.env.params["mu_b"][j]
            Cj = self.env.params.get("cost_j", [0, 0, 0])[j]
            profit = (mu_p - Cj) * mu_b * (1 if accepted[j+1] > 0 else 0)
            results.append(profit)
        return results


# ===============================================================
# 5. Regulator / Policy Maker
# ===============================================================

class Regulator:
    def __init__(self, env):
        self.env = env
        self.controls = {"c_minus": env.params["c_minus"],
                         "c_plus": env.params["c_plus"],
                         "tau": env.params.get("tau", np.zeros(24)),
                         "R_target": env.params.get("R_target", 0.0)}

    def set_controls(self, controls):
        for k, v in controls.items():
            self.controls[k] = v
        self.env.params.update(controls)

    def welfare(self, demand, price, quantities):
        U = self.env.params["U_func"]
        total_payment = np.sum([price * q for q in quantities])
        return U(demand) - total_payment

    def emissions(self, quantities):
        alphas = self.env.params["alpha"]
        return np.sum([alphas[j] * quantities[j+1] for j in range(3)])  # skip renewable

    def reliability(self, demand, quantities):
        total_supply = np.sum(quantities)
        return max(demand - total_supply, 0)


# ===============================================================
# 6. Unified Market Simulation Interface
# ===============================================================

def simulate_market(control_dict, uncertainty_dict):
    """
    Simulate the unified market dynamics for 24 hours.

    Inputs:
        control_dict: dict of controls per perspective
        uncertainty_dict: dict of uncertain variables or samples

    Returns:
        dict of objective values for each perspective
    """
    params = {
        "mu_D": 100, "sigma_D": 10,
        "mu_b": [30, 40, 50],
        "sigma_b": [5, 5, 5],
        "mu_p": [40, 45, 50],
        "sigma_p": [5, 5, 5],
        "c_minus": 20, "c_plus": 10,
        "g_bounds": (0, 800),
        "alpha": [0.5, 0.6, 0.8],
        "U_func": lambda d: 100 * np.sqrt(d)
    }

    env = MarketEnvironment(params, uncertainty_dict)
    renewable = RenewableProducer(env)
    renewable.set_controls(control_dict.get("renewable", {}))

    regulator = Regulator(env)
    regulator.set_controls(control_dict.get("regulator", {}))

    operator = SystemOperator()
    conventional = ConventionalProducers(env)

    # Initialize accumulators
    total_profit_r = []
    total_imbalance_r = []
    total_welfare = 0.0
    total_emissions = 0.0
    total_reliability = 0.0

    # Hourly simulation
    for t in range(24):
        D_t = env.get_demand(t)
        g_t = env.get_generation(t)
        conv_bids = env.get_conventional_bids(t)
        bid_r = (renewable.controls["b_r"][t], renewable.controls["p_r"][t])
        bids_all = [bid_r] + conv_bids

        pi_t, q_t_all = operator.clear_market(bids_all, D_t)

        # --- Renewable outcomes ---
        profit_t = renewable.hourly_profit(t, pi_t, q_t_all[0], g_t)
        imbalance_t = renewable.hourly_imbalance(t, g_t)
        total_profit_r.append(profit_t)
        total_imbalance_r.append(imbalance_t)

        # --- Regulator outcomes ---
        total_welfare += regulator.welfare(D_t, pi_t, q_t_all)
        total_emissions += regulator.emissions(q_t_all)
        total_reliability += regulator.reliability(D_t, q_t_all)

    results = {
        "RenewableProducer": {
            "ExpectedProfit": total_profit_r,
            "ImbalanceExposure": total_imbalance_r
        },
        "ConventionalProducers": {
            "ExpectedProfits": conventional.expected_profit_metrics(pi_t, q_t_all)
        },
        "Regulator": {
            "SocialWelfare": total_welfare,
            "Emissions": total_emissions,
            "ReliabilityPenalty": total_reliability
        }
    }

    return results


control_inputs = {
    "renewable": {"b_r": np.full(24, 20), "p_r": np.full(24, 35)},
    "regulator": {"c_minus": 20, "c_plus": 10, "tau": np.full(24, 2)}
}

uncertainty_inputs = {"g_t": [max(0, np.random.lognormal(mean=3.0, sigma=0.4) / 100) for i in range(24)]}

outputs = simulate_market(control_inputs, uncertainty_inputs)
print(outputs["RenewableProducer"]["ExpectedProfit"])
