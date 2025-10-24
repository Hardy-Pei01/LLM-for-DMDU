import numpy as np

# ============================================================
# Shared Environment Definition
# ============================================================

class MarketEnvironment:
    """
    Shared environment for the day-ahead electricity market.
    Includes fixed constants, random variables, and market clearing.
    """

    def __init__(self, uncertain_params):
        """
        uncertain_params: dictionary of uncertain parameters:
            {
                'G_bounds': [(float, float)]*24,
                'price_cap': float
            }
        """
        # Fixed constants (market baseline)
        self.T = 24
        self.mu_D, self.sigma_D = 1000.0, 100      # mean & std of demand
        self.mu_b = [400, 350, 300]            # mean bid quantities
        self.sigma_b = [50, 40, 30]
        self.mu_p = [45, 50, 60]               # mean offer prices
        self.sigma_p = [5, 5, 5]

        # Uncertain parameters
        self.params = uncertain_params
        self.G_bounds = uncertain_params["G_bounds"]
        self.price_cap = uncertain_params["price_cap"]

    def sample_random_variables(self):
        """Sample random demand and conventional producer bids."""
        D = np.random.normal(self.mu_D, self.sigma_D, self.T)
        b_i = [np.random.normal(self.mu_b[i], self.sigma_b[i], self.T) for i in range(3)]
        p_i = [np.random.normal(self.mu_p[i], self.sigma_p[i], self.T) for i in range(3)]
        return D, b_i, p_i

    def market_clearing(self, D_t, bids_conventional, bids_renewable):
        """
        Simulate hourly market clearing.
        - D_t: realized demand
        - bids_conventional: list of (b_i, p_i)
        - bids_renewable: (b_r, p_r)
        Returns: (P_c, q_i, q_r)
        """
        offers = []
        for i, (b_i, p_i) in enumerate(bids_conventional):
            offers.append((p_i, b_i, f"conv_{i}"))
        offers.append((bids_renewable[1], bids_renewable[0], "renewable"))

        # Sort offers by ascending price (merit order)
        offers.sort(key=lambda x: x[0])

        remaining_demand = D_t
        q = {}
        P_c = 0.0
        for price, quantity, name in offers:
            accepted = min(quantity, remaining_demand)
            q[name] = accepted
            remaining_demand -= accepted
            if remaining_demand <= 1e-6:
                P_c = price
                break

        # Fill zeros for unaccepted offers
        for _, _, name in offers:
            q.setdefault(name, 0.0)

        q_i = [q.get(f"conv_{i}", 0.0) for i in range(3)]
        q_r = q.get("renewable", 0.0)
        return P_c, q_i, q_r


# ============================================================
# Renewable Producer Perspective
# ============================================================

class RenewableProducer:
    def __init__(self, env: MarketEnvironment, controls):
        """
        controls: {'b': [...], 'p': [...]}
        """
        self.env = env
        self.b = np.array(controls["b"])
        self.p = np.array(controls["p"])

    def profit_per_hour(self, P_c, q_r, G_t, P_u, P_o):
        """
        Hourly profit function with penalty coefficients.
        P_c: clearing price ($/MWh)
        q_r: accepted quantity (MWh)
        G_t: realized renewable generation (MWh)
        P_u: under-delivery penalty coefficient ($/MWh)
        P_o: over-delivery penalty coefficient ($/MWh)
        """
        under_delivery = max(0, q_r - G_t)
        over_delivery = max(0, G_t - q_r)
        return P_c * q_r - P_u * under_delivery - P_o * over_delivery

    def imbalance_per_hour(self, G_t, q_r):
        """Absolute imbalance."""
        return abs(G_t - q_r)

    def evaluate_objectives(self, env_realizations, penalties):
        """
        Compute expected robust profit and imbalance.
        env_realizations: (D, b_i, p_i)
        penalties: {'P_u': [...], 'P_o': [...]}
        """
        D, b_i, p_i = env_realizations
        T = self.env.T
        G_bounds = self.env.G_bounds
        P_u, P_o = np.array(penalties["P_u"]), np.array(penalties["P_o"])

        exp_profit, exp_imbalance = [], 0.0

        for t in range(T):
            bids_conventional = [(b_i[j][t], p_i[j][t]) for j in range(3)]
            bids_renewable = (self.b[t], self.p[t])
            P_c, _, q_r = self.env.market_clearing(D[t], bids_conventional, bids_renewable)

            # Worst-case generation within bounds
            G_low, G_high = G_bounds[t]
            profit_low = self.profit_per_hour(P_c, q_r, G_low, P_u[t], P_o[t])
            profit_high = self.profit_per_hour(P_c, q_r, G_high, P_u[t], P_o[t])
            worst_profit = min(profit_low, profit_high)

            imbalance_low = self.imbalance_per_hour(G_low, q_r)
            imbalance_high = self.imbalance_per_hour(G_high, q_r)
            worst_imbalance = max(imbalance_low, imbalance_high)

            exp_profit.append(worst_profit)
            exp_imbalance += worst_imbalance

        return exp_profit, exp_imbalance


# ============================================================
# System Regulator Perspective
# ============================================================

class SystemRegulator:
    def __init__(self, env: MarketEnvironment, controls):
        """
        controls: {'P_u': [...], 'P_o': [...]}
        """
        self.env = env
        self.P_u = np.array(controls["P_u"])
        self.P_o = np.array(controls["P_o"])

    def system_cost_per_hour(self, P_c, D_t, q_r, G_t, P_u, P_o):
        """
        Total system cost = total market payment + penalty costs.
        Penalties increase total cost when imbalances occur.
        """
        under_delivery = max(0, q_r - G_t)
        over_delivery = max(0, G_t - q_r)
        return P_c * D_t + P_u * under_delivery + P_o * over_delivery

    def evaluate_objectives(self, env_realizations, renewable_controls):
        """
        Compute expected system cost and imbalance.
        env_realizations: (D, b_i, p_i)
        renewable_controls: {'b': [...], 'p': [...]}
        """
        D, b_i, p_i = env_realizations
        T = self.env.T
        b_r, p_r = np.array(renewable_controls["b"]), np.array(renewable_controls["p"])
        G_bounds = self.env.G_bounds

        exp_cost, exp_imbalance = 0.0, 0.0

        for t in range(T):
            bids_conventional = [(b_i[j][t], p_i[j][t]) for j in range(3)]
            bids_renewable = (b_r[t], p_r[t])
            P_c, _, q_r = self.env.market_clearing(D[t], bids_conventional, bids_renewable)

            G_low, G_high = G_bounds[t]
            cost_low = self.system_cost_per_hour(P_c, D[t], q_r, G_low, self.P_u[t], self.P_o[t])
            cost_high = self.system_cost_per_hour(P_c, D[t], q_r, G_high, self.P_u[t], self.P_o[t])
            worst_cost = max(cost_low, cost_high)

            imbalance_low = abs(G_low - q_r)
            imbalance_high = abs(G_high - q_r)
            worst_imbalance = max(imbalance_low, imbalance_high)

            exp_cost += worst_cost
            exp_imbalance += worst_imbalance

        return exp_cost, exp_imbalance


# ============================================================
# Unified Simulation Interface
# ============================================================

def simulate_market(control_vars, uncertain_params):
    """
    Simulate unified market dynamics and return objective values.

    control_vars:
        {
            'renewable': {'b': [...], 'p': [...]},
            'regulator': {'P_u': [...], 'P_o': [...]}
        }

    uncertain_params:
        {
            'G_bounds': [(low, high)] * 24,
            'price_cap': float
        }

    Returns:
        {
            'renewable_profit': float,
            'renewable_imbalance': float,
            'system_cost': float,
            'system_imbalance': float
        }
    """
    # 1. Initialize environment
    env = MarketEnvironment(uncertain_params)

    # 2. Sample random variables (demand and conventional bids)
    D, b_i, p_i = env.sample_random_variables()
    env_realizations = (D, b_i, p_i)

    # 3. Initialize agents
    renewable = RenewableProducer(env, control_vars["renewable"])
    regulator = SystemRegulator(env, control_vars["regulator"])

    # 4. Evaluate objectives
    penalties = control_vars["regulator"]
    profit, imbalance_r = renewable.evaluate_objectives(env_realizations, penalties)
    cost, imbalance_sys = regulator.evaluate_objectives(env_realizations, control_vars["renewable"])

    # 5. Return results
    return {
        "renewable_profit": profit,
        "renewable_imbalance": imbalance_r,
        "system_cost": cost,
        "system_imbalance": imbalance_sys,
    }


# ============================================================
# Example (not executed)
# ============================================================
control_vars = {
    'renewable': {'b': [220]*24, 'p': [50]*24},
    'regulator': {'P_u': [100]*24, 'P_o': [50]*24}
}

t_lis = [np.random.uniform(150, 250) for i in range(24)]
uncertain_params = {
    'G_bounds': [(t_lis[i], t_lis[i]) for i in range(24)],
    'price_cap': 100.0
}
