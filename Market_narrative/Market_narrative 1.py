import numpy as np

# ==============================================================
# 1. Common Environment Class
# ==============================================================

class MarketEnvironment:
    """
    Shared 24-hour day-ahead electricity market environment.
    Defines constants, stochastic structure, and market clearing logic.
    """

    def __init__(self):
        # Fixed constants (global market configuration)
        self.T = 24
        # Demand distribution parameters
        self.mu_D = 1000
        self.sigma_D = 100
        # Conventional producers: mean bid quantities and prices
        self.mu_b = [400, 350, 300]
        self.sigma_b = [50, 40, 30]
        self.mu_p = [45, 50, 60]
        self.sigma_p = [5, 5, 5]
        # Renewable generation uncertainty range
        self.G_lower = [150] * 24
        self.G_upper = [250] * 24
        # Emission factors for conventional producers (tCO₂/MWh)
        self.E_CO2 = [0.5, 0.6, 0.7]

    # --------------------------
    # Sampling methods
    # --------------------------

    def sample_demand(self):
        """Sample hourly demand from Normal(μ, σ)."""
        return np.random.normal(self.mu_D, self.sigma_D, self.T)

    def sample_conventional_bids(self):
        """Sample (quantity, price) for conventional producers."""
        b = np.zeros((3, self.T))
        p = np.zeros((3, self.T))
        for i in range(3):
            b[i, :] = np.random.normal(self.mu_b[i], self.sigma_b[i], self.T)
            p[i, :] = np.random.normal(self.mu_p[i], self.sigma_p[i], self.T)
        return b, p

    def sample_renewable_generation(self):
        """Return midpoints of uncertainty intervals for renewable generation."""
        return np.array([(low + high) / 2 for low, high in zip(self.G_lower, self.G_upper)])

    # --------------------------
    # Market clearing
    # --------------------------

    def market_clearing(self, D_t, bids):
        """
        Merit-order market clearing given demand and offers.
        bids: list of tuples (b_i, p_i) including renewable first.
        Returns clearing price P_clr and accepted quantities q_i.
        """
        offers = sorted(enumerate(bids), key=lambda x: x[1][1])  # sort by price
        q = np.zeros(len(bids))
        remaining = D_t
        P_clr = 0.0
        for idx, (b_i, p_i) in offers:
            accept = min(b_i, remaining)
            q[idx] = accept
            remaining -= accept
            P_clr = p_i
            if remaining <= 0:
                break
        return P_clr, q


# ==============================================================
# 2. Renewable Producer Perspective
# ==============================================================

class RenewableProducer:
    """
    Renewable producer: selects bid quantity and offer price.
    Computes profit given market clearing and realized generation.
    """

    def __init__(self, env, controls):
        self.env = env
        self.b_R = np.array(controls.get("b_R", [100] * env.T))
        self.p_R = np.array(controls.get("p_R", [45] * env.T))
        # Penalty coefficients set by regulator
        self.c_under = controls.get("c_under", 10)
        self.c_over = controls.get("c_over", 5)

    def profit(self, g_R, P_clr, accepted, b_R_t):
        """Compute profit for one hour."""
        if accepted:
            penalty = self.c_under * max(0, b_R_t - g_R) + self.c_over * max(0, g_R - b_R_t)
            return P_clr * b_R_t - penalty
        else:
            return 0.0

    def evaluate(self, demand, conv_bids, g_R):
        """Evaluate expected (simulated) profit over 24 hours."""
        total_profit = 0.0
        profits = []
        for t in range(self.env.T):
            D_t = demand[t]
            # Combine renewable + conventional offers
            bids = [(self.b_R[t], self.p_R[t])] + [
                (conv_bids[0][i, t], conv_bids[1][i, t]) for i in range(3)
            ]
            P_clr, q = self.env.market_clearing(D_t, bids)
            accepted = self.p_R[t] <= P_clr
            profit_t = self.profit(g_R[t], P_clr, accepted, self.b_R[t])
            total_profit += profit_t
            profits.append(profit_t)
        # print(profits)
        return profits


# ==============================================================
# 3. System Regulator Perspective
# ==============================================================

class SystemRegulator:
    """
    Regulator: controls penalty parameters and emission cap.
    Evaluates system-wide cost, emission, and imbalance metrics.
    """

    def __init__(self, env, controls):
        self.env = env
        self.c_under = controls.get("c_under", 10)
        self.c_over = controls.get("c_over", 5)
        self.E_cap = controls.get("E_cap", 300)

    def evaluate(self, demand, conv_bids, ren_bids):
        """
        Compute system cost, emission violation, and imbalance metrics.
        """
        total_cost = 0.0
        total_violation = 0.0
        total_imbalance = 0.0

        for t in range(self.env.T):
            D_t = demand[t]
            bids = [ren_bids[t]] + [
                (conv_bids[0][i, t], conv_bids[1][i, t]) for i in range(3)
            ]
            P_clr, q = self.env.market_clearing(D_t, bids)

            # System cost (accepted offers)
            total_cost += sum([p * q_i for (_, p), q_i in zip(bids, q)])

            # Emissions (only conventional producers)
            E_t = sum([self.env.E_CO2[i] * q[i + 1] for i in range(3)])
            total_violation += max(0, E_t - self.E_cap)

            # Imbalance (supply-demand mismatch)
            total_imbalance += abs(D_t - sum(q))

        return {
            "system_cost": total_cost,
            "emission_violation": total_violation,
            "imbalance": total_imbalance,
        }


# ==============================================================
# 4. Unified Model Interface Function
# ==============================================================

def simulate_market(controls, uncertainties):
    """
    Unified interface to simulate the shared market environment.
    Inputs:
        controls: dict containing
            - "b_R", "p_R" : renewable bids
            - "c_under", "c_over", "E_cap" : regulator controls
        uncertainties: dict containing
            - "D_t" : realized demand array (optional)
            - "b_conv", "p_conv" : realized conventional bids (optional)
            - "g_R" : realized renewable generation array (optional)
    Returns:
        dict with:
            - 'renewable_profit'
            - 'system_cost'
            - 'emission_violation'
            - 'imbalance'
    """
    # Initialize environment (constants embedded)
    env = MarketEnvironment()

    # Sample or use provided uncertain realizations
    D_t = uncertainties.get("D_t", env.sample_demand())
    b_conv = uncertainties.get("b_conv", None)
    p_conv = uncertainties.get("p_conv", None)
    if b_conv is None or p_conv is None:
        b_conv, p_conv = env.sample_conventional_bids()
    g_R = uncertainties.get("g_R", env.sample_renewable_generation())

    # Instantiate both perspectives
    rp = RenewableProducer(env, controls)
    sr = SystemRegulator(env, controls)

    # Evaluate each perspective
    rp_profit = rp.evaluate(D_t, (b_conv, p_conv), g_R)
    sr_results = sr.evaluate(D_t, (b_conv, p_conv), list(zip(rp.b_R, rp.p_R)))

    # Combine results
    return {
        "renewable_profit": rp_profit,
        **sr_results
    }


# ==============================================================
# Example (not executed)
# ==============================================================


controls = {
    "b_R": [220]*24,
    "p_R": [50]*24,
    "c_under": 100,
    "c_over": 50,
    "E_cap": 300
}

uncertainties = {
    # "g_R": [150]*24
}