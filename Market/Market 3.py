import numpy as np

# ============================================================
# 1. COMMON ENVIRONMENT
# ============================================================

class MarketEnvironment:
    """
    Shared market environment containing global parameters,
    stochastic demand, and deeply uncertain renewable generation.
    """

    def __init__(self, T=24, mu_D=1000, sigma_D=50,
                 conventional_params=None, uncertainty_sets=None,
                 value_function=None):
        self.T = T
        self.mu_D = mu_D
        self.sigma_D = sigma_D
        self.value_function = value_function or (lambda D: 1.2 * D)
        self.uncertainty_sets = uncertainty_sets or [(200, 800)] * T  # bounds for P_4,t

        # parameters for conventional producers
        self.conventional_params = conventional_params or [
            {"mu_b": 300, "sigma_b": 20, "mu_p": 45, "sigma_p": 5},
            {"mu_b": 400, "sigma_b": 30, "mu_p": 50, "sigma_p": 5},
            {"mu_b": 350, "sigma_b": 25, "mu_p": 55, "sigma_p": 5},
        ]

    def sample_demand(self):
        """Sample stochastic demand for each hour."""
        return np.random.normal(self.mu_D, self.sigma_D, self.T)

    def sample_conventional_bids(self):
        """Sample bids (quantity and price) for conventional producers."""
        bids = []
        for i, params in enumerate(self.conventional_params):
            b_i = np.random.normal(params["mu_b"], params["sigma_b"], self.T)
            p_i = np.random.normal(params["mu_p"], params["sigma_p"], self.T)
            bids.append((b_i, p_i))
        return bids


# ============================================================
# 2. PERSPECTIVE MODULES
# ============================================================

# ------------------------------------------------------------
# (a) Renewable Producer
# ------------------------------------------------------------

class RenewableProducer:
    """
    Renewable producer's perspective with robust optimization
    under deep uncertainty of generation.
    """

    def __init__(self, q_u, q_o):
        self.q_u = q_u  # underproduction penalty
        self.q_o = q_o  # overproduction penalty

    def profit(self, b_t, p_t, c_t, P_t):
        """Compute hourly profit for a given realization."""
        if p_t > c_t:  # bid not accepted
            return 0.0
        penalty = self.q_u * max(b_t - P_t, 0) + self.q_o * max(P_t - b_t, 0)
        return c_t * b_t - penalty

    def expected_worst_case_profit(self, bids, prices, c_t_series, uncertainty_sets):
        """
        Compute total robust expected profit across all hours.
        Assumes worst-case renewable generation within each uncertainty set.
        """
        total_profit = 0.0
        for t, (b_t, p_t, c_t) in enumerate(zip(bids, prices, c_t_series)):
            # worst-case P_t minimizes profit
            P_min, P_max = uncertainty_sets[t]
            profit_min = self.profit(b_t, p_t, c_t, P_min)
            profit_max = self.profit(b_t, p_t, c_t, P_max)
            total_profit += min(profit_min, profit_max)
        return total_profit


# ------------------------------------------------------------
# (b) Conventional Producers
# ------------------------------------------------------------

class ConventionalProducers:
    """
    Represent conventional producers as stochastic agents.
    Their bids are random draws and not optimized.
    """

    def __init__(self, marginal_costs=None):
        self.marginal_costs = marginal_costs or [35, 40, 45]

    def expected_profit(self, bids, prices, clearing_prices):
        """
        Compute expected profits for the three conventional producers.
        """
        profits = []
        for i, (b_i, p_i) in enumerate(zip(bids, prices)):
            c_i = clearing_prices
            accepted = (p_i <= c_i)
            profit_i = np.mean((c_i - self.marginal_costs[i]) * b_i * accepted)
            profits.append(profit_i)
        return profits


# ------------------------------------------------------------
# (c) Market Operator
# ------------------------------------------------------------

class MarketOperator:
    """
    Determines market-clearing price and accepted bids.
    """

    def __init__(self, value_function=None):
        self.value_function = value_function or (lambda D: 1.2 * D)

    def clear_market(self, D_t, all_bids, all_prices):
        """
        Compute market clearing price c_t given all bids.
        """
        # flatten bids and sort by price
        quantities = np.concatenate(all_bids)
        prices = np.concatenate(all_prices)
        order = np.argsort(prices)
        sorted_prices = prices[order]
        sorted_quantities = quantities[order].cumsum()

        # find clearing price where cumulative supply >= demand
        idx = np.searchsorted(sorted_quantities, D_t)
        if idx >= len(sorted_prices):
            c_t = sorted_prices[-1]
        else:
            c_t = sorted_prices[idx]
        return c_t

    def social_welfare(self, D_series, clearing_prices, all_bids, all_prices):
        """
        Calculate total social welfare across all hours.
        """
        total_welfare = 0.0
        for t, D_t in enumerate(D_series):
            V_D = self.value_function(D_t)
            accepted_supply = sum(
                b[t] for b, p in zip(all_bids, all_prices) if p[t] <= clearing_prices[t]
            )
            total_welfare += V_D - clearing_prices[t] * accepted_supply
        return total_welfare

    def imbalance_cost(self, D_series, clearing_prices, all_bids, all_prices):
        """
        Compute total imbalance cost (absolute supply-demand mismatch).
        """
        imbalance = 0.0
        for t, D_t in enumerate(D_series):
            supply = sum(
                b[t] for b, p in zip(all_bids, all_prices) if p[t] <= clearing_prices[t]
            )
            imbalance += abs(supply - D_t)
        return imbalance


# ------------------------------------------------------------
# (d) Regulator
# ------------------------------------------------------------

class Regulator:
    """
    Defines system-wide objectives: efficiency, stability, fairness.
    """

    def __init__(self, q_u, q_o):
        self.q_u = q_u
        self.q_o = q_o

    def evaluate_objectives(self, welfare, imbalance, renewable_profits):
        """
        Compute multi-objective metrics:
        - Efficiency (maximize welfare)
        - Reliability (minimize imbalance)
        - Fairness (minimize variance of renewable profit)
        """
        J1 = -welfare  # negative welfare (for minimization)
        J2 = imbalance
        J3 = np.var(renewable_profits)
        return {"efficiency": J1, "reliability": J2, "fairness": J3}


# ============================================================
# 3. UNIFIED MODEL INTERFACE
# ============================================================

def simulate_market(b4_series, p4_series, q_u, q_o,
                    mu_D, sigma_D, conventional_params,
                    uncertainty_sets):
    """
    Unified interface function:
    Simulates the market given control variables and uncertain parameters,
    and returns the objective function values of each perspective.
    """

    # --- Shared Environment
    env = MarketEnvironment(mu_D=mu_D, sigma_D=sigma_D,
                            conventional_params=conventional_params,
                            uncertainty_sets=uncertainty_sets)

    # --- Sample stochastic variables
    D_series = env.sample_demand()
    conv_bids = env.sample_conventional_bids()

    # --- Instantiate modules
    renewable = RenewableProducer(q_u=q_u, q_o=q_o)
    conventional = ConventionalProducers()
    operator = MarketOperator(value_function=env.value_function)
    regulator = Regulator(q_u=q_u, q_o=q_o)

    # --- Combine all bids for market clearing
    all_bids = [b for b, _ in conv_bids] + [np.array(b4_series)]
    all_prices = [p for _, p in conv_bids] + [np.array(p4_series)]

    # --- Market clearing for each hour
    c_series = []
    for t in range(env.T):
        c_t = operator.clear_market(D_series[t],
                                    [b[:, None][t] if b.ndim == 2 else b for b in all_bids],
                                    [p[:, None][t] if p.ndim == 2 else p for p in all_prices])
        c_series.append(c_t)
    c_series = np.array(c_series)

    # --- Compute each perspective’s outcomes
    # Renewable producer
    renewable_profit = renewable.expected_worst_case_profit(
        bids=b4_series,
        prices=p4_series,
        c_t_series=c_series,
        uncertainty_sets=uncertainty_sets,
    )

    # Conventional producers
    conv_profits = conventional.expected_profit(
        bids=[b for b, _ in conv_bids],
        prices=[p for _, p in conv_bids],
        clearing_prices=c_series,
    )

    # Market operator
    welfare = operator.social_welfare(D_series, c_series, all_bids, all_prices)
    imbalance = operator.imbalance_cost(D_series, c_series, all_bids, all_prices)

    # Regulator
    regulatory_objectives = regulator.evaluate_objectives(
        welfare=welfare,
        imbalance=imbalance,
        renewable_profits=[renewable_profit],
    )

    # --- Return all objectives
    results = {
        "renewable_profit": renewable_profit,
        "conventional_profits": conv_profits,
        "social_welfare": welfare,
        "imbalance": imbalance,
        "regulator_objectives": regulatory_objectives,
    }
    return results



# ============================================================
# EXAMPLE CALL TO THE UNIFIED MARKET MODEL
# ============================================================

if __name__ == "__main__":

    # --- Define control variables and parameters
    T = 24
    b4_series = np.full(T, 20)      # Renewable producer bids 500 MWh each hour
    p4_series = np.full(T, 35)      # Offer price = 48 $/MWh

    q_u = 80                        # Underproduction penalty ($/MWh)
    q_o = 20                        # Overproduction (curtailment) penalty ($/MWh)

    mu_D = 100                        # Mean demand (MWh)
    sigma_D = 10                       # Demand variability (MWh)

    # Conventional producers: mean and variance for (quantity, price)
    conventional_params = [
        {"mu_b": 30, "sigma_b": 5, "mu_p": 40, "sigma_p": 5},
        {"mu_b": 40, "sigma_b": 5, "mu_p": 45, "sigma_p": 5},
        {"mu_b": 50, "sigma_b": 5, "mu_p": 50, "sigma_p": 5},
    ]

    # Deep uncertainty: renewable generation bounds for each hour
    # (can vary hourly, but we assume constant here)
    uncertainty_sets = [(200, 800)] * T  # bounds on P_4,t (MWh)

    # --- Run the unified simulation
    results = simulate_market(
        b4_series=b4_series,
        p4_series=p4_series,
        q_u=q_u,
        q_o=q_o,
        mu_D=mu_D,
        sigma_D=sigma_D,
        conventional_params=conventional_params,
        uncertainty_sets=uncertainty_sets,
    )

    # --- Extract renewable profit for each interval
    renewable_profit_total = results["renewable_profit"]

    print("Renewable Producer’s Total Robust Expected Profit:", renewable_profit_total)

    # If desired, compute profit per hour (for analysis or plotting)
    renewable = RenewableProducer(q_u=q_u, q_o=q_o)

    # Suppose we also have the clearing prices returned by simulate_market
    # (in a real execution, we could extract them from c_series)
    # Here we show how to compute per-hour profit conceptually:
    clearing_prices = np.linspace(40, 60, T)  # placeholder range

    hourly_profits = [
        renewable.profit(
            b_t=b4_series[t],
            p_t=p4_series[t],
            c_t=clearing_prices[t],
            P_t=np.random.uniform(*uncertainty_sets[t])  # one realization within uncertainty bounds
        )
        for t in range(T)
    ]

    print("Hourly renewable profits (sample realization):")
    for t, pi_t in enumerate(hourly_profits, start=1):
        print(f"Hour {t:02d}: Profit = {pi_t:.2f}")
