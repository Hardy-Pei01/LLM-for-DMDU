import random

import numpy as np

# ==============================================================
# 1. Shared Market Environment
# ==============================================================

class MarketEnvironment:
    """
    Shared stochastic environment representing the day-ahead market.
    Demand is modeled as a normal distribution with fixed mean and variance.
    Renewable generation is provided as an uncertain external function f_P().
    """

    def __init__(self, T=24, q_u=20, q_o=10, f_P=None):
        self.T = T
        self.q_u = q_u
        self.q_o = q_o

        # Fixed demand parameters (deterministic across all simulations)
        self.mu_D = 100
        self.sigma_D = 10

        # Deeply uncertain renewable generation function
        # This must be provided by the user as part of simulation input
        if f_P is None:
            raise ValueError("A renewable generation function f_P must be provided.")
        self.f_P = f_P

        # Conventional producer parameters (fixed, stochastic but with constant mean/variance)
        self.N_c = 3
        self.mu_b = [30, 40, 50]
        self.sigma_b = [5, 5, 5]
        self.mu_p = [40, 45, 50]
        self.sigma_p = [5, 5, 5]

    def sample_demand(self):
        """Hourly total demand (Normal with constant mean and variance)."""
        return np.random.normal(self.mu_D, self.sigma_D)

    def sample_conventional_bids(self):
        """Sample bids (quantity, price) for each of the 3 conventional producers."""
        bids = []
        for i in range(self.N_c):
            b_i = np.random.normal(self.mu_b[i], self.sigma_b[i])
            p_i = np.random.normal(self.mu_p[i], self.sigma_p[i])
            bids.append((b_i, p_i))
        return bids

    def market_clearing_price(self, D_t, bids_conv, bid_renew):
        """
        Compute market-clearing price c_t.
        Simplified rule: sort offers by price until cumulative quantity ≥ demand.
        """
        offers = []
        for b, p in bids_conv:
            offers.append((p, b))
        offers.append((bid_renew[1], bid_renew[0]))
        offers.sort(key=lambda x: x[0])  # ascending by price

        cum_qty, clearing_price = 0, offers[-1][0]
        for p, b in offers:
            cum_qty += b
            if cum_qty >= D_t:
                clearing_price = p
                break
        return clearing_price


# ==============================================================
# 2. Perspective: Renewable Producer
# ==============================================================

class RenewableProducer:
    """
    Strategic renewable energy producer.
    Chooses bid quantity and price to maximize expected profit.
    """

    def __init__(self, env: MarketEnvironment):
        self.env = env
        self.profit = []

    def profit_function(self, b_t, p_t, c_t, P_t):
        q_u, q_o = self.env.q_u, self.env.q_o
        if p_t <= c_t:
            penalty_under = q_u * max(0, b_t - P_t)
            penalty_over = q_o * max(0, P_t - b_t)
            return c_t * b_t - penalty_under - penalty_over
        else:
            return 0

    def step(self, b_t, p_t, c_t, P_t):
        """Compute hourly profit and update total."""
        pi_t = self.profit_function(b_t, p_t, c_t, P_t)
        self.profit.append(pi_t)
        return pi_t


# ==============================================================
# 3. Perspective: Conventional Producers
# ==============================================================

class ConventionalProducer:
    """
    Non-strategic conventional producer with stochastic bids.
    """

    def __init__(self, env: MarketEnvironment, i: int, alpha=5, beta=0.1):
        self.env = env
        self.i = i
        self.alpha = alpha
        self.beta = beta
        self.profit = 0

    def cost(self, b_i):
        return self.alpha * b_i + self.beta * b_i ** 2

    def profit_function(self, b_i, p_i, c_t):
        if p_i <= c_t:
            return c_t * b_i - self.cost(b_i)
        else:
            return 0

    def step(self, b_i, p_i, c_t):
        pi_t = self.profit_function(b_i, p_i, c_t)
        self.profit += pi_t
        return pi_t


# ==============================================================
# 4. Perspective: Market Operator
# ==============================================================

class MarketOperator:
    """
    Market-clearing mechanism and welfare evaluation.
    """

    def __init__(self, env: MarketEnvironment):
        self.env = env
        self.welfare = 0
        self.imbalance = 0

    def compute_welfare(self, D_t, c_t, producers):
        total_supply = sum([p[0] for p in producers])
        self.imbalance += abs(D_t - total_supply)
        # Simplified welfare metric
        return D_t * c_t - sum([0.5 * c_t * p[0] for p in producers])

    def step(self, D_t, c_t, producers):
        W_t = self.compute_welfare(D_t, c_t, producers)
        self.welfare += W_t
        return W_t


# ==============================================================
# 5. Perspective: Regulator / System Planner
# ==============================================================

class Regulator:
    """
    Regulator with policy levers: imbalance penalties (q_u, q_o) and renewable incentives (tau_t).
    """

    def __init__(self, env: MarketEnvironment, tau_t=0):
        self.env = env
        self.tau_t = tau_t
        self.metrics = {"reliability": 0, "price_stability": [], "renewable_share": []}

    def evaluate(self, D_t, c_t, b_R, accepted_R):
        rho_t = (b_R / D_t) if accepted_R else 0
        self.metrics["renewable_share"].append(rho_t)
        self.metrics["price_stability"].append(c_t)

    def finalize_metrics(self, total_imbalance):
        self.metrics["reliability"] = -total_imbalance
        self.metrics["price_stability"] = -np.var(self.metrics["price_stability"])
        self.metrics["renewable_share"] = np.mean(self.metrics["renewable_share"])
        return self.metrics


# ==============================================================
# 6. Unified Market Simulation Interface
# ==============================================================

def simulate_market(
    controls_R,
    controls_G,
    f_P_instance,
    T=24
):
    """
    Unified simulation interface for the day-ahead electricity market.

    Inputs:
        controls_R : dict
            Renewable producer bids {t: (b_t, p_t)}.
        controls_G : dict
            Regulator settings {'q_u': val, 'q_o': val, 'tau_t': val}.
        f_P_instance : callable
            A function representing a possible instance of renewable generation.
            Must return a random sample of P_t (MWh) each time it's called.
        T : int
            Number of hourly intervals (default 24).

    Output:
        dict containing objective function values for each perspective.
    """

    # --- Initialize environment and agents ---
    env = MarketEnvironment(
        T=T,
        q_u=controls_G.get("q_u", 20),
        q_o=controls_G.get("q_o", 10),
        f_P=f_P_instance
    )

    renewable = RenewableProducer(env)
    conventionals = [ConventionalProducer(env, i) for i in range(3)]
    operator = MarketOperator(env)
    regulator = Regulator(env, tau_t=controls_G.get("tau_t", 0))

    # --- Simulation loop ---
    for t in range(T):
        D_t = env.sample_demand()  # deterministic distribution with fixed parameters
        P_t = env.f_P()            # deeply uncertain renewable generation realization
        bids_conv = env.sample_conventional_bids()
        b_R, p_R = controls_R.get(t, (0, 0))

        # Market clearing
        c_t = env.market_clearing_price(D_t, bids_conv, (b_R, p_R))

        # Compute profits
        renewable.step(b_R, p_R, c_t, P_t)
        for i, conv in enumerate(conventionals):
            conv.step(*bids_conv[i], c_t)

        # Operator metrics
        producers = bids_conv + [(b_R, p_R)]
        operator.step(D_t, c_t, producers)

        # Regulator metrics
        accepted_R = (p_R <= c_t)
        regulator.evaluate(D_t, c_t, b_R, accepted_R)

    # --- Compute final objectives ---
    regulator_metrics = regulator.finalize_metrics(operator.imbalance)
    results = {
        "Renewable_Profit": renewable.profit,
        "Conventional_Profit": sum([conv.profit for conv in conventionals]),
        "Market_Welfare": operator.welfare,
        "Regulator_Objectives": regulator_metrics
    }
    return results


# Example of deeply uncertain renewable generation function (e.g., wind)
def uncertain_generation():
    return random.uniform(10,30)

controls_R = {t: (20, 35) for t in range(24)}  # renewable bid quantity & price
controls_G = {"q_u": 80, "q_o": 20, "tau_t": 2}

# Simulate one scenario of renewable generation
results = simulate_market(controls_R, controls_G, f_P_instance=uncertain_generation)

# results contains all objective values (no need to execute here)
print(results["Renewable_Profit"])