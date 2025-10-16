import numpy as np

# ============================================================
# 1. COMMON ENVIRONMENT
# ============================================================

class MarketEnvironment:
    """
    Shared environment for all perspectives.
    Holds global parameters, stochastic generation functions, and market-clearing logic.
    """

    def __init__(self, T=24, mu_D=100, sigma_D=10, q_u=50, q_o=30,
                 mu_b=None, sigma_b=None, mu_p=None, sigma_p=None, alpha=None):
        self.T = T
        self.mu_D = mu_D
        self.sigma_D = sigma_D
        self.q_u = q_u
        self.q_o = q_o
        self.mu_b = mu_b if mu_b is not None else [50, 60, 55]
        self.sigma_b = sigma_b if sigma_b is not None else [5, 6, 5]
        self.mu_p = mu_p if mu_p is not None else [70, 65, 68]
        self.sigma_p = sigma_p if sigma_p is not None else [5, 4, 6]
        self.alpha = alpha if alpha is not None else [0.6, 0.55, 0.58]

    def sample_demand(self):
        return np.random.normal(self.mu_D, self.sigma_D, self.T)

    def sample_conventional_bids(self):
        """Generate stochastic bids for the 3 conventional producers."""
        b = [np.random.normal(self.mu_b[i], self.sigma_b[i], self.T) for i in range(3)]
        p = [np.random.normal(self.mu_p[i], self.sigma_p[i], self.T) for i in range(3)]
        return np.array(b), np.array(p)

    def clearing_price(self, D_t, b_conv, p_conv, b_R, p_R):
        """
        Determine market clearing price c_t such that accepted bids meet demand.
        Approximate discrete clearing mechanism by sorting offers.
        """
        offers = []
        # Add renewable producer's offer
        offers.append((p_R, b_R, "R"))
        # Add conventional producers' offers
        for i in range(3):
            offers.append((p_conv[i], b_conv[i], f"C{i+1}"))

        # Sort all offers by price
        offers.sort(key=lambda x: x[0])
        total_supply = 0
        for offer_price, offer_qty, _ in offers:
            total_supply += offer_qty
            if total_supply >= D_t:
                return offer_price
        return offers[-1][0]  # if supply < demand, highest price wins


# ============================================================
# 2. PERSPECTIVES
# ============================================================

# -------------------------------
# Renewable Producer Perspective
# -------------------------------
class RenewableProducer:
    def __init__(self, env: MarketEnvironment):
        self.env = env

    def profit(self, c_t, b_t, p_t, P_t):
        """Compute realized profit for the renewable producer."""
        if p_t > c_t:  # bid rejected
            return 0.0
        q_u, q_o = self.env.q_u, self.env.q_o
        under = max(0, b_t - P_t)
        over = max(0, P_t - b_t)
        return c_t * b_t - q_u * under - q_o * over

    def evaluate_objectives(self, b_R, p_R, c, P_R):
        profits = np.array([self.profit(c[t], b_R[t], p_R[t], P_R[t]) for t in range(self.env.T)])
        J1 = np.mean(profits.sum())  # expected profit (approx.)
        J2 = np.var(profits.sum())   # profit variance
        print(list(profits))
        return J1, J2


# -------------------------------
# Conventional Producers Perspective (Aggregate Stochastic)
# -------------------------------
class ConventionalProducers:
    def __init__(self, env: MarketEnvironment):
        self.env = env

    def expected_profit(self, c_t, b_t, p_t):
        """Expected profit for conventional producers under price c_t."""
        accepted = p_t <= c_t
        return np.mean((c_t - p_t[accepted]) * b_t[accepted])

    def evaluate_objectives(self, c, b_conv, p_conv):
        T = self.env.T
        J1_list, J2_list = [], []
        for i in range(3):
            profits = np.array([
                (c[t] - p_conv[i, t]) * b_conv[i, t] * (p_conv[i, t] <= c[t])
                for t in range(T)
            ])
            J1_list.append(np.mean(profits.sum()))
            J2_list.append(np.var(b_conv[i]))
        return J1_list, J2_list


# -------------------------------
# Market Operator Perspective
# -------------------------------
class MarketOperator:
    def __init__(self, env: MarketEnvironment):
        self.env = env

    def evaluate_objectives(self, D, P_total, c):
        """Compute imbalance and price volatility."""
        q_u, q_o = self.env.q_u, self.env.q_o
        imbalance = P_total - D
        imbalance_costs = q_u * np.maximum(0, -imbalance) + q_o * np.maximum(0, imbalance)
        J1 = np.mean(imbalance_costs.sum())  # expected imbalance cost
        J2 = np.var(c)                       # price volatility
        return J1, J2


# -------------------------------
# System Planner Perspective
# -------------------------------
class SystemPlanner:
    def __init__(self, env: MarketEnvironment):
        self.env = env

    def evaluate_objectives(self, D, b_conv, P_total):
        """Evaluate welfare, reliability, and emissions."""
        alpha = self.env.alpha
        epsilon = 5  # reliability tolerance

        # (1) Simplified welfare measure
        U = lambda d: 100 * np.log(1 + d)
        C = lambda b: 0.5 * b ** 2
        welfare = np.sum([U(D[t]) - np.sum([C(b_conv[i, t]) for i in range(3)]) for t in range(self.env.T)])

        # (2) Reliability (probability of large imbalance)
        imbalance = np.abs(P_total - D)
        reliability = np.mean(imbalance > epsilon)

        # (3) Environmental impact
        emissions = np.sum([alpha[i] * np.sum(b_conv[i, :]) for i in range(3)])

        J1 = welfare
        J2 = reliability
        J3 = emissions
        return J1, J2, J3


# ============================================================
# 3. UNIFIED SIMULATION INTERFACE
# ============================================================

def simulate_market(b_R, p_R, P_R, policy_params, seed=None):
    """
    Unified simulation interface.
    Inputs:
        b_R, p_R : arrays of renewable bid quantities and prices
        P_R      : array of renewable generation realizations
        policy_params : dict with possible keys {'q_u', 'q_o'}
    Output:
        dict of objective values for all perspectives
    """

    if seed is not None:
        np.random.seed(seed)

    # --- Initialize environment ---
    env = MarketEnvironment(q_u=policy_params.get('q_u', 80),
                            q_o=policy_params.get('q_o', 20))

    # --- Sample exogenous uncertainties ---
    D = env.sample_demand()
    b_conv, p_conv = env.sample_conventional_bids()

    # --- Determine market-clearing price for each hour ---
    c = np.zeros(env.T)
    for t in range(env.T):
        c[t] = env.clearing_price(D[t], b_conv[:, t], p_conv[:, t], b_R[t], p_R[t])

    # --- Compute total production ---
    P_total = P_R + np.sum(b_conv, axis=0)

    # --- Instantiate perspectives ---
    renewable = RenewableProducer(env)
    conventional = ConventionalProducers(env)
    operator = MarketOperator(env)
    planner = SystemPlanner(env)

    # --- Evaluate objectives ---
    J_R = renewable.evaluate_objectives(b_R, p_R, c, P_R)
    J_C = conventional.evaluate_objectives(c, b_conv, p_conv)
    J_M = operator.evaluate_objectives(D, P_total, c)
    J_G = planner.evaluate_objectives(D, b_conv, P_total)

    # --- Return all objectives ---
    results = {
        "RenewableProducer": {"ExpectedProfit": J_R[0], "ProfitVariance": J_R[1]},
        "ConventionalProducers": {"ExpectedProfits": J_C[0], "BidVariances": J_C[1]},
        "MarketOperator": {"ImbalanceCost": J_M[0], "PriceVariance": J_M[1]},
        "SystemPlanner": {"Welfare": J_G[0], "Reliability": J_G[1], "Emissions": J_G[2]},
    }

    return results



# ============================================================
# EXAMPLE CALL TO THE UNIFIED MARKET SIMULATION
# ============================================================

# Import the unified model (assuming the above code is saved as unified_market.py)
# from unified_market import simulate_market

# --- Example control variables for the renewable producer ---
T = 24  # number of hourly intervals

# Renewable producer's bid quantities (in MWh)
b_R = np.array([20 for t in range(T)])

# Renewable producer's bid prices ($/MWh)
p_R = np.array([35 for t in range(T)])

# Realized renewable generation (deeply uncertain)
P_R = np.array([np.random.uniform(10,30) for _ in range(T)])

# --- Example policy parameters (system planner's controls) ---
policy_params = {
    'q_u': 80,   # penalty for under-delivery
    'q_o': 20    # penalty for over-delivery
}

# --- Simulate the unified model ---
results = simulate_market(
    b_R=b_R,
    p_R=p_R,
    P_R=P_R,
    policy_params=policy_params,
    seed=123  # random seed for reproducibility
)

# --- Display results ---
# print("Unified Market Simulation Results:")
# for perspective, metrics in results.items():
#     print(f"\n{perspective}:")
#     for k, v in metrics.items():
#         print(f"  {k}: {v}")
