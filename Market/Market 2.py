import numpy as np

# ============================================================
# 1. Common Market Environment
# ============================================================

class MarketEnvironment:
    def __init__(self, T, mu_D, sigma_D, q_u, q_o):
        """
        Shared environment parameters
        """
        self.T = T
        self.mu_D = mu_D
        self.sigma_D = sigma_D
        self.q_u = q_u
        self.q_o = q_o

    def sample_demand(self):
        """Generate stochastic hourly demand."""
        return np.random.normal(self.mu_D, self.sigma_D)

    def transition(self, state, theta, omega_t):
        """
        Update the state for next hour given renewable uncertainty.
        state = (D_t, P_t, c_t)
        """
        D_next = np.random.normal(self.mu_D, self.sigma_D)
        P_next = self.renewable_generation(theta, omega_t)
        return D_next, P_next

    @staticmethod
    def renewable_generation():
        """
        Renewable generation model under deep uncertainty.
        Example functional form (can be replaced by any model):
        """
        return np.random.uniform(10,30)

# ============================================================
# 2. Renewable Producer Module
# ============================================================

class RenewableProducer:
    def __init__(self, env):
        self.env = env

    def profit(self, b_t, p_t, c_t, P_t):
        """
        Hourly profit function for renewable producer.
        """
        if p_t > c_t:
            return 0.0
        elif P_t < b_t:
            return b_t * c_t - self.env.q_u * (b_t - P_t)
        else:
            return b_t * c_t - self.env.q_o * (P_t - b_t)

    def objectives(self, bids, prices, clear_prices, generations):
        """
        Compute renewable producer objectives across all hours.
        """
        profit_list, imbalance_list = [], []
        for t in range(self.env.T):
            pi_t = self.profit(bids[t], prices[t], clear_prices[t], generations[t])
            profit_list.append(pi_t)
            imbalance_list.append(abs(generations[t] - bids[t]))
        J_R1 = np.mean(np.sum(profit_list))        # expected profit
        J_R2 = -np.mean(np.sum(imbalance_list))    # reliability
        print(profit_list)
        return {"J_R1": J_R1, "J_R2": J_R2}

# ============================================================
# 3. Conventional Producers Module
# ============================================================

class ConventionalProducers:
    def __init__(self, env, params):
        """
        params = {
            i: {"mu_b": ..., "sigma_b": ..., "mu_p": ..., "sigma_p": ..., "MC": ...}
        }
        """
        self.env = env
        self.params = params

    def sample_bids(self):
        """Generate stochastic bids for all conventional producers."""
        bids, prices = {}, {}
        for i, prm in self.params.items():
            bids[i] = np.random.normal(prm["mu_b"], prm["sigma_b"], self.env.T)
            prices[i] = np.random.normal(prm["mu_p"], prm["sigma_p"], self.env.T)
        return bids, prices

    def objectives(self, bids, prices, clear_prices):
        """Expected profits for each conventional producer."""
        results = {}
        for i, prm in self.params.items():
            profits = []
            for t in range(self.env.T):
                if prices[i][t] <= clear_prices[t]:
                    profits.append((clear_prices[t] - prm["MC"]) * bids[i][t])
                else:
                    profits.append(0.0)
            results[f"J_C{i}"] = np.mean(np.sum(profits))
        return results

# ============================================================
# 4. Market Operator Module
# ============================================================

class MarketOperator:
    def __init__(self, env):
        self.env = env

    @staticmethod
    def clear_market(all_bids, all_prices, demand):
        """
        Determine market-clearing price given supply and demand.
        Simple merit-order rule: lowest price bids first until demand met.
        """
        offers = []
        for i in all_bids.keys():
            offers.append((all_prices[i], all_bids[i]))
        offers = sorted(offers, key=lambda x: x[0])
        total, price = 0.0, 0.0
        for p, b in offers:
            total += b
            price = p
            if total >= demand:
                break
        return price

    def objectives(self, all_bids, all_prices, D_t, c_t, P_t, c_prev):
        """
        Compute operator objectives for a single hour.
        """
        supply = sum([all_bids[i] for i in all_bids.keys() if all_prices[i] <= c_t])
        imbalance = (supply - D_t)**2
        price_stability = (c_t - c_prev)**2
        reliability = self.env.q_u * max(b := all_bids["renew"], 0)**0 + \
                      self.env.q_o * max(P_t - b, 0)**0
        return {"imbalance": imbalance, "price_stability": price_stability, "reliability": reliability}

# ============================================================
# 5. Regulator / Policy-Maker Module
# ============================================================

class Regulator:
    def __init__(self, env, emission_factors):
        self.env = env
        self.emission_factors = emission_factors

    def objectives(self, conv_bids, conv_prices, c_t, D_t, utility_func):
        """
        Evaluate regulator objectives for a given hour.
        """
        # Emission reduction
        emissions = sum([self.emission_factors[i] * conv_bids[i]
                         for i in conv_bids.keys() if conv_prices[i] <= c_t])

        # Social welfare (utility - payment)
        welfare = utility_func(D_t) - c_t * sum([conv_bids[i] for i in conv_bids.keys()
                                                 if conv_prices[i] <= c_t])
        return {"emission": -emissions, "welfare": welfare, "efficiency": -c_t**2}

# ============================================================
# 6. Unified Simulation Interface
# ============================================================

def simulate_market(controls, uncertain_params):
    """
    Simulate the unified electricity market model.

    Inputs:
    - controls: dictionary containing control variables for renewable producer and regulator, e.g.,
        {
            "renewable_bids": np.array([...]),
            "renewable_prices": np.array([...]),
            "policy": {"tau": ..., "eta": ..., "gamma": ...}
        }
    - uncertain_params: dictionary with stochastic parameters, e.g.,
        {
            "theta": {"scale": ..., "mean": ..., "var": ...},
            "omega": np.array([...])   # random generation factors
        }

    Output:
    - Dictionary with objective values for all perspectives.
    """
    # === Initialize environment and modules ===
    env = MarketEnvironment(T=24, mu_D=100, sigma_D=10, q_u=40, q_o=20)

    conv_params = {
        1: {"mu_b": 30, "sigma_b": 5, "mu_p": 40, "sigma_p": 5, "MC": 25},
        2: {"mu_b": 40, "sigma_b": 5,  "mu_p": 45, "sigma_p": 5, "MC": 27},
        3: {"mu_b": 50, "sigma_b": 5,  "mu_p": 50, "sigma_p": 5, "MC": 28},
    }

    renewable = RenewableProducer(env)
    conventional = ConventionalProducers(env, conv_params)
    operator = MarketOperator(env)
    regulator = Regulator(env, emission_factors={1: 0.5, 2: 0.6, 3: 0.7})

    # === Sample stochastic conventional bids ===
    conv_bids, conv_prices = conventional.sample_bids()

    # === Initialize containers ===
    clear_prices = []
    generations = []
    D_series = []
    c_prev = 0.0

    # === Simulation loop ===
    for t in range(env.T):
        # Sample demand and renewable generation
        D_t = env.sample_demand()
        P_t = env.renewable_generation()

        # Combine bids
        all_bids = {i: conv_bids[i][t] for i in conv_bids.keys()}
        all_prices = {i: conv_prices[i][t] for i in conv_prices.keys()}
        all_bids["renew"] = controls["renewable_bids"][t]
        all_prices["renew"] = controls["renewable_prices"][t]

        # Market clearing
        c_t = operator.clear_market(all_bids, all_prices, D_t)

        # Store results
        clear_prices.append(c_t)
        generations.append(P_t)
        D_series.append(D_t)
        c_prev = c_t

    # === Compute objectives ===
    renewable_obj = renewable.objectives(
        bids=controls["renewable_bids"],
        prices=controls["renewable_prices"],
        clear_prices=np.array(clear_prices),
        generations=np.array(generations),
    )

    conventional_obj = conventional.objectives(conv_bids, conv_prices, np.array(clear_prices))

    # Simplified welfare utility (e.g., quadratic utility of demand)
    U = lambda D: 1000 * np.log(1 + D)
    regulator_obj = {"J_G1": 0, "J_G2": 0, "J_G3": 0}
    for t in range(env.T):
        reg_t = regulator.objectives(
            {i: conv_bids[i][t] for i in conv_bids.keys()},
            {i: conv_prices[i][t] for i in conv_prices.keys()},
            clear_prices[t],
            D_series[t],
            U,
        )
        regulator_obj["J_G1"] += reg_t["emission"]
        regulator_obj["J_G2"] += reg_t["welfare"]
        regulator_obj["J_G3"] += reg_t["efficiency"]

    # === Return all objective values ===
    return {
        "RenewableProducer": renewable_obj,
        "ConventionalProducers": conventional_obj,
        "Regulator": regulator_obj
    }


controls_R = {"renewable_bids": [20 for i in range(24)], "renewable_prices": [35 for k in range(24)]}  # renewable bid quantity & price
controls_G = {"q_u": 80, "q_o": 20, "tau_t": 2}

# Simulate one scenario of renewable generation
results = simulate_market(controls_R, controls_G)

# results contains all objective values (no need to execute here)
# print(results["Renewable_Profit"])