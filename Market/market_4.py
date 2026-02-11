import numpy as np
from typing import Dict, Any


# ============================================================
# Common Environment
# ============================================================

class MarketEnvironment:
    """
    Shared stochastic environment and market-clearing mechanism.
    """

    def __init__(self, constant_params, uncertain_params):
        self.const = constant_params
        self.uncertain = uncertain_params

    def sample_exogenous_processes(self, t: int):
        D_t = np.random.normal(
            self.const["mu_D"], self.const["sigma_D"]
        )
        G_t = np.random.normal(
            self.const["mu_G"], self.const["sigma_G"]
        )

        p_conv = [
            np.random.normal(mu, sigma)
            for mu, sigma in self.uncertain["conv_price_params"]
        ]

        p_solar = np.random.normal(
            self.uncertain["solar_price_mu"],
            self.uncertain["solar_price_sigma"],
        )

        b_solar = max(
            0.0,
            self.const["solar_a"]
            + self.const["solar_b"] * np.cos(2 * np.pi * t / 24),
        )

        return {
            "D_t": D_t,
            "G_t": G_t,
            "p_conv": p_conv,
            "p_solar": p_solar,
            "b_solar": b_solar,
        }

    def clear_market(self, bids, D_t):
        """
        Merit-order clearing with full acceptance of all bids
        priced at or below the clearing price.
        """

        # bids: list of (quantity, price, name)
        bids_sorted = sorted(bids, key=lambda x: x[1])

        cumulative_supply = 0.0
        clearing_price = None

        # Step 1: find clearing price
        for q, p, _ in bids_sorted:
            cumulative_supply += q
            if cumulative_supply >= D_t:
                clearing_price = p
                break

        # If demand is never met, highest price clears
        if clearing_price is None:
            clearing_price = bids_sorted[-1][1]

        # Step 2: accept all bids with price <= clearing price
        dispatch = {}
        for q, p, name in bids_sorted:
            if p <= clearing_price:
                dispatch[name] = q
            else:
                dispatch[name] = 0.0

        return clearing_price, dispatch


# ============================================================
# Perspective: Wind Producer
# ============================================================

class WindProducer:
    def __init__(self, decision_vars: Dict[str, Any]):
        self.bids = decision_vars["wind_bids"]  # list of (b_wt, p_wt)

    def get_bid(self, t: int):
        return self.bids[t]


# ============================================================
# Perspective: Regulator
# ============================================================

class SystemRegulator:
    def __init__(self, decision_vars: Dict[str, Any]):
        self.q_u = decision_vars["q_u"]


# ============================================================
# Unified Simulator
# ============================================================

class UnifiedSimulator:
    def __init__(
        self,
        environment: MarketEnvironment,
        wind: WindProducer,
        regulator: SystemRegulator,
    ):
        self.env = environment
        self.wind = wind
        self.regulator = regulator

        # Storage for outcomes
        self.results = {
            "wind_profit": 0.0,
            "wind_imbalance": 0.0,
            "prices": [],
            "wind_dispatch": [],
            "penalty_revenue": 0.0,
        }

    def run(self):
        for t in range(24):
            exo = self.env.sample_exogenous_processes(t)

            # Build bids
            bids = []

            # Conventional producers
            for i, b_i in enumerate(self.env.const["conv_capacities"]):
                bids.append((b_i, exo["p_conv"][i], f"conv_{i}"))

            # Solar producer
            bids.append((exo["b_solar"], exo["p_solar"], "solar"))

            # Wind producer
            b_wt, p_wt = self.wind.get_bid(t)
            bids.append((b_wt, p_wt, "wind"))

            # Market clearing
            c_t, dispatch = self.env.clear_market(bids, exo["D_t"])
            x_wt = dispatch.get("wind", 0.0)

            # Real-time imbalance
            u_t = max(0.0, x_wt - exo["G_t"])
            penalty = self.regulator.q_u * u_t

            # Store results
            self.results["wind_profit"] += c_t * x_wt - penalty
            self.results["wind_imbalance"] += u_t
            self.results["penalty_revenue"] += penalty
            self.results["prices"].append(c_t)
            self.results["wind_dispatch"].append(x_wt)

        return self.results


# ============================================================
# Interface Function
# ============================================================

def simulate_unified_model(
    decision_vars: Dict[str, Any],
    uncertain_params: Dict[str, Any],
    constant_params: Dict[str, Any],
) -> Dict[str, float]:
    """
    Interface function required by the specification.

    Inputs
    ------
    decision_vars:
        {
            "wind_bids": [(b_w1, p_w1), ..., (b_w24, p_w24)],
            "q_u": penalty_per_MWh
        }

    uncertain_params:
        {
            "conv_price_params": [(mu_p1, sigma_p1), (mu_p2, sigma_p2), (mu_p3, sigma_p3)],
            "solar_price_mu": float,
            "solar_price_sigma": float
        }

    constant_params:
        {
            "mu_D", "sigma_D",
            "mu_G", "sigma_G",
            "solar_a", "solar_b",
            "conv_capacities": [b1, b2, b3]
        }

    Returns
    -------
    Dictionary mapping objective names to values.
    """

    env = MarketEnvironment(constant_params, uncertain_params)
    wind = WindProducer(decision_vars)
    regulator = SystemRegulator(decision_vars)

    simulator = UnifiedSimulator(env, wind, regulator)
    results = simulator.run()

    objectives = {
        # Wind producer objectives
        "wind_expected_profit": results["wind_profit"],
        "wind_total_imbalance": results["wind_imbalance"],
        "wind_profit_variance_proxy": np.var(results["prices"]),

        # Regulator objectives
        "system_price_variance": np.var(results["prices"]),
        "renewable_utilization": np.sum(results["wind_dispatch"]),
        "total_penalty_revenue": results["penalty_revenue"],
    }

    return objectives
