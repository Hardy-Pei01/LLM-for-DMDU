import numpy as np
from typing import Dict, Any

class MarketEnvironment:
    """
    Shared stochastic environment and market-clearing mechanism.
    """

    def __init__(self, constants, uncertainties):
        self.constants = constants
        self.uncertainties = uncertainties
        self.T = constants.get("T", 24)

    def sample_state(self, rng):
        """
        Sample exogenous stochastic variables for all hours.
        """
        T = self.T

        # Demand (distribution parameters are constants)
        D = rng.normal(
            self.constants["mu_D"],
            self.constants["sigma_D"],
            size=T
        )

        # Conventional bid prices (uncertain)
        p_conv = {
            i: rng.normal(
                self.uncertainties["mu_p_conv"][i],
                self.uncertainties["sigma_p_conv"][i],
                size=T
            )
            for i in range(len(self.constants["b_conv"]))
        }

        # Solar bid prices (uncertain)
        p_solar = rng.normal(
            self.uncertainties["mu_p_solar"],
            self.uncertainties["sigma_p_solar"],
            size=T
        )

        # Wind real-time generation (distribution parameters are constants)
        G_wind = rng.normal(
            self.constants["mu_G"],
            self.constants["sigma_G"],
            size=T
        )

        return D, p_conv, p_solar, G_wind

    def solar_quantity(self, t):
        a = self.constants["a"]
        b = self.constants["b"]
        return max(0.0, a + b * np.cos(2 * np.pi * (t + 1) / 24))

    def clear_market(self, D_t, bids):
        """
        Merit-order clearing for a single hour.
        """
        bids_sorted = sorted(bids, key=lambda x: x[1])

        supply = 0.0
        clearing_price = bids_sorted[-1][1]

        for q, p, _ in bids_sorted:
            supply += q
            if supply >= D_t:
                clearing_price = p
                break

        dispatch = {
            pid: q if p <= clearing_price else 0.0
            for q, p, pid in bids_sorted
        }

        return clearing_price, dispatch


class WindProducer:
    def __init__(self, decisions):
        self.bids = decisions["wind_bids"]

    def evaluate(self, prices, dispatch, generation, penalty):
        revenue = 0.0
        imbalance = 0.0

        for t in range(len(prices)):
            x = dispatch[t]
            g = generation[t]
            u = max(0.0, x - g)

            revenue += prices[t] * x - penalty * u
            imbalance += u

        return {
            "expected_revenue": revenue,
            "expected_imbalance": imbalance
        }


class Regulator:
    def __init__(self, decisions):
        self.penalty = decisions["imbalance_penalty"]

    def evaluate(self, prices, wind_dispatch, wind_imbalance):
        return {
            "expected_imbalance": sum(wind_imbalance),
            "price_variance": np.var(prices),
            "renewable_integration": sum(wind_dispatch),
            "penalty_transfers": self.penalty * sum(wind_imbalance)
        }


def simulate_unified_model(
    decision_vars,
    uncertain_params,
    constant_params,
    random_seed=0
):
    """
    Unified simulation interface with correct parameter roles.
    """

    rng = np.random.default_rng(random_seed)

    # Initialize components
    env = MarketEnvironment(constant_params, uncertain_params)
    wind = WindProducer(decision_vars)
    regulator = Regulator(decision_vars)

    # Sample environment
    D, p_conv, p_solar, G_wind = env.sample_state(rng)

    prices = []
    wind_dispatch = []
    wind_imbalance = []

    for t in range(env.T):
        bids = []

        # Conventional producers
        for i, b_i in enumerate(constant_params["b_conv"]):
            bids.append((b_i, p_conv[i][t], f"conv_{i}"))

        # Solar producer
        b_s = env.solar_quantity(t)
        bids.append((b_s, p_solar[t], "solar"))

        # Wind producer
        b_w, p_w = wind.bids[t]
        bids.append((b_w, p_w, "wind"))

        # Market clearing
        c_t, dispatch = env.clear_market(D[t], bids)

        prices.append(c_t)

        x_w = dispatch["wind"]
        wind_dispatch.append(x_w)
        wind_imbalance.append(max(0.0, x_w - G_wind[t]))

    # Evaluate objectives
    wind_objectives = wind.evaluate(
        prices, wind_dispatch, G_wind, regulator.penalty
    )

    regulator_objectives = regulator.evaluate(
        prices, wind_dispatch, wind_imbalance
    )

    return {
        "wind_producer_objectives": wind_objectives,
        "regulator_objectives": regulator_objectives
    }
