import numpy as np
from typing import Dict, Any


# ============================================================
# Common Environment
# ============================================================

class MarketEnvironment:
    """
    Shared stochastic environment and market-clearing logic.
    """

    def __init__(self, const: Dict[str, Any], uncertain: Dict[str, Any]):
        self.const = const
        self.uncertain = uncertain

    def sample_state(self, rng: np.random.Generator, h: int):
        D = max(0.0, rng.normal(self.const["mu_D"], self.const["sigma_D"]))

        W = rng.normal(self.const["mu_W"], self.const["sigma_W"])
        W = np.clip(W, 0.0, self.const["W_max"])

        p_conv = rng.normal(
            self.uncertain["mu_conv"],
            self.uncertain["sigma_conv"],
            size=self.const["n_conv"]
        )

        p_solar = rng.normal(
            self.uncertain["mu_solar"],
            self.uncertain["sigma_solar"]
        )

        return {
            "D": D,
            "W": W,
            "p_conv": p_conv,
            "p_solar": p_solar
        }

    def solar_quantity(self, h: int):
        a = self.const["solar_a"]
        b = self.const["solar_b"]
        return max(0.0, a + b * np.cos(2 * np.pi * h / 24))

    def clear_market(self, state, wind_bid):
        """
        Merit-order clearing with full acceptance of all bids
        priced at or below the clearing price.
        """

        bids = []

        # Conventional bids
        for q, p in zip(self.const["q_conv"], state["p_conv"]):
            bids.append((p, q, "conv"))

        # Solar bid
        bids.append((
            state["p_solar"],
            self.solar_quantity(wind_bid["hour"]),
            "solar"
        ))

        # Wind bid
        bids.append((
            wind_bid["price"],
            wind_bid["quantity"],
            "wind"
        ))

        # Sort bids by price
        bids.sort(key=lambda x: x[0])

        # --------------------------------------------------
        # Step 1: Determine clearing price
        # --------------------------------------------------
        cumulative = 0.0
        clearing_price = bids[-1][0]  # fallback (very high demand case)

        for p, q, _ in bids:
            cumulative += q
            if cumulative >= state["D"]:
                clearing_price = p
                break

        # --------------------------------------------------
        # Step 2: Fully accept all bids priced <= clearing price
        # --------------------------------------------------
        wind_dispatch = 0.0

        for p, q, label in bids:
            if p <= clearing_price and label == "wind":
                wind_dispatch = q
                break

        return clearing_price, wind_dispatch


# ============================================================
# Wind Producer Perspective
# ============================================================

class WindProducer:
    """
    Wind producer decision logic and objectives.
    """

    def __init__(self, decision_vars: Dict[str, Any], penalty: float):
        self.q = decision_vars["wind_quantity"]   # array-like, length 24
        self.p = decision_vars["wind_price"]      # array-like, length 24
        self.penalty = penalty

    def evaluate(self, env: MarketEnvironment, rng, n_scenarios=1000):
        revenues = []

        for _ in range(n_scenarios):
            daily_revenue = 0.0

            for h in range(24):
                state = env.sample_state(rng, h)

                price, q_da = env.clear_market(
                    state,
                    {
                        "hour": h,
                        "price": self.p[h],
                        "quantity": self.q[h]
                    }
                )

                delivered = min(state["W"], q_da)
                shortfall = max(0.0, q_da - state["W"])

                revenue = price * q_da - self.penalty * shortfall
                daily_revenue += revenue

            revenues.append(daily_revenue)

        revenues = np.array(revenues)

        return {
            "W_expected_revenue": revenues.mean(),
            "W_revenue_variance": revenues.var()
        }


# ============================================================
# Regulator Perspective
# ============================================================

class Regulator:
    """
    System regulator objectives.
    """

    def __init__(self, penalty: float):
        self.penalty = penalty

    def evaluate(self, env: MarketEnvironment, wind: WindProducer, rng, n_scenarios=1000):
        total_imbalance = []
        total_cost = []
        price_variance = []

        for _ in range(n_scenarios):
            imbalance = 0.0
            system_cost = 0.0
            prices = []

            for h in range(24):
                state = env.sample_state(rng, h)

                price, q_da = env.clear_market(
                    state,
                    {
                        "hour": h,
                        "price": wind.p[h],
                        "quantity": wind.q[h]
                    }
                )

                shortfall = max(0.0, q_da - state["W"])

                imbalance += shortfall
                system_cost += price * state["D"] + self.penalty * shortfall
                prices.append(price)

            total_imbalance.append(imbalance)
            total_cost.append(system_cost)
            price_variance.append(np.var(prices))

        return {
            "R_expected_imbalance": np.mean(total_imbalance),
            "R_expected_system_cost": np.mean(total_cost),
            "R_price_variance": np.mean(price_variance)
        }


# ============================================================
# Unified Model Interface
# ============================================================

def run_unified_market_model(
    decision_vars: Dict[str, Any],
    uncertain_params: Dict[str, Any],
    constant_params: Dict[str, Any],
    n_scenarios: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Unified interface function.

    Inputs:
    - decision_vars: dict of decision variables (wind bids, penalty)
    - uncertain_params: dict of uncertain distribution parameters
    - constant_params: dict of fixed model parameters

    Returns:
    - dict mapping objective names to values
    """

    rng = np.random.default_rng(seed)

    env = MarketEnvironment(constant_params, uncertain_params)

    wind = WindProducer(
        decision_vars=decision_vars,
        penalty=decision_vars["penalty"]
    )

    regulator = Regulator(
        penalty=decision_vars["penalty"]
    )

    results = {}
    results.update(wind.evaluate(env, rng, n_scenarios))
    results.update(regulator.evaluate(env, wind, rng, n_scenarios))

    return results
