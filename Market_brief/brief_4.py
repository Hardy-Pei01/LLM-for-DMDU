import numpy as np
from typing import Dict, Any


# ============================================================
# Common Environment
# ============================================================

class MarketEnvironment:
    """
    Shared stochastic environment and market-clearing mechanism.
    Contains no stakeholder decision variables.
    """

    def __init__(self, constants, uncertainties):
        self.mu_D = constants["mu_D"]
        self.sigma_D = constants["sigma_D"]
        self.mu_w = constants["mu_w"]
        self.sigma_w = constants["sigma_w"]
        self.a = constants["solar_a"]
        self.b = constants["solar_b"]
        self.conventional_capacities = constants["conventional_capacities"]
        self.hours = constants.get("hours", 24)

        self.conv_price_params = uncertainties["conventional_price_params"]
        self.solar_price_params = uncertainties["solar_price_params"]

    # ----------------------------
    # Stochastic processes
    # ----------------------------

    def sample_demand(self):
        return max(0.0, np.random.normal(self.mu_D, self.sigma_D))

    def sample_wind_production(self):
        return max(0.0, np.random.normal(self.mu_w, self.sigma_w))

    def sample_conventional_prices(self):
        prices = []
        for params in self.conv_price_params:
            prices.append(np.random.normal(params["mu"], params["sigma"]))
        return prices

    def sample_solar_price(self):
        return np.random.normal(
            self.solar_price_params["mu"],
            self.solar_price_params["sigma"]
        )

    def solar_quantity(self, hour):
        return max(0.0, self.a + self.b * np.cos(2 * np.pi * hour / 24))

    # ----------------------------
    # Corrected market clearing
    # ----------------------------

    def clear_market(self, bids, demand):
        """
        bids: list of tuples (quantity, price, producer_id)

        Rule:
        1. Determine clearing price as the marginal price where
           cumulative supply first meets or exceeds demand.
        2. Fully accept ALL bids with price <= clearing price,
           even if total supply exceeds demand.
        """

        # Sort bids by price (merit order)
        bids_sorted = sorted(bids, key=lambda x: x[1])

        cumulative_supply = 0.0
        clearing_price = None

        # Step 1: find clearing price
        for q, p, _ in bids_sorted:
            cumulative_supply += q
            clearing_price = p
            if cumulative_supply >= demand:
                break

        # Step 2: accept all bids priced <= clearing price
        dispatch = {}
        for q, p, pid in bids_sorted:
            if p <= clearing_price:
                dispatch[pid] = q
            else:
                dispatch[pid] = 0.0

        return dispatch, clearing_price


# ============================================================
# Wind Producer Perspective
# ============================================================

class WindProducer:
    """
    Wind producer decisions and objectives.
    """

    def __init__(self, decisions: Dict[str, Any]):
        self.bid_quantities = decisions["bid_quantities"]  # list length 24
        self.bid_prices = decisions["bid_prices"]          # list length 24

    def profit(self, prices, dispatch, wind_output, imbalance_penalty):
        """
        Compute total profit over all hours.
        """
        profit = 0.0
        imbalance_volume = 0.0

        for h in range(len(prices)):
            dispatched = dispatch[h]
            produced = wind_output[h]
            shortfall = max(0.0, dispatched - produced)

            revenue = prices[h] * dispatched
            penalty = imbalance_penalty * shortfall

            profit += revenue - penalty
            imbalance_volume += shortfall

        return profit, imbalance_volume


# ============================================================
# System Regulator Perspective
# ============================================================

class SystemRegulator:
    """
    System regulator decisions and objectives.
    """

    def __init__(self, decisions: Dict[str, Any]):
        self.imbalance_penalty = decisions["imbalance_penalty"]

    def objectives(self, prices, imbalance_volume, wind_profit):
        return {
            "expected_imbalance": imbalance_volume,
            "price_variance": np.var(prices),
            "wind_revenue": wind_profit,
            "total_system_payment": np.sum(prices)
        }


# ============================================================
# Unified Simulation Interface
# ============================================================

def simulate_market(
    decision_variables: Dict[str, Any],
    uncertain_parameters: Dict[str, Any],
    constant_parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Interface function for the unified model.

    Inputs
    ------
    decision_variables:
        {
            "wind": {
                "bid_quantities": [...],
                "bid_prices": [...]
            },
            "regulator": {
                "imbalance_penalty": float
            }
        }

    uncertain_parameters:
        {
            "conventional_price_params": [
                {"mu": float, "sigma": float}, ...
            ],
            "solar_price_params": {
                "mu": float,
                "sigma": float
            }
        }

    constant_parameters:
        {
            "mu_D": float,
            "sigma_D": float,
            "mu_w": float,
            "sigma_w": float,
            "solar_a": float,
            "solar_b": float,
            "conventional_capacities": [float, float, float],
            "hours": int
        }

    Returns
    -------
    Dictionary of objective values for each perspective.
    """

    # Instantiate components
    env = MarketEnvironment(constant_parameters, uncertain_parameters)
    wind = WindProducer(decision_variables["wind"])
    regulator = SystemRegulator(decision_variables["regulator"])

    prices = []
    wind_dispatch = []
    wind_output = []

    # ----------------------------
    # Simulate one day
    # ----------------------------

    for h in range(env.hours):
        demand = env.sample_demand()
        wind_prod = env.sample_wind_production()

        conv_prices = env.sample_conventional_prices()
        solar_price = env.sample_solar_price()

        bids = []

        # Conventional bids
        for i, cap in enumerate(env.conventional_capacities):
            bids.append((cap, conv_prices[i], f"conv_{i}"))

        # Solar bid
        solar_q = env.solar_quantity(h)
        bids.append((solar_q, solar_price, "solar"))

        # Wind bid
        bids.append((
            wind.bid_quantities[h],
            wind.bid_prices[h],
            "wind"
        ))

        dispatch, clearing_price = env.clear_market(bids, demand)

        prices.append(clearing_price)
        wind_dispatch.append(dispatch.get("wind", 0.0))
        wind_output.append(wind_prod)

    # ----------------------------
    # Evaluate objectives
    # ----------------------------

    wind_profit, imbalance_volume = wind.profit(
        prices,
        wind_dispatch,
        wind_output,
        regulator.imbalance_penalty
    )

    regulator_objectives = regulator.objectives(
        prices,
        imbalance_volume,
        wind_profit
    )

    return {
        "wind_producer": {
            "expected_profit": wind_profit,
            "expected_imbalance": imbalance_volume
        },
        "system_regulator": regulator_objectives
    }
