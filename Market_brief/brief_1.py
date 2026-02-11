import numpy as np
from typing import Dict, Any


# ============================================================
# Common Environment
# ============================================================

class ElectricityMarketEnvironment:
    """
    Shared stochastic environment and market-clearing mechanism.
    """

    def __init__(self, constants: Dict[str, Any], uncertainties: Dict[str, Any]):
        self.constants = constants
        self.uncertainties = uncertainties

        self.T = constants["T"]
        self.rng = np.random.default_rng(uncertainties.get("seed", None))

    # ------------------------
    # Stochastic processes
    # ------------------------

    def sample_demand(self):
        mu, sigma = self.constants["mu_D"], self.constants["sigma_D"]
        return max(0.0, self.rng.normal(mu, sigma))

    def sample_wind(self):
        mu, sigma = self.constants["mu_W"], self.constants["sigma_W"]
        return max(0.0, self.rng.normal(mu, sigma))

    def sample_coal_prices(self):
        prices = []
        for mu, sigma in self.uncertainties["coal_price_params"]:
            prices.append(self.rng.normal(mu, sigma))
        return prices

    def sample_solar_price(self):
        mu, sigma = self.uncertainties["solar_price_params"]
        return self.rng.normal(mu, sigma)

    def solar_quantity(self, t):
        a, b = self.constants["solar_a"], self.constants["solar_b"]
        return max(0.0, a + b * np.cos(2 * np.pi * t / 24))

    # ------------------------
    # Market clearing
    # ------------------------

    def clear_market(self, demand, bids):
        """
        Merit-order clearing with full acceptance at clearing price.

        bids: list of dicts with keys {agent, quantity, price}

        Returns
        -------
        clearing_price : float
        dispatch : dict {agent: dispatched_quantity}
        """

        # Sort bids by ascending price
        bids_sorted = sorted(bids, key=lambda x: x["price"])

        # ------------------------------------------------
        # Phase 1: Determine clearing price
        # ------------------------------------------------
        cumulative_quantity = 0.0
        clearing_price = None

        for bid in bids_sorted:
            cumulative_quantity += bid["quantity"]
            if cumulative_quantity >= demand:
                clearing_price = bid["price"]
                break

        # If demand is never met, clearing price is highest bid
        if clearing_price is None:
            clearing_price = bids_sorted[-1]["price"]

        # ------------------------------------------------
        # Phase 2: Dispatch all bids priced ≤ clearing price
        # ------------------------------------------------
        dispatch = {}

        for bid in bids_sorted:
            if bid["price"] <= clearing_price:
                dispatch[bid["agent"]] = bid["quantity"]
            else:
                dispatch[bid["agent"]] = 0.0

        return clearing_price, dispatch


# ============================================================
# Wind Producer Perspective
# ============================================================

class WindProducer:
    """
    Wind producer decision logic and objective evaluation.
    """

    def __init__(self, decisions: Dict[str, Any], constants: Dict[str, Any]):
        self.q = decisions["wind_quantity"]      # list of length T
        self.p = decisions["wind_price"]         # list of length T
        self.capacity = constants["wind_capacity"]

    def bid(self, t):
        return {
            "agent": "wind",
            "quantity": min(self.q[t], self.capacity),
            "price": self.p[t],
        }

    def evaluate_objective(self, prices, dispatch, wind_realizations, penalty):
        revenue = 0.0
        for t in range(len(prices)):
            dispatched = dispatch[t].get("wind", 0.0)
            shortfall = max(0.0, dispatched - wind_realizations[t])
            revenue += dispatched * prices[t] - penalty * shortfall
        return revenue


# ============================================================
# Regulator Perspective
# ============================================================

class SystemRegulator:
    """
    Regulator policy parameters and multi-objective evaluation.
    """

    def __init__(self, decisions: Dict[str, Any]):
        self.penalty = decisions["penalty"]
        self.price_cap = decisions.get("price_cap", np.inf)

    def apply_price_cap(self, price):
        return min(price, self.price_cap)

    def evaluate_objectives(self, dispatch, prices, wind_realizations):
        imbalance = 0.0
        system_cost = 0.0
        wind_utilization = 0.0

        for t in range(len(prices)):
            for agent, qty in dispatch[t].items():
                system_cost += qty * prices[t]

            wind_dispatch = dispatch[t].get("wind", 0.0)
            shortfall = max(0.0, wind_dispatch - wind_realizations[t])
            imbalance += shortfall

            if wind_realizations[t] > 0:
                wind_utilization += min(wind_dispatch, wind_realizations[t]) / wind_realizations[t]

        return {
            "expected_imbalance": imbalance,
            "expected_system_cost": system_cost,
            "expected_wind_utilization": wind_utilization,
        }


# ============================================================
# Unified Simulation Interface
# ============================================================

def simulate_unified_model(
    decision_vars: Dict[str, Dict[str, Any]],
    uncertain_params: Dict[str, Any],
    constant_params: Dict[str, Any],
):
    """
    Interface function.

    Inputs
    ------
    decision_vars:
        {
            "wind": {
                "wind_quantity": [...],
                "wind_price": [...]
            },
            "regulator": {
                "penalty": float,
                "price_cap": float (optional)
            }
        }

    uncertain_params:
        {
            "coal_price_params": [(mu1, sigma1), (mu2, sigma2), (mu3, sigma3)],
            "solar_price_params": (mu, sigma),
            "seed": int (optional)
        }

    constant_params:
        {
            "T": 24,
            "mu_D": float,
            "sigma_D": float,
            "mu_W": float,
            "sigma_W": float,
            "coal_quantities": [Q1, Q2, Q3],
            "solar_a": float,
            "solar_b": float,
            "wind_capacity": float
        }

    Returns
    -------
    Dictionary with objective values for each perspective.
    """

    # Initialize components
    env = ElectricityMarketEnvironment(constant_params, uncertain_params)
    wind = WindProducer(decision_vars["wind"], constant_params)
    regulator = SystemRegulator(decision_vars["regulator"])

    prices = []
    dispatches = []
    wind_realizations = []

    # Simulation loop
    for t in range(constant_params["T"]):
        demand = env.sample_demand()
        wind_real = env.sample_wind()
        wind_realizations.append(wind_real)

        coal_prices = env.sample_coal_prices()
        solar_price = env.sample_solar_price()
        solar_qty = env.solar_quantity(t)

        bids = []

        # Coal bids
        for i, (q, p) in enumerate(zip(constant_params["coal_quantities"], coal_prices)):
            bids.append({"agent": f"coal_{i}", "quantity": q, "price": p})

        # Solar bid
        bids.append({"agent": "solar", "quantity": solar_qty, "price": solar_price})

        # Wind bid
        bids.append(wind.bid(t))

        price, dispatch = env.clear_market(demand, bids)
        price = regulator.apply_price_cap(price)

        prices.append(price)
        dispatches.append(dispatch)

    # Objective evaluation
    wind_profit = wind.evaluate_objective(
        prices, dispatches, wind_realizations, regulator.penalty
    )

    regulator_objectives = regulator.evaluate_objectives(
        dispatches, prices, wind_realizations
    )

    return {
        "wind_expected_profit": wind_profit,
        **regulator_objectives,
    }
