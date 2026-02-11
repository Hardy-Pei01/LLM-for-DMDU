import numpy as np
from typing import Dict, Any, List


# ============================================================
# Common Environment
# ============================================================

class MarketEnvironment:
    """
    Shared environment: stochastic processes, market clearing,
    and state transitions.
    """

    def __init__(self, constants: Dict[str, Any], uncertainties: Dict[str, Any]):
        self.constants = constants
        self.uncertainties = uncertainties

    def sample_demand(self, t: int) -> float:
        return np.random.normal(
            self.constants["mu_D"],
            self.constants["sigma_D"]
        )

    def sample_wind_generation(self, t: int) -> float:
        return np.random.normal(
            self.constants["mu_G"],
            self.constants["sigma_G"]
        )

    def sample_conventional_bids(self, t: int):
        bids = []
        for i in range(self.constants["n_conventional"]):
            p = np.random.normal(
                self.uncertainties["mu_pi"][i],
                self.uncertainties["sigma_pi"][i]
            )
            b = self.constants["b_i"][i]
            bids.append({"agent": f"conv_{i}", "b": b, "p": p})
        return bids

    def sample_solar_bid(self, t: int):
        a = self.constants["solar_a"]
        b = self.constants["solar_b"]
        quantity = max(0.0, a + b * np.cos(2 * np.pi * t / 24))
        price = np.random.normal(
            self.uncertainties["mu_ps"],
            self.uncertainties["sigma_ps"]
        )
        return {"agent": "solar", "b": quantity, "p": price}

    def clear_market(self, bids: List[Dict[str, float]], demand: float):
        """
        Merit-order clearing with uniform pricing.

        Any bid with price <= clearing price is accepted in full,
        even if total accepted supply exceeds demand.
        """
        bids_sorted = sorted(bids, key=lambda x: x["p"])

        # ----------------------------------------------------
        # Step 1: Determine clearing price
        # ----------------------------------------------------
        cumulative_supply = 0.0
        clearing_price = None

        for bid in bids_sorted:
            cumulative_supply += bid["b"]
            if cumulative_supply >= demand:
                clearing_price = bid["p"]
                break

        # ----------------------------------------------------
        # Step 2: Accept all bids with p <= clearing price
        # ----------------------------------------------------
        dispatch = {}
        for bid in bids_sorted:
            if bid["p"] <= clearing_price:
                dispatch[bid["agent"]] = bid["b"]
            else:
                dispatch[bid["agent"]] = 0.0

        return clearing_price, dispatch


# ============================================================
# Wind Producer Perspective
# ============================================================

class WindProducer:
    """
    Wind producer decision module.
    """

    def __init__(self, decisions: Dict[str, Any]):
        self.bids = decisions["wind_bids"]  # dict: t -> (b_wt, p_wt)

    def submit_bid(self, t: int):
        b, p = self.bids[t]
        return {"agent": "wind", "b": b, "p": p}

    def profit(self, clearing_price, dispatch, wind_generation, penalty):
        d_w = dispatch.get("wind", 0.0)
        shortfall = max(0.0, d_w - wind_generation)
        return clearing_price * d_w - penalty * shortfall


# ============================================================
# System Regulator Perspective
# ============================================================

class SystemRegulator:
    """
    Regulator policy module.
    """

    def __init__(self, decisions: Dict[str, Any]):
        self.q_u = decisions["q_u"]

    def penalty(self):
        return self.q_u


# ============================================================
# Unified Simulation Interface (Single Run)
# ============================================================

def simulate_unified_model(
    decisions: Dict[str, Any],
    uncertainties: Dict[str, Any],
    constants: Dict[str, Any]
) -> Dict[str, float]:
    """
    Unified interface function.

    Inputs
    ------
    decisions : dict
        {
            "wind_bids": {t: (b_wt, p_wt)},
            "q_u": imbalance penalty
        }

    uncertainties : dict
        Distributional parameters for bidding behavior
        (e.g., mu_pi, sigma_pi, mu_ps, sigma_ps).

    constants : dict
        Fixed market and technology parameters, INCLUDING
        mu_D, sigma_D, mu_G, sigma_G.

    Returns
    -------
    objectives : dict
        Realized values of all objective functions for one simulation.
    """

    env = MarketEnvironment(constants, uncertainties)
    wind = WindProducer(decisions)
    regulator = SystemRegulator(decisions)

    total_profit = 0.0
    total_imbalance = 0.0
    total_system_cost = 0.0
    total_wind_dispatch = 0.0

    for t in range(1, 25):
        demand = env.sample_demand(t)
        wind_gen = env.sample_wind_generation(t)

        bids = []
        bids.extend(env.sample_conventional_bids(t))
        bids.append(env.sample_solar_bid(t))
        bids.append(wind.submit_bid(t))

        price, dispatch = env.clear_market(bids, demand)

        # Wind producer profit
        profit = wind.profit(
            price,
            dispatch,
            wind_gen,
            regulator.penalty()
        )
        total_profit += profit

        # System-level quantities
        for agent, d in dispatch.items():
            if agent == "wind":
                imbalance = max(0.0, d - wind_gen)
            else:
                imbalance = 0.0  # conventional and solar assumed firm

            total_imbalance += imbalance
            total_system_cost += price * d + regulator.penalty() * imbalance

        total_wind_dispatch += dispatch.get("wind", 0.0)

    # ========================================================
    # Objective Values (Single Realization)
    # ========================================================

    objectives = {
        # Wind producer objective
        "wind_profit": total_profit,

        # Regulator objectives
        "system_imbalance": total_imbalance,
        "system_cost": total_system_cost,
        "wind_dispatch": total_wind_dispatch,
    }

    return objectives
