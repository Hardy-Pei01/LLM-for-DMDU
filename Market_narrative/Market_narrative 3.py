import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any


# ============================================================
# 1. Shared Environment
# ============================================================

@dataclass
class Environment:
    """Global market environment shared by all perspectives."""
    T: int = 24
    mu_D: float = 100.0
    sigma_D: float = 10.0
    mu_b: Dict[int, float] = field(default_factory=lambda: {1: 50, 2: 60, 3: 70})
    sigma_b: Dict[int, float] = field(default_factory=lambda: {1: 5, 2: 5, 3: 5})
    mu_p: Dict[int, float] = field(default_factory=lambda: {1: 40, 2: 45, 3: 50})
    sigma_p: Dict[int, float] = field(default_factory=lambda: {1: 3, 2: 3, 3: 3})
    G_bounds: Tuple[float, float] = (0, 100)
    c_under: float = 20.0
    c_over: float = 10.0
    emissions: Dict[int, float] = field(default_factory=lambda: {1: 0.8, 2: 0.9, 3: 1.0})
    delta: float = 0.05

    def sample_state(self) -> Dict[str, Any]:
        """Sample a random market state for one hour."""
        D = np.random.normal(self.mu_D, self.sigma_D)
        b = {i: np.random.normal(self.mu_b[i], self.sigma_b[i]) for i in range(1, 4)}
        p = {i: np.random.normal(self.mu_p[i], self.sigma_p[i]) for i in range(1, 4)}
        return {"D": D, "b": b, "p": p}


# ============================================================
# 2. System Operator (Market Clearing)
# ============================================================

class SystemOperator:
    """Implements the merit-order market clearing mechanism."""

    def __init__(self, env: Environment):
        self.env = env

    def clear_market(self, bids: Dict[int, float], prices: Dict[int, float], D: float):
        """Return clearing price and accepted quantities."""
        # Sort producers by offer price
        sorted_producers = sorted(prices.keys(), key=lambda i: prices[i])
        cumulative = 0
        P_t = 0
        q = {i: 0.0 for i in prices.keys()}

        for i in sorted_producers:
            if cumulative + bids[i] < D:
                q[i] = bids[i]
                cumulative += bids[i]
            else:
                q[i] = max(0, D - cumulative)
                P_t = prices[i]
                cumulative = D
                break

        if P_t == 0:  # all offers needed
            P_t = prices[sorted_producers[-1]]
        return P_t, q


# ============================================================
# 3. Renewable Producer
# ============================================================

class RenewableProducer:
    """Renewable producer's bidding and profit evaluation."""

    def __init__(self, env: Environment, tau: float = 0.0, sigma: float = 0.0):
        self.env = env
        self.tau = tau        # carbon tax (affects market indirectly)
        self.sigma = sigma    # renewable subsidy

    def profit(self, b0: float, p0: float, G: float, P: float, accepted: float) -> float:
        """Profit given generation, clearing price, and acceptance."""
        c_under, c_over = self.env.c_under, self.env.c_over
        under = max(0, b0 - G)
        over = max(0, G - b0)
        return (P + self.sigma) * accepted - c_under * under - c_over * over

    def expected_profit(self, market_states: List[Dict[str, Any]],
                        bids0: List[float], prices0: List[float],
                        so: SystemOperator) -> float:
        """Compute expected profit across stochastic market states."""
        profits = []
        for t, state in enumerate(market_states):
            D, b_conv, p_conv = state["D"], state["b"], state["p"]
            b0, p0 = bids0[t], prices0[t]

            # combine all producers
            all_bids = {0: b0, **b_conv}
            all_prices = {0: p0, **p_conv}
            P_t, q = so.clear_market(all_bids, all_prices, D)

            # sample renewable generation (robustly choose worst case)
            G_min, G_max = self.env.G_bounds
            # Worst case: generation opposite to accepted quantity direction
            worst_G = G_min if b0 > (G_min + G_max) / 2 else G_max
            profit = self.profit(b0, p0, worst_G, P_t, q[0])
            profits.append(profit)
        return float(np.mean(profits))


# ============================================================
# 4. Conventional Producers
# ============================================================

class ConventionalProducers:
    """Exogenous stochastic actors with evaluative profit metrics."""

    def __init__(self, env: Environment):
        self.env = env

    def expected_profit(self, market_states: List[Dict[str, Any]],
                        so: SystemOperator,
                        bids0: List[float], prices0: List[float]) -> Dict[int, float]:
        """Compute average profit for each conventional producer."""
        profits = {i: [] for i in [1, 2, 3]}
        for t, state in enumerate(market_states):
            D, b_conv, p_conv = state["D"], state["b"], state["p"]
            all_bids = {0: bids0[t], **b_conv}
            all_prices = {0: prices0[t], **p_conv}
            P_t, q = so.clear_market(all_bids, all_prices, D)
            for i in [1, 2, 3]:
                profits[i].append(P_t * q[i] - p_conv[i] * q[i])
        return {i: float(np.mean(profits[i])) for i in profits}


# ============================================================
# 5. Regulator
# ============================================================

class Regulator:
    """Tracks environmental and policy objectives."""

    def __init__(self, env: Environment):
        self.env = env

    def evaluate(self, market_states: List[Dict[str, Any]],
                 outputs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute emissions, renewable share, and cost objectives."""
        total_E, total_R, total_cost = 0.0, 0.0, 0.0
        for t, (state, out) in enumerate(zip(market_states, outputs)):
            D = state["D"]
            q = out["q"]
            P = out["P"]

            # emissions and renewable share
            E_t = sum(self.env.emissions[i] * q[i] for i in [1, 2, 3])
            R_t = q[0] / D if D > 0 else 0
            total_E += E_t
            total_R += R_t
            total_cost += P * D

        return {
            "emission_min": total_E,
            "renewable_max": total_R,
            "cost_min": total_cost
        }


# ============================================================
# 6. Unified Simulation Interface
# ============================================================

def simulate_market(controls: Dict[str, Any],
                    params: Dict[str, Any]) -> Dict[str, float]:
    """
    Simulate the unified market model.

    Parameters
    ----------
    controls : dict
        {
            'bids0': [b0_t for t in 1..24],
            'prices0': [p0_t for t in 1..24],
            'policy': {'tau': value, 'sigma': value}
        }
    params : dict
        Dictionary of environment parameter values.

    Returns
    -------
    results : dict
        Objective values for each perspective.
    """
    # Create environment
    env = Environment(**params)
    so = SystemOperator(env)

    # Initialize perspectives
    tau, sigma = controls.get("policy", {}).get("tau", 0.0), controls.get("policy", {}).get("sigma", 0.0)
    renewable = RenewableProducer(env, tau=tau, sigma=sigma)
    conventional = ConventionalProducers(env)
    regulator = Regulator(env)

    # Simulate stochastic market states
    market_states = [env.sample_state() for _ in range(env.T)]

    # --- Renewable objectives ---
    bids0 = controls["bids0"]
    prices0 = controls["prices0"]
    J_renewable = renewable.expected_profit(market_states, bids0, prices0, so)

    # --- Conventional metrics ---
    J_conventional = conventional.expected_profit(market_states, so, bids0, prices0)

    # --- Regulator evaluation ---
    outputs = []
    for t, state in enumerate(market_states):
        D, b_conv, p_conv = state["D"], state["b"], state["p"]
        all_bids = {0: bids0[t], **b_conv}
        all_prices = {0: prices0[t], **p_conv}
        P_t, q = so.clear_market(all_bids, all_prices, D)
        outputs.append({"P": P_t, "q": q})
    J_regulator = regulator.evaluate(market_states, outputs)

    # --- System Operator diagnostics ---
    balance_violation = np.mean([abs(s["D"] - sum(out["q"].values())) for s, out in zip(market_states, outputs)])
    avg_price = np.mean([out["P"] for out in outputs])

    # Return all objectives
    results = {
        "Renewable_expected_profit": J_renewable,
        "Conventional_expected_profits": J_conventional,
        "SystemOperator_balance_violation": balance_violation,
        "SystemOperator_avg_price": avg_price,
        "Regulator_emissions": J_regulator["emission_min"],
        "Regulator_renewable_share": J_regulator["renewable_max"],
        "Regulator_total_cost": J_regulator["cost_min"]
    }
    return results


# ============================================================
# Example Usage (not executed)
# ============================================================

# Example interface call (not to be run here):
# controls = {
#     "bids0": [50.0]*24,
#     "prices0": [40.0]*24,
#     "policy": {"tau": 10.0, "sigma": 5.0}
# }
# params = {"mu_D": 100.0, "sigma_D": 10.0}
# results = simulate_market(controls, params)
# print(results)
