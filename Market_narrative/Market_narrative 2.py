import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ===============================================================
#   Common Environment
# ===============================================================

@dataclass
class MarketEnvironment:
    """Shared environment containing stochastic parameters and distributions."""
    T: int = 24
    num_conventional: int = 3

    # Demand distribution
    mu_D: float = 1000.0
    sigma_D: float = 100.0

    # Conventional producer bid distributions
    mu_b: List[float] = field(default_factory=lambda: [300, 400, 500])
    sigma_b: List[float] = field(default_factory=lambda: [30, 40, 50])
    mu_p: List[float] = field(default_factory=lambda: [45, 50, 55])
    sigma_p: List[float] = field(default_factory=lambda: [5, 5, 5])

    # Renewable generation bounds
    G_bounds: Tuple[float, float] = (0.0, 800.0)

    def sample_uncertainties(self, uncertain_params: Dict) -> Dict:
        """Generate stochastic realizations or use provided ones."""
        D = uncertain_params.get(
            "D", np.random.normal(self.mu_D, self.sigma_D, self.T)
        )
        G = uncertain_params.get(
            "G", np.random.uniform(self.G_bounds[0], self.G_bounds[1], self.T)
        )

        b_conv, p_conv = [], []
        for i in range(self.num_conventional):
            b_i = uncertain_params.get(
                f"b_{i+1}", np.random.normal(self.mu_b[i], self.sigma_b[i], self.T)
            )
            p_i = uncertain_params.get(
                f"p_{i+1}", np.random.normal(self.mu_p[i], self.sigma_p[i], self.T)
            )
            b_conv.append(b_i)
            p_conv.append(p_i)

        return {"D": D, "G": G, "b_conv": b_conv, "p_conv": p_conv}


# ===============================================================
#   Renewable Producer
# ===============================================================

@dataclass
class RenewableProducer:
    """Strategic renewable producer controlling its hourly bids."""
    c_under: float
    c_over: float

    def profit_hour(self, b_t: float, p_t: float, pi_t: float, G_t: float) -> float:
        x_t = 1 if p_t <= pi_t else 0
        shortfall = max(b_t - G_t, 0)
        surplus = max(G_t - b_t, 0)
        penalty = self.c_under * shortfall + self.c_over * surplus
        return x_t * (pi_t * b_t - penalty)


# ===============================================================
#   Conventional Producers
# ===============================================================

@dataclass
class ConventionalProducer:
    """Non-strategic conventional producer with given cost coefficient."""
    id: int
    cost_coef: float = 10.0

    def profit_hour(self, b_i: float, p_i: float, pi_t: float) -> float:
        x_i = 1 if p_i <= pi_t else 0
        cost = self.cost_coef * b_i
        return x_i * (pi_t * b_i - cost)


# ===============================================================
#   Market Operator
# ===============================================================

@dataclass
class MarketOperator:
    """Determines clearing price and acceptances via merit order."""
    def clear_hour(self, renewable_bid: Tuple[float, float],
                   conv_bids: List[Tuple[float, float]], D_t: float):
        """Perform clearing for one hour."""
        bids = [(renewable_bid[1], renewable_bid[0], 0)]  # (price, qty, id=0)
        for i, (b_i, p_i) in enumerate(conv_bids, start=1):
            bids.append((p_i, b_i, i))

        bids.sort(key=lambda x: x[0])  # ascending price
        cumulative, pi_t = 0.0, bids[-1][0]
        accepted = {idx: 0 for _, _, idx in bids}

        for price, qty, idx in bids:
            if cumulative + qty < D_t:
                accepted[idx] = 1
                cumulative += qty
            else:
                accepted[idx] = 1
                pi_t = price
                break

        return pi_t, accepted


# ===============================================================
#   Regulator
# ===============================================================

@dataclass
class Regulator:
    """Evaluates system-level policy objectives."""
    def evaluate_hour(self, D_t: float, pi_t: float, bids: Dict[int, Tuple[float, float]],
                      accepted: Dict[int, int]) -> Dict[str, float]:
        """Compute regulatory metrics for one hour."""
        renewable_supply = accepted[0] * bids[0][0]
        total_supply = sum(accepted[i] * bids[i][0] for i in accepted)

        reliability = -abs(D_t - total_supply)
        welfare = (D_t * 100 - pi_t * D_t)  # simplified linear consumer utility
        renewable_share = renewable_supply / total_supply if total_supply > 0 else 0.0
        price_stability = -pi_t ** 2

        return dict(
            reliability=reliability,
            welfare=welfare,
            renewable_share=renewable_share,
            price_stability=price_stability,
        )


# ===============================================================
#   Unified Simulation Interface
# ===============================================================

def simulate_energy_market(
    control_vars: Dict,
    uncertain_params: Dict,
) -> Dict:
    """
    Unified multi-period market simulation.
    control_vars: e.g., {'b': [600]*24, 'p': [40]*24, 'c_under': 20, 'c_over': 10}
    uncertain_params: may contain arrays of stochastic realizations for D, G, b_i, p_i
    Returns a dict with all perspective objectives aggregated over 24 hours.
    """

    # ----- Setup -----
    env = MarketEnvironment()
    samples = env.sample_uncertainties(uncertain_params)
    operator = MarketOperator()
    regulator = Regulator()

    b_vec = control_vars.get("b", np.full(env.T, 500))
    p_vec = control_vars.get("p", np.full(env.T, 40))
    c_under = control_vars.get("c_under", 20)
    c_over = control_vars.get("c_over", 10)
    renewable = RenewableProducer(c_under=c_under, c_over=c_over)
    conventional = [ConventionalProducer(id=i) for i in range(1, env.num_conventional + 1)]

    # ----- Storage for results -----
    renewable_profit, conv_profit = [], {f"conv_{i.id}": [] for i in conventional}
    regulator_metrics = {"reliability": [], "welfare": [], "renewable_share": [], "price_stability": []}
    clearing_prices = []

    # ----- Hourly simulation -----
    for t in range(env.T):
        D_t, G_t = samples["D"][t], samples["G"][t]
        conv_bids = [(samples["b_conv"][i][t], samples["p_conv"][i][t]) for i in range(env.num_conventional)]

        # Market clearing
        pi_t, accepted = operator.clear_hour((b_vec[t], p_vec[t]), conv_bids, D_t)
        clearing_prices.append(pi_t)

        # Renewable profit
        renewable_profit.append(renewable.profit_hour(b_vec[t], p_vec[t], pi_t, G_t))

        # Conventional profits
        for i, c in enumerate(conventional, start=1):
            pr = c.profit_hour(samples["b_conv"][i-1][t], samples["p_conv"][i-1][t], pi_t)
            conv_profit[f"conv_{c.id}"].append(pr)

        # Regulator evaluation
        bids = {0: (b_vec[t], p_vec[t])}
        for i in range(1, env.num_conventional + 1):
            bids[i] = (samples["b_conv"][i-1][t], samples["p_conv"][i-1][t])

        reg = regulator.evaluate_hour(D_t, pi_t, bids, accepted)
        for k in regulator_metrics:
            regulator_metrics[k].append(reg[k])

    # ----- Aggregate results -----
    print(renewable_profit)
    avg = lambda x: float(np.mean(x))
    renewable_obj = {"expected_profit": avg(renewable_profit)}
    conventional_obj = {k: {"expected_profit": avg(v), "profit_variance": float(np.var(v))} for k, v in conv_profit.items()}
    market_obj = {"avg_clearing_price": avg(clearing_prices), "price_volatility": float(np.var(clearing_prices))}
    regulator_obj = {k: avg(v) for k, v in regulator_metrics.items()}

    # ----- Output -----
    return {
        "RenewableProducer": renewable_obj,
        "ConventionalProducers": conventional_obj,
        "MarketOperator": market_obj,
        "Regulator": regulator_obj,
    }


controls = {
    "b": [550 + 10*np.sin(t) for t in range(24)],  # renewable bid quantities
    "p": [42]*24,                                  # offer prices
    "c_under": 25, "c_over": 10
}

uncertainties = {
    "D": np.random.normal(1000, 80, 24),           # stochastic demand
    "G": np.random.uniform(400, 700, 24)           # renewable generation realizations
}

results = simulate_energy_market(controls, uncertainties)
print(results)
