import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ======================================================
# ============= 1. COMMON ENVIRONMENT ==================
# ======================================================

@dataclass
class MarketEnvironment:
    """Shared market environment for all stakeholders."""
    T: int = 24  # number of hourly intervals

    # --- Global parameters ---
    mu_D: float = 1000  # mean demand (MW)
    sigma_D: float = 100  # std dev of demand
    mu_b: List[float] = field(default_factory=lambda: [400, 350, 300])
    sigma_b: List[float] = field(default_factory=lambda: [50, 40, 30])
    mu_p: List[float] = field(default_factory=lambda: [45, 50, 60])
    sigma_p: List[float] = field(default_factory=lambda: [5, 5, 5])

    alpha: List[float] = field(default_factory=lambda: [10.0, 12.0, 15.0])
    beta: List[float] = field(default_factory=lambda: [0.05, 0.04, 0.03])
    gamma: List[float] = field(default_factory=lambda: [0.5, 0.4, 0.3])

    G_bounds: List[Tuple[float, float]] = field(
        default_factory=lambda: [(150.0, 250.0)] * 24
    )

    # --- Random sampling ---
    def sample_demand(self) -> np.ndarray:
        return np.random.normal(self.mu_D, self.sigma_D, self.T)

    def sample_conventional_bids(self) -> Tuple[np.ndarray, np.ndarray]:
        """Random bid quantities and prices for conventional producers."""
        N = len(self.mu_b)
        b = np.zeros((N, self.T))
        p = np.zeros((N, self.T))
        for i in range(N):
            b[i, :] = np.random.normal(self.mu_b[i], self.sigma_b[i], self.T)
            p[i, :] = np.random.normal(self.mu_p[i], self.sigma_p[i], self.T)
        return b, p


# ======================================================
# ============= 2. RENEWABLE PRODUCER ==================
# ======================================================

class RenewableProducer:
    """Renewable producer model with bid-based profit evaluation."""

    def __init__(self, env: MarketEnvironment):
        self.env = env

    def profit(self, b0, p0, piU, piO, D, b_conv, p_conv, G_real):
        """Compute total profit across all hours given bids and penalties."""
        T = self.env.T
        profits = np.zeros(T)

        for t in range(T):
            # Combine all bids
            b_all = np.append(b_conv[:, t], b0[t])
            p_all = np.append(p_conv[:, t], p0[t])

            # Merit-order sort
            order = np.argsort(p_all)
            b_sorted, p_sorted = b_all[order], p_all[order]

            # Determine clearing price and dispatch
            cum_supply = np.cumsum(b_sorted)
            idx = np.searchsorted(cum_supply, D[t])
            idx = np.clip(idx, 0, len(b_sorted) - 1)
            P_t = p_sorted[idx]
            q_sorted = np.zeros_like(b_sorted)
            q_sorted[:idx] = b_sorted[:idx]
            q_sorted[idx] = D[t] - np.sum(b_sorted[:idx])

            # Retrieve renewable dispatch
            renewable_index = np.where(order == len(order) - 1)[0][0]
            q0_t = q_sorted[renewable_index]

            # Compute penalties and profit
            G_t = np.clip(G_real[t], *self.env.G_bounds[t])
            s_minus = max(0, q0_t - G_t)
            s_plus = max(0, G_t - q0_t)
            profits[t] = P_t * q0_t - piU * s_minus - piO * s_plus
        # print(profits)
        return profits

    def evaluate_objectives(self, controls: Dict, uncertainty: Dict) -> Dict[str, float]:
        """Evaluate renewable producer objectives."""
        b0 = np.array(controls.get("b0", [200.0] * self.env.T))
        p0 = np.array(controls.get("p0", [50.0] * self.env.T))
        piU = controls["piU"]
        piO = controls["piO"]

        D = uncertainty["D"]
        b_conv = uncertainty["b_conv"]
        p_conv = uncertainty["p_conv"]
        G_real = uncertainty["G_real"]

        # Objective 1: profit
        profit_val = self.profit(b0, p0, piU, piO, D, b_conv, p_conv, G_real)

        # Objective 2: reliability (expected absolute deviation)
        reliability = float(np.mean(np.abs(G_real - b0)))

        return {"J1_profit": profit_val, "J2_reliability": reliability}


# ======================================================
# ============= 3. SYSTEM REGULATOR ====================
# ======================================================

class SystemRegulator:
    """Regulator model for evaluating cost, emissions, and price volatility."""

    def __init__(self, env: MarketEnvironment):
        self.env = env

    def market_clearing(self, b0, p0, D, b_conv, p_conv):
        """Return prices, dispatch quantities, and system emissions."""
        T = self.env.T
        N = len(self.env.mu_b)
        P = np.zeros(T)
        q_conv = np.zeros((N, T))
        q0 = np.zeros(T)

        for t in range(T):
            b_all = np.append(b_conv[:, t], b0[t])
            p_all = np.append(p_conv[:, t], p0[t])
            order = np.argsort(p_all)
            b_sorted, p_sorted = b_all[order], p_all[order]
            cum_supply = np.cumsum(b_sorted)
            idx = np.searchsorted(cum_supply, D[t])
            idx = np.clip(idx, 0, len(b_sorted) - 1)
            P[t] = p_sorted[idx]
            q_sorted = np.zeros_like(b_sorted)
            q_sorted[:idx] = b_sorted[:idx]
            q_sorted[idx] = D[t] - np.sum(b_sorted[:idx])

            q_all = np.zeros_like(q_sorted)
            q_all[order] = q_sorted
            q_conv[:, t] = q_all[:N]
            q0[t] = q_all[N]

        gamma = np.array(self.env.gamma)
        S = np.dot(gamma, q_conv)
        return P, q_conv, q0, S

    def evaluate_objectives(self, controls: Dict, uncertainty: Dict) -> Dict[str, float]:
        """Evaluate regulator objectives based on market outcomes."""
        piU = controls["piU"]
        piO = controls["piO"]
        E_cap = controls["E_cap"]

        b0 = controls["b0"]
        p0 = controls["p0"]
        D = uncertainty["D"]
        b_conv = uncertainty["b_conv"]
        p_conv = uncertainty["p_conv"]

        P, q_conv, q0, S = self.market_clearing(b0, p0, D, b_conv, p_conv)

        # System cost
        alpha, beta = np.array(self.env.alpha), np.array(self.env.beta)
        cost = np.sum(alpha[:, None] * q_conv + 0.5 * beta[:, None] * q_conv ** 2)

        # Emissions
        emissions = np.sum(S)
        emission_penalty = max(0, emissions - E_cap)

        # Price volatility
        price_volatility = float(np.var(P))

        return {
            "J1_system_cost": float(cost),
            "J2_emissions": float(emissions),
            "J3_price_volatility": price_volatility,
            "emission_cap_violation": emission_penalty,
        }


# ======================================================
# ============= 4. UNIFIED SIMULATION ==================
# ======================================================

def simulate_market(controls: Dict, uncertainty: Dict) -> Dict[str, float]:
    """
    Unified model interface.

    Parameters
    ----------
    controls : dict
        Control variables for both stakeholders:
            {
                "b0": [...], "p0": [...],   # renewable bids
                "piU": float, "piO": float, # penalty rates (regulator)
                "E_cap": float              # emissions cap
            }

    uncertainty : dict
        Stochastic / uncertain inputs:
            {
                "D": np.array,        # hourly demand
                "b_conv": np.ndarray, # competitor bid quantities
                "p_conv": np.ndarray, # competitor offer prices
                "G_real": np.array    # realized renewable generation
            }

    Returns
    -------
    dict
        Objective function values from both perspectives.
    """

    # Shared environment
    env = MarketEnvironment()

    # Instantiate agents
    renewable = RenewableProducer(env)
    regulator = SystemRegulator(env)

    # Evaluate renewable producer objectives
    renewable_objs = renewable.evaluate_objectives(controls, uncertainty)

    # Evaluate regulator objectives
    regulator_objs = regulator.evaluate_objectives(controls, uncertainty)

    # Combine and return results
    results = {**renewable_objs, **regulator_objs}
    return results


# ======================================================
# ========== EXAMPLE USAGE (not executed) ==============
# ======================================================
controls = {
    "b0": [220.0]*24,
    "p0": [50.0]*24,
    "piU": 100.0,
    "piO": 50.0,
    "E_cap": 10000.0,
}

# Generate uncertainty samples
env = MarketEnvironment()
D = env.sample_demand()
b_conv, p_conv = env.sample_conventional_bids()
G_real = np.random.uniform(150, 250, 24)

uncertainty = {
    "D": D,
    "b_conv": b_conv,
    "p_conv": p_conv,
    "G_real": G_real,
}