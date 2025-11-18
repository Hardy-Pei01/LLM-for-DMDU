import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List


# ============================================================
# 1. Common Environment Definition
# ============================================================

@dataclass
class MarketEnvironment:
    """Shared stochastic environment for all agents."""
    mu_D: float
    sigma_D: float
    mu_P: float
    sigma_P: float
    a: float
    b: float
    mu_pi: List[float]          # means of conventional producers' bid prices
    sigma_pi: List[float]       # stds of conventional producers' bid prices
    mu_ps: float                # mean of solar bid price
    sigma_ps: float             # std of solar bid price
    b_i: List[float]            # bid quantities for conventional producers
    hours: int = 24
    seed: int = 42

    # Internal state variables initialized post-construction
    D_t: np.ndarray = field(init=False)
    P_t: np.ndarray = field(init=False)
    b_st: np.ndarray = field(init=False)
    p_it: np.ndarray = field(init=False)
    p_st: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize stochastic processes for one day."""
        np.random.seed(self.seed)
        self.D_t = np.random.normal(self.mu_D, self.sigma_D, self.hours)
        self.P_t = np.random.normal(self.mu_P, self.sigma_P, self.hours)
        self.b_st = np.maximum(0, self.a + self.b * np.cos(2 * np.pi * np.arange(1, self.hours + 1) / 24))

        # Conventional producer prices: shape (3, hours)
        self.p_it = np.stack([
            np.random.normal(self.mu_pi[i], self.sigma_pi[i], self.hours)
            for i in range(3)
        ])
        # Solar prices
        self.p_st = np.random.normal(self.mu_ps, self.sigma_ps, self.hours)

    @staticmethod
    def market_clearing_price(bids: List[Tuple[float, float]], D: float) -> float:
        """Compute clearing price given all bids [(b_i, p_i)] and demand D."""
        sorted_bids = sorted(bids, key=lambda x: x[1])  # sort by price
        cumulative = 0.0
        for b, p in sorted_bids:
            cumulative += b
            if cumulative >= D:
                return p
        # If total supply < demand, return highest price
        return sorted_bids[-1][1]


# ============================================================
# 2. Renewable Producer Perspective
# ============================================================

@dataclass
class RenewableProducer:
    """Wind-power producer model."""
    q_u: float  # penalty for under-delivery
    q_o: float  # curtailment cost

    def revenue(self, c_t: float, b_t: float, p_t: float, P_t: float) -> float:
        """Compute realized revenue for the given hour."""
        if p_t > c_t:
            return 0.0
        if P_t < b_t:
            return c_t * b_t - self.q_u * (b_t - P_t)
        if P_t > b_t:
            return c_t * b_t - self.q_o * (P_t - b_t)
        return c_t * b_t


# ============================================================
# 3. Regulator Perspective
# ============================================================

@dataclass
class Regulator:
    """System regulator evaluating metrics."""
    q_u: float
    q_o: float

    def compute_metrics(
        self,
        D_t: float,
        c_t: float,
        bids_all: List[Tuple[str, float, float]],
        actual_gen: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute quantifiable system metrics."""
        # Determine accepted bids
        accepted = [(name, b, p) for (name, b, p) in bids_all if p <= c_t]
        total_supply = sum(b for _, b, _ in accepted)

        # Unserved demand
        delta_t = max(0.0, D_t - total_supply)

        # Curtailment (for renewables only)
        gamma_t = 0.0
        for (name, b, _) in accepted:
            if name in actual_gen:
                gamma_t += max(0.0, actual_gen[name] - b)

        # Renewable dispatch share
        renew_disp = sum(b for (name, b, _) in accepted if name in ["solar", "wind"])
        total_disp = sum(b for _, b, _ in accepted)
        eta_t = renew_disp / total_disp if total_disp > 0 else 0.0

        return {"delta_t2": delta_t**2, "gamma_t": gamma_t, "eta_t": eta_t}


# ============================================================
# 4. Unified Simulation Function
# ============================================================

def simulate_market(
    control_vars: Dict[str, Dict[str, float]],
    uncertain_params: Dict[str, float],
    seed: int = 42
) -> Dict[str, float]:
    """
    Unified interface for simulating the 24-hour market.

    Parameters
    ----------
    control_vars : dict
        {
            "producer": {"bids": [b_t], "prices": [p_t]},
            "regulator": {"q_u": val, "q_o": val}
        }
    uncertain_params : dict
        Model parameters (means, variances, constants).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        {
            "producer_profit": float,
            "reg_reliability": float,
            "reg_curtailment": float,
            "reg_renew_share": float
        }
    """
    np.random.seed(seed)

    # --- Initialize environment ---
    env = MarketEnvironment(
        mu_D=uncertain_params["mu_D"],
        sigma_D=uncertain_params["sigma_D"],
        mu_P=uncertain_params["mu_P"],
        sigma_P=uncertain_params["sigma_P"],
        a=uncertain_params["a"],
        b=uncertain_params["b"],
        mu_pi=uncertain_params["mu_pi"],
        sigma_pi=uncertain_params["sigma_pi"],
        mu_ps=uncertain_params["mu_ps"],
        sigma_ps=uncertain_params["sigma_ps"],
        b_i=uncertain_params["b_i"],
        seed=seed
    )

    # --- Initialize agents ---
    q_u = control_vars["regulator"]["q_u"]
    q_o = control_vars["regulator"]["q_o"]
    producer = RenewableProducer(q_u=q_u, q_o=q_o)
    regulator = Regulator(q_u=q_u, q_o=q_o)

    # --- Accumulators ---
    producer_profit = 0.0
    reg_reliability = 0.0
    reg_curtailment = 0.0
    reg_renew_share = 0.0

    # --- Hourly simulation loop ---
    for t in range(env.hours):
        D_t = env.D_t[t]
        P_t = env.P_t[t]
        b_st = env.b_st[t]
        p_st = env.p_st[t]
        b_t = control_vars["producer"]["bids"][t]
        p_t = control_vars["producer"]["prices"][t]

        # Conventional producers (3 total)
        bids_conv = [(f"conv{i+1}", env.b_i[i], env.p_it[i, t]) for i in range(3)]
        bids_ren = [("solar", b_st, p_st), ("wind", b_t, p_t)]
        all_bids = bids_conv + bids_ren

        # Market clearing (use only (b, p))
        clearing_price = env.market_clearing_price([(b, p) for _, b, p in all_bids], D_t)

        # --- Producer Revenue ---
        revenue_t = producer.revenue(clearing_price, b_t, p_t, P_t)
        producer_profit += revenue_t

        # --- Regulator Metrics ---
        actual_gen = {"solar": b_st, "wind": P_t}
        metrics = regulator.compute_metrics(D_t, clearing_price, all_bids, actual_gen)

        reg_reliability += metrics["delta_t2"]
        reg_curtailment += metrics["gamma_t"]
        reg_renew_share += metrics["eta_t"]

    return {
        "producer_profit": producer_profit,
        "reg_reliability": reg_reliability,
        "reg_curtailment": reg_curtailment,
        "reg_renew_share": reg_renew_share,
    }
