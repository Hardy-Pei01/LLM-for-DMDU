import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

# ============================================================
# 1. Shared Environment
# ============================================================

@dataclass
class MarketEnvironment:
    """Shared stochastic market environment."""
    T: int = 24

    # === Fixed structural constants ===
    mu_D: float = 1000       # mean demand
    sigma_D: float = 100     # std dev of demand
    mu_bi: List[float] = field(default_factory=lambda: [400, 350, 300])
    sigma_bi: List[float] = field(default_factory=lambda: [50, 40, 30])
    mu_pi: List[float] = field(default_factory=lambda: [45, 50, 60])
    sigma_pi: List[float] = field(default_factory=lambda: [5, 5, 5])

    # === Uncertain renewable generation bounds ===
    P_bounds: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 500.0)] * 24)

    # ----------------------------------------------------
    # Stochastic sampling methods
    # ----------------------------------------------------
    def sample_demand(self) -> float:
        """Sample hourly electricity demand."""
        return np.random.normal(self.mu_D, self.sigma_D)

    def sample_conventional_bids(self) -> List[Tuple[float, float]]:
        """Sample bids (quantity, price) for conventional producers."""
        bids = []
        for i in range(3):
            b_i = np.random.normal(self.mu_bi[i], self.sigma_bi[i])
            p_i = np.random.normal(self.mu_pi[i], self.sigma_pi[i])
            bids.append((b_i, p_i))
        return bids

    def sample_renewable_output(self, t: int) -> float:
        """Sample renewable generation within bounds for hour t."""
        low, high = self.P_bounds[t - 1]
        return np.random.uniform(low, high)

    @staticmethod
    def market_clearing_price(D_t: float, bids: List[Tuple[float, float]]) -> float:
        """
        Determine clearing price: sort bids by price ascending,
        find smallest price such that cumulative quantity >= demand.
        """
        sorted_bids = sorted(bids, key=lambda x: x[1])
        cum_quantity = 0.0
        for b, p in sorted_bids:
            cum_quantity += b
            if cum_quantity >= D_t:
                return p
        return sorted_bids[-1][1]  # if not enough supply, last price


# ============================================================
# 2. Renewable Producer
# ============================================================

@dataclass
class RenewableProducer:
    """Renewable producer perspective."""
    b_max: float = 500.0
    p_min: float = 0.0
    p_max: float = 200.0

    def profit(self, b_t: float, p_t: float, c_t: float, P_t: float,
               q_u: float, q_o: float) -> float:
        """Compute realized profit for one hour."""
        if p_t > c_t:
            return 0.0
        shortfall = max(b_t - P_t, 0.0)
        surplus = max(P_t - b_t, 0.0)
        return c_t * b_t - q_u * shortfall - q_o * surplus

    def evaluate_objectives(
        self,
        env: MarketEnvironment,
        b_seq: List[float],
        p_seq: List[float],
        q_u: float,
        q_o: float
    ) -> Dict[str, float]:
        """Compute expected and worst-case profit objectives."""
        T = env.T
        profits = []
        for t in range(1, T + 1):
            D_t = env.sample_demand()
            conv_bids = env.sample_conventional_bids()
            P_t = env.sample_renewable_output(t)
            bids = conv_bids + [(b_seq[t - 1], p_seq[t - 1])]
            c_t = env.market_clearing_price(D_t, bids)
            pi_t = self.profit(b_seq[t - 1], p_seq[t - 1], c_t, P_t, q_u, q_o)
            profits.append(pi_t)

        expected_profit = float(np.mean(profits))
        worst_case_profit = float(np.min(profits))
        return {
            "ExpectedProfit": expected_profit,
            "WorstCaseProfit": profits
        }


# ============================================================
# 3. System Regulator
# ============================================================

@dataclass
class SystemRegulator:
    """System regulator perspective."""

    def compute_metrics(
        self,
        env: MarketEnvironment,
        b_seq: List[float],
        p_seq: List[float],
        q_u: float,
        q_o: float
    ) -> Dict[str, float]:
        """Compute quantifiable system-level objectives."""
        T = env.T
        imbalance_sq = []
        penalty_sum = []
        accepted_renewable = []

        for t in range(1, T + 1):
            D_t = env.sample_demand()
            conv_bids = env.sample_conventional_bids()
            P_t = env.sample_renewable_output(t)
            bids = conv_bids + [(b_seq[t - 1], p_seq[t - 1])]
            c_t = env.market_clearing_price(D_t, bids)

            # accepted quantities
            b_acc = b_seq[t - 1] if p_seq[t - 1] <= c_t else 0.0
            b_conv_acc = sum(b for b, p in conv_bids if p <= c_t)
            R_t = D_t - (P_t + b_acc + b_conv_acc)
            Psi_t = q_u * max(b_seq[t - 1] - P_t, 0.0) + q_o * max(P_t - b_seq[t - 1], 0.0)
            E_t_ren = P_t if p_seq[t - 1] <= c_t else 0.0

            imbalance_sq.append(R_t ** 2)
            penalty_sum.append(Psi_t)
            accepted_renewable.append(E_t_ren)

        # Quantifiable objectives
        J1 = -float(np.mean(imbalance_sq))       # reliability (minimize imbalance)
        J2 = -float(np.mean(penalty_sum))        # efficiency (minimize penalties)
        J3 = float(np.mean(accepted_renewable))  # renewable integration
        return {"Reliability": J1, "Efficiency": J2, "RenewableShare": J3}


# ============================================================
# 4. Unified Interface Function
# ============================================================

def simulate_market(
    controls: Dict[str, any],
    uncertainties: Dict[str, any]
) -> Dict[str, Dict[str, float]]:
    """
    Unified simulation interface.

    Parameters
    ----------
    controls : dict
        {
          "renewable_b": [b1,...,b24],  # renewable bid quantities
          "renewable_p": [p1,...,p24],  # renewable bid prices
          "q_u": float,                 # under-delivery penalty
          "q_o": float                  # over-delivery penalty
        }
    uncertainties : dict
        {
          "P_bounds": [(low_t, high_t), ...]   # optional renewable output bounds
        }

    Returns
    -------
    dict
        {
          "Renewable": {"ExpectedProfit": ..., "WorstCaseProfit": ...},
          "Regulator": {"Reliability": ..., "Efficiency": ..., "RenewableShare": ...}
        }
    """

    # --- Initialize environment and agents ---
    P_bounds = uncertainties.get("P_bounds", [(0.0, 500.0)] * 24)
    env = MarketEnvironment(P_bounds=P_bounds)
    renewable = RenewableProducer()
    regulator = SystemRegulator()

    b_seq = controls["renewable_b"]
    p_seq = controls["renewable_p"]
    q_u = controls["q_u"]
    q_o = controls["q_o"]

    # --- Evaluate objectives ---
    renewable_results = renewable.evaluate_objectives(env, b_seq, p_seq, q_u, q_o)
    regulator_results = regulator.compute_metrics(env, b_seq, p_seq, q_u, q_o)

    return {
        "Renewable": renewable_results,
        "Regulator": regulator_results
    }

# ============================================================
# End of unified model module
# ============================================================

controls={
    "renewable_b": [220]*24,
    "renewable_p": [50]*24,
    "q_u": 100,
    "q_o": 50
}

t_lis = [np.random.uniform(150, 250) for i in range(24)]
uncertainties = {
    "P_bounds": [(t_lis[i], t_lis[i]) for i in range(24)]
}