import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List


# ============================================================
# 1. Shared Environment
# ============================================================

@dataclass
class MarketEnvironment:
    """Shared market environment with fixed stochastic parameters."""
    T: int = 24
    mu_D: float = 1000.0
    sigma_D: float = 100.0
    mu_b: List[float] = field(default_factory=lambda: [400, 350, 300])
    sigma_b: List[float] = field(default_factory=lambda: [50, 40, 30])
    mu_p: List[float] = field(default_factory=lambda: [40, 45, 50])
    sigma_p: List[float] = field(default_factory=lambda: [5, 5, 5])
    g_bounds: List[Tuple[float, float]] = field(default_factory=lambda: [(150, 250)] * 24)

    D_t: np.ndarray = field(init=False)
    bids_conv: List[Tuple[np.ndarray, np.ndarray]] = field(init=False)
    g_t: np.ndarray = field(init=False)

    def sample_uncertainties(self):
        """Sample stochastic demand and competitor bids."""
        self.D_t = np.random.normal(self.mu_D, self.sigma_D, self.T)
        self.bids_conv = []
        for i in range(3):
            b_i = np.random.normal(self.mu_b[i], self.sigma_b[i], self.T)
            p_i = np.random.normal(self.mu_p[i], self.sigma_p[i], self.T)
            self.bids_conv.append((b_i, p_i))

    def sample_generation(self, mode: str = "midpoint"):
        """Select generation scenario."""
        low, high = np.array([gb[0] for gb in self.g_bounds]), np.array([gb[1] for gb in self.g_bounds])
        if mode == "worst_case":
            self.g_t = low
        elif mode == "best_case":
            self.g_t = high
        else:
            self.g_t = 0.5 * (low + high)

    def clear_market(self, b_Rt: np.ndarray, p_Rt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute market clearing price, acceptance, and supply."""
        pi_t, delta_t, Q_t = np.zeros(self.T), np.zeros(self.T), np.zeros(self.T)
        for t in range(self.T):
            bids = []
            for i in range(3):
                b_i, p_i = self.bids_conv[i][0][t], self.bids_conv[i][1][t]
                bids.append((p_i, b_i))
            bids.append((p_Rt[t], b_Rt[t]))
            bids.sort(key=lambda x: x[0])

            demand = self.D_t[t]
            cumulative = 0.0
            clearing_price = bids[-1][0]
            accepted_supply = 0.0
            for price, qty in bids:
                cumulative += qty
                accepted_supply += qty
                if cumulative >= demand:
                    clearing_price = price
                    break

            pi_t[t] = clearing_price
            delta_t[t] = 1.0 if p_Rt[t] <= pi_t[t] else 0.0
            Q_t[t] = accepted_supply
        return pi_t, delta_t, Q_t


# ============================================================
# 2. Renewable Producer Module
# ============================================================

@dataclass
class RenewableProducer:
    """Renewable producer perspective."""
    env: MarketEnvironment
    b_Rt: np.ndarray
    p_Rt: np.ndarray
    c_under: np.ndarray
    c_over: np.ndarray
    b_max: float = 500.0

    def compute_profit(self, pi_t, delta_t, g_t) -> np.ndarray:
        """Compute hourly profit for given generation."""
        under_penalty = self.c_under * np.maximum(self.b_Rt - g_t, 0)
        over_penalty = self.c_over * np.maximum(g_t - self.b_Rt, 0)
        profit = delta_t * (pi_t * self.b_Rt - under_penalty - over_penalty)
        return profit

    def objective_expected_profit(self, pi_t, delta_t, g_t) -> float:
        """Expected profit for realized generation."""
        profit = self.compute_profit(pi_t, delta_t, g_t)
        return np.sum(profit)

    def objective_worst_case_profit(self, pi_t, delta_t) -> float:
        """
        Worst-case profit under deep uncertainty in g_t.

        For each hour, the profit is piecewise linear in g_t with slope change at g_t = b_Rt.
        The minimum therefore occurs at one of {lower bound, upper bound, b_Rt (if within bounds)}.
        """
        profit_worst = np.zeros(self.env.T)
        for t in range(self.env.T):
            g_candidates = []
            low, high = self.env.g_bounds[t]
            # candidate points
            g_candidates.append(low)
            g_candidates.append(high)
            if low <= self.b_Rt[t] <= high:
                g_candidates.append(self.b_Rt[t])
            # evaluate profits at candidate g values
            profits = []
            for g_val in g_candidates:
                under_penalty = self.c_under[t] * max(self.b_Rt[t] - g_val, 0)
                over_penalty = self.c_over[t] * max(g_val - self.b_Rt[t], 0)
                p = delta_t[t] * (pi_t[t] * self.b_Rt[t] - under_penalty - over_penalty)
                profits.append(p)
            profit_worst[t] = min(profits)
        return np.sum(profit_worst)


# ============================================================
# 3. System Regulator Module
# ============================================================

@dataclass
class SystemRegulator:
    """System regulator perspective."""
    env: MarketEnvironment
    c_under: np.ndarray
    c_over: np.ndarray
    L_t: np.ndarray
    c_max: float = 200.0

    def evaluate_metrics(self, pi_t, Q_t):
        """Compute measurable regulator objectives."""
        J1 = -np.var(pi_t)                         # price stability
        J2 = -np.mean(np.abs(Q_t - self.env.D_t))  # reliability
        J3 = -np.mean(pi_t * self.env.D_t)         # cost containment
        return J1, J2, J3


# ============================================================
# 4. Unified Model Interface
# ============================================================

def evaluate_unified_model(
    controls: Dict[str, np.ndarray],
    uncertainties: Dict[str, List[Tuple[float, float]]]
) -> Dict[str, float]:
    """
    Unified model interface.

    Inputs:
        controls:
            {
              'b_Rt': np.array(24),
              'p_Rt': np.array(24),
              'c_under': np.array(24),
              'c_over': np.array(24)
            }
        uncertainties:
            {
              'g_bounds': list of (low, high)
            }

    Output:
        dict of objective function values:
            {
              'J1_R': expected_profit,
              'J2_R': worst_case_profit,
              'J1_Reg': price_stability,
              'J2_Reg': reliability,
              'J3_Reg': cost_containment
            }
    """

    # ---- Initialize shared environment ----
    env = MarketEnvironment(g_bounds=uncertainties['g_bounds'])
    env.sample_uncertainties()
    env.sample_generation(mode="midpoint")

    # ---- Instantiate modules ----
    regulator = SystemRegulator(env, controls['c_under'], controls['c_over'], L_t=np.ones(24) * 10.0)
    producer = RenewableProducer(env, controls['b_Rt'], controls['p_Rt'],
                                 controls['c_under'], controls['c_over'])

    # ---- Market simulation ----
    pi_t, delta_t, Q_t = env.clear_market(producer.b_Rt, producer.p_Rt)

    # ---- Renewable objectives ----
    J1_R = producer.objective_expected_profit(pi_t, delta_t, env.g_t)
    J2_R = producer.objective_worst_case_profit(pi_t, delta_t)

    # ---- Regulator objectives ----
    J1_Reg, J2_Reg, J3_Reg = regulator.evaluate_metrics(pi_t, Q_t)

    return {
        'J1_R': float(J1_R),
        'J2_R': float(J2_R),
        'J1_Reg': float(J1_Reg),
        'J2_Reg': float(J2_Reg),
        'J3_Reg': float(J3_Reg)
    }


controls = {
    'b_Rt': np.array([220]*24),       # renewable bid quantities (24 values)
    'p_Rt': np.array([50]*24),       # renewable offer prices (24 values)
    'c_under': np.array([100]*24),    # regulator under-delivery penalties
    'c_over': np.array([50]*24)      # regulator over-delivery penalties
}

uncertainties = {
    'g_bounds': [(150, 250) for t in range(24)]  # renewable generation bounds
}

evaluate_unified_model(controls, uncertainties)