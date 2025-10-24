import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple


# ==============================================================
# 1. COMMON ENVIRONMENT
# ==============================================================

@dataclass
class MarketEnvironment:
    """Shared environment with fixed constants and stochastic sampling."""
    # --- Fixed constants ---
    mu_D: float = 1000
    sigma_D: float = 100
    mu_bi: List[float] = None
    sigma_bi: List[float] = None
    mu_pi: List[float] = None
    sigma_pi: List[float] = None
    rho_target: float = 0.3
    hours: int = 24

    # --- Uncertain renewable generation bounds ---
    P_lower: List[float] = None
    P_upper: List[float] = None

    def __post_init__(self):
        if self.mu_bi is None:
            self.mu_bi = [400, 350, 300]
        if self.sigma_bi is None:
            self.sigma_bi = [50, 40, 30]
        if self.mu_pi is None:
            self.mu_pi = [45, 50, 60]
        if self.sigma_pi is None:
            self.sigma_pi = [5, 5, 5]

    # --- Sampling methods for stochastic variables ---
    def sample_demand(self) -> np.ndarray:
        """Sample hourly electricity demand."""
        return np.random.normal(self.mu_D, self.sigma_D, self.hours)

    def sample_conventional_bids(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample bids (quantities, prices) for conventional producers."""
        n_conv = len(self.mu_bi)
        b = np.zeros((n_conv, self.hours))
        p = np.zeros((n_conv, self.hours))
        for i in range(n_conv):
            b[i, :] = np.random.normal(self.mu_bi[i], self.sigma_bi[i], self.hours)
            p[i, :] = np.random.normal(self.mu_pi[i], self.sigma_pi[i], self.hours)
        return b, p


# ==============================================================
# 2. MARKET CLEARING (MERIT ORDER)
# ==============================================================

def merit_order_clearing(bids: List[Tuple[float, float]], demand: float) -> Tuple[float, List[int]]:
    """
    Compute market clearing price and acceptance indicators using merit order.
    :param bids: list of (quantity, price)
    :param demand: total demand
    :return: (clearing_price, acceptance_indicators)
    """
    sorted_indices = np.argsort([p for (_, p) in bids])
    cumulative = 0.0
    accepted = [0] * len(bids)
    clearing_price = 0.0

    for idx in sorted_indices:
        b, p = bids[idx]
        cumulative += b
        accepted[idx] = 1
        if cumulative >= demand:
            clearing_price = p
            break

    if clearing_price == 0.0:
        clearing_price = bids[sorted_indices[-1]][1]

    return clearing_price, accepted


# ==============================================================
# 3. RENEWABLE PRODUCER MODULE
# ==============================================================

@dataclass
class RenewableProducer:
    """Renewable producer's decision and objectives."""
    env: MarketEnvironment
    b_t: List[float]
    p_t: List[float]
    q_u: float
    q_o: float

    def compute_profit(self, C_t, x_t, P_t, b_t):
        """Compute realized profit for one hour."""
        if x_t == 0:
            return 0.0
        under = max(b_t - P_t, 0)
        over = max(P_t - b_t, 0)
        return C_t * b_t - self.q_u * under - self.q_o * over

    def expected_profit(self, D_t, conv_bids_b, conv_bids_p) -> float:
        """Expected profit over market and generation uncertainty."""
        profits = []
        for t in range(self.env.hours):
            bids = [(self.b_t[t], self.p_t[t])]
            for i in range(len(conv_bids_b)):
                bids.append((conv_bids_b[i, t], conv_bids_p[i, t]))
            C_t, x_vec = merit_order_clearing(bids, D_t[t])
            x_r = x_vec[0]
            P_t = np.random.uniform(self.env.P_lower[t], self.env.P_upper[t])
            profits.append(self.compute_profit(C_t, x_r, P_t, self.b_t[t]))
        return profits

    def worst_case_penalty(self, D_t, conv_bids_b, conv_bids_p) -> float:
        """Compute worst-case penalty across renewable generation uncertainty."""
        total_penalty = 0.0
        for t in range(self.env.hours):
            bids = [(self.b_t[t], self.p_t[t])]
            for i in range(len(conv_bids_b)):
                bids.append((conv_bids_b[i, t], conv_bids_p[i, t]))
            C_t, x_vec = merit_order_clearing(bids, D_t[t])
            if x_vec[0] == 1:
                # Penalties for extreme generation values
                penalties = []
                for P_t in [self.env.P_lower[t], self.env.P_upper[t]]:
                    under = max(self.b_t[t] - P_t, 0)
                    over = max(P_t - self.b_t[t], 0)
                    penalties.append(self.q_u * under + self.q_o * over)
                total_penalty += max(penalties)
        return total_penalty


# ==============================================================
# 4. REGULATOR MODULE
# ==============================================================

@dataclass
class Regulator:
    """System regulator perspective and objectives."""
    env: MarketEnvironment
    q_u: float
    q_o: float

    def compute_system_metrics(
        self, D_t, conv_bids_b, conv_bids_p, ren_b_t, ren_p_t
    ) -> Dict[str, float]:
        """Compute regulator's measurable objectives."""
        prices = np.zeros(self.env.hours)
        imbalances = np.zeros(self.env.hours)
        renewable_shares = np.zeros(self.env.hours)

        for t in range(self.env.hours):
            bids = [(ren_b_t[t], ren_p_t[t])]
            for i in range(len(conv_bids_b)):
                bids.append((conv_bids_b[i, t], conv_bids_p[i, t]))
            C_t, x_vec = merit_order_clearing(bids, D_t[t])
            prices[t] = C_t
            x_r = x_vec[0]
            P_t = np.random.uniform(self.env.P_lower[t], self.env.P_upper[t])

            # Delivered supply
            delivered = 0.0
            if x_r == 1:
                delivered += min(ren_b_t[t], P_t)
            for i in range(1, len(bids)):
                if x_vec[i] == 1:
                    delivered += bids[i][0]

            imbalances[t] = abs(D_t[t] - delivered)
            renewable_shares[t] = x_r * ren_b_t[t] / max(D_t[t], 1e-6)

        # Quantifiable objectives
        J1_price_stability = -np.var(prices)
        J2_reliability = -np.mean(imbalances)
        J3_renewable_integration = -np.sum(np.abs(renewable_shares - self.env.rho_target))

        return {
            "price_stability": J1_price_stability,
            "reliability": J2_reliability,
            "renewable_integration": J3_renewable_integration,
        }


# ==============================================================
# 5. UNIFIED SIMULATION INTERFACE
# ==============================================================

def simulate_market(controls: Dict[str, List[float]], uncertainty: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Unified interface for simulating the unified electricity market model.
    --------------------------------------------------------------------
    Inputs:
        controls: dict of control variables
            {
                "b_t": list of renewable bid quantities (len=24),
                "p_t": list of renewable bid prices (len=24),
                "q_u": underdelivery penalty (float),
                "q_o": overdelivery penalty (float)
            }

        uncertainty: dict of uncertain renewable generation bounds
            {
                "P_lower": [...],
                "P_upper": [...]
            }

    Output:
        Dict containing objective values for both perspectives.
    """

    # === Construct environment (constants embedded, uncertain inputs provided) ===
    env = MarketEnvironment(
        P_lower=uncertainty["P_lower"],
        P_upper=uncertainty["P_upper"],
    )

    # === Sample endogenous uncertainties ===
    D_t = env.sample_demand()
    conv_bids_b, conv_bids_p = env.sample_conventional_bids()

    # === Initialize agents ===
    renewable = RenewableProducer(
        env=env,
        b_t=controls["b_t"],
        p_t=controls["p_t"],
        q_u=controls["q_u"],
        q_o=controls["q_o"],
    )

    regulator = Regulator(env=env, q_u=controls["q_u"], q_o=controls["q_o"])

    # === Compute objectives ===
    J1_r = renewable.expected_profit(D_t, conv_bids_b, conv_bids_p)
    J2_r = -renewable.worst_case_penalty(D_t, conv_bids_b, conv_bids_p)
    regulator_metrics = regulator.compute_system_metrics(
        D_t, conv_bids_b, conv_bids_p, controls["b_t"], controls["p_t"]
    )

    # === Return unified results ===
    return {
        "renewable_expected_profit": J1_r,
        "renewable_worst_case_penalty": J2_r,
        "regulator_price_stability": regulator_metrics["price_stability"],
        "regulator_reliability": regulator_metrics["reliability"],
        "regulator_renewable_integration": regulator_metrics["renewable_integration"],
    }


# ==============================================================
# End of unified model implementation
# ==============================================================


controls = {
    "b_t": [220]*24,
    "p_t": [50]*24,
    "q_u": 100,
    "q_o": 50
}

params = {
    "P_lower": [150]*24,
    "P_upper": [250]*24
}

