import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ================================================================
# 1. COMMON ENVIRONMENT
# ================================================================

@dataclass
class MarketEnvironment:
    """Shared stochastic electricity market environment."""
    T: int = 24

    # Exogenous stochastic parameters
    mu_D: np.ndarray = field(default_factory=lambda: np.zeros(24))      # Mean demand
    sigma_D: np.ndarray = field(default_factory=lambda: np.ones(24))    # Std dev demand
    mu_G: np.ndarray = field(default_factory=lambda: np.zeros(24))      # Mean wind generation
    sigma_G: np.ndarray = field(default_factory=lambda: np.ones(24))    # Std dev wind generation

    # Solar parameters
    a: float = 50.0
    b: float = 40.0
    mu_s: np.ndarray = field(default_factory=lambda: np.full(24, 30.0))
    sigma_s: np.ndarray = field(default_factory=lambda: np.full(24, 5.0))

    # Conventional producer parameters
    alpha: np.ndarray = field(default_factory=lambda: np.array([25.0, 27.0, 30.0]))
    beta: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.6, 0.4]))
    sigma_conv: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.5, 3.0]))
    Q_conv: np.ndarray = field(default_factory=lambda: np.array([100.0, 120.0, 150.0]))

    # Global penalties (nominal)
    c_short: np.ndarray = field(default_factory=lambda: np.full(24, 80.0))
    c_surp: np.ndarray = field(default_factory=lambda: np.full(24, 10.0))

    # Wind capacity
    Q_w_max: float = 100.0

    def solar_quantity(self, t: int) -> float:
        """Solar producer’s deterministic bid quantity."""
        return max(0.0, self.a + self.b * np.cos(2 * np.pi * t / 24))

    def sample_state(self, uncertainties: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate stochastic realizations for demand, wind, and solar bids."""
        D = np.random.normal(self.mu_D, self.sigma_D)
        G = np.random.normal(self.mu_G, self.sigma_G)
        P_s = np.random.normal(self.mu_s, self.sigma_s)
        return D, G, P_s


# ================================================================
# 2. RENEWABLE PRODUCER MODULE
# ================================================================

@dataclass
class WindProducer:
    """Wind-power producer’s model."""
    env: MarketEnvironment
    q: np.ndarray = field(default_factory=lambda: np.full(24, 50.0))   # bid quantity (control)
    p_bid: np.ndarray = field(default_factory=lambda: np.full(24, 30.0))  # bid price (control)

    # Effective penalties (scaled by regulator)
    lambda_penalty: float = 1.0

    def dispatch_quantity(self, p_market: float, t: int) -> float:
        """Determine dispatched quantity based on bid acceptance."""
        return self.q[t] if self.p_bid[t] <= p_market else 0.0

    def profit_objective(self, p_market: np.ndarray, G: np.ndarray) -> float:
        """Compute expected profit over all hours."""
        profit = 0.0
        for t in range(self.env.T):
            Q_disp = self.dispatch_quantity(p_market[t], t)
            revenue = Q_disp * p_market[t]
            c_short_eff = self.lambda_penalty * self.env.c_short[t]
            c_surp_eff = self.lambda_penalty * self.env.c_surp[t]
            penalty = c_short_eff * max(0, Q_disp - G[t]) + c_surp_eff * max(0, G[t] - Q_disp)
            profit += revenue - penalty
        return profit

    def reliability_objective(self, p_market: np.ndarray, G: np.ndarray) -> float:
        """Probability (fraction of hours) of meeting or exceeding commitment."""
        reliability = 0.0
        for t in range(self.env.T):
            Q_disp = self.dispatch_quantity(p_market[t], t)
            reliability += 1.0 if G[t] >= Q_disp else 0.0
        return reliability / self.env.T


# ================================================================
# 3. REGULATOR MODULE
# ================================================================

@dataclass
class Regulator:
    """System regulator or market operator model."""
    env: MarketEnvironment
    lambda_t: np.ndarray = field(default_factory=lambda: np.ones(24))    # imbalance penalty scale
    phi_t: np.ndarray = field(default_factory=lambda: np.zeros(24))      # conventional price adjustment

    def conv_bid_prices(self, t: int) -> np.ndarray:
        """Compute conventional producers’ bid prices."""
        prices = []
        for i in range(len(self.env.alpha)):
            base = self.env.alpha[i] + self.env.beta[i] * self.phi_t[t]
            prices.append(np.random.normal(base, self.env.sigma_conv[i]))
        return np.array(prices)

    def market_clearing(self, t: int, q_w: float, p_w: float, P_s: float) -> Tuple[float, Dict[str, float]]:
        """Compute market-clearing price using simplified merit order."""
        # Construct supply stack
        bids = []
        # Conventional producers
        for i, Q_i in enumerate(self.env.Q_conv):
            bids.append((self.conv_bid_prices(t)[i], Q_i))
        # Solar
        bids.append((P_s, self.env.solar_quantity(t)))
        # Wind
        bids.append((p_w, q_w))
        # Sort by price
        bids = sorted(bids, key=lambda x: x[0])
        # Demand draw
        D_t = np.random.normal(self.env.mu_D[t], self.env.sigma_D[t])
        supply_cum, price = 0.0, 0.0
        for (p, q) in bids:
            supply_cum += q
            if supply_cum >= D_t:
                price = p
                break
        return price, {"demand": D_t, "supply": supply_cum}

    def evaluate_market(
        self,
        q_w: np.ndarray,
        p_w: np.ndarray,
        P_s: np.ndarray
    ) -> Tuple[np.ndarray, float, float, float]:
        """Simulate market over T and compute objectives."""
        prices, reliability_flags = [], []
        total_consumer_surplus, total_producer_surplus = 0.0, 0.0
        for t in range(self.env.T):
            p_market, stats = self.market_clearing(t, q_w[t], p_w[t], P_s[t])
            prices.append(p_market)
            # Approximate surpluses (simplified for demonstration)
            consumer_surplus = 0.5 * stats["demand"] * max(0, 100 - p_market)
            producer_surplus = stats["supply"] * max(0, p_market - 20)
            total_consumer_surplus += consumer_surplus
            total_producer_surplus += producer_surplus
            reliability_flags.append(stats["supply"] >= stats["demand"])
        prices = np.array(prices)
        reliability = np.mean(reliability_flags)
        efficiency = total_consumer_surplus + total_producer_surplus
        price_variance = np.var(prices)
        return prices, efficiency, reliability, price_variance


# ================================================================
# 4. UNIFIED SIMULATION INTERFACE
# ================================================================

def simulate_unified_model(
    controls: Dict[str, np.ndarray],
    uncertainties: Dict[str, np.ndarray],
    env_params: Dict[str, float] = {}
) -> Dict[str, float]:
    """
    Unified simulation interface.
    Inputs:
        controls: dict with control variables:
            {
              "q_w": np.ndarray(24),
              "p_w": np.ndarray(24),
              "lambda_t": np.ndarray(24),
              "phi_t": np.ndarray(24)
            }
        uncertainties: dict with stochastic parameter overrides (optional)
        env_params: optional overrides for MarketEnvironment parameters
    Output:
        dict with all objective function values:
            {
              "wind_profit": float,
              "wind_reliability": float,
              "reg_efficiency": float,
              "reg_reliability": float,
              "price_variance": float
            }
    """
    # Initialize environment
    env = MarketEnvironment(**env_params)

    # Initialize agents
    wind = WindProducer(env=env, q=controls["q_w"], p_bid=controls["p_w"], lambda_penalty=np.mean(controls["lambda_t"]))
    regulator = Regulator(env=env, lambda_t=controls["lambda_t"], phi_t=controls["phi_t"])

    # Sample stochastic components
    D, G, P_s = env.sample_state(uncertainties)

    # Run market simulation from regulator’s perspective
    p_market, eff, rel_sys, var_p = regulator.evaluate_market(
        q_w=controls["q_w"], p_w=controls["p_w"], P_s=P_s
    )

    # Evaluate wind producer objectives given market outcome
    profit = wind.profit_objective(p_market, G)
    rel_wind = wind.reliability_objective(p_market, G)

    return {
        "wind_profit": profit,
        "wind_reliability": rel_wind,
        "reg_efficiency": eff,
        "reg_reliability": rel_sys,
        "price_variance": var_p
    }
