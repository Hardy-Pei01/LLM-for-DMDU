import numpy as np
from typing import Dict, Any, List


# ============================================================
# 1. Common Environment
# ============================================================

class MarketEnvironment:
    """
    Shared stochastic environment and market-clearing mechanism.
    The imbalance penalty (lambda_penalty) is provided exogenously
    by the system regulator at run time.
    """

    def __init__(self, constants: Dict[str, Any], uncertainties: Dict[str, Any]):
        # Fixed (known) parameters
        self.mu_D = constants["mu_D"]
        self.sigma_D = constants["sigma_D"]
        self.mu_W = constants["mu_W"]
        self.sigma_W = constants["sigma_W"]

        self.Q_w = constants["Q_w"]
        self.Q_c = constants["Q_c"]          # dict: conventional producer -> capacity
        self.a = constants["solar_a"]
        self.b = constants["solar_b"]

        # Deep-uncertainty parameters (scenario inputs)
        self.mu_c = uncertainties["mu_c"]    # dict
        self.sigma_c = uncertainties["sigma_c"]
        self.mu_s = uncertainties["mu_s"]
        self.sigma_s = uncertainties["sigma_s"]

        self.T = 24

    # -----------------------------
    # Exogenous stochastic processes
    # -----------------------------

    def solar_quantity(self, t: int) -> float:
        return max(0.0, self.a + self.b * np.cos(2 * np.pi * t / 24))

    def sample_demand(self) -> float:
        return max(0.0, np.random.normal(self.mu_D, self.sigma_D))

    def sample_wind(self) -> float:
        return max(0.0, np.random.normal(self.mu_W, self.sigma_W))

    def sample_prices(self) -> Dict[str, float]:
        prices = {}
        for c in self.mu_c:
            prices[c] = np.random.normal(self.mu_c[c], self.sigma_c[c])
        prices["solar"] = np.random.normal(self.mu_s, self.sigma_s)
        return prices

    # -----------------------------
    # Market-clearing transition
    # -----------------------------

    def clear_market(
            self,
            bids: Dict[str, Dict[str, float]],
            demand: float,
            wind_realization: float,
            lambda_penalty: float
    ) -> Dict[str, Any]:
        """
        Merit-order market clearing where any bid with price less than or
        equal to the clearing price is fully accepted, even if demand is exceeded.
        """

        # ------------------------------------------------
        # Step 1: Determine clearing price
        # ------------------------------------------------
        sorted_bids = sorted(bids.items(), key=lambda x: x[1]["price"])

        cumulative = 0.0
        clearing_price = None

        for producer, bid in sorted_bids:
            cumulative += bid["quantity"]
            if cumulative >= demand:
                clearing_price = bid["price"]
                break

        # Safety fallback (should not occur if capacity is sufficient)
        if clearing_price is None:
            clearing_price = max(bid["price"] for bid in bids.values())

        # ------------------------------------------------
        # Step 2: Full acceptance of all bids ≤ clearing price
        # ------------------------------------------------
        dispatch = {}
        for producer, bid in bids.items():
            if bid["price"] <= clearing_price:
                dispatch[producer] = bid["quantity"]
            else:
                dispatch[producer] = 0.0

        # ------------------------------------------------
        # Step 3: Wind shortfall and penalty
        # ------------------------------------------------
        wind_dispatch = dispatch.get("wind", 0.0)
        shortfall = max(0.0, wind_dispatch - wind_realization)
        penalty = lambda_penalty * shortfall

        return {
            "dispatch": dispatch,
            "price": clearing_price,
            "wind_shortfall": shortfall,
            "wind_penalty": penalty
        }


# ============================================================
# 2. Wind Producer Perspective
# ============================================================

class WindProducer:
    """
    Wind producer: controls bidding decisions and evaluates its objectives.
    """

    def __init__(self, decisions: Dict[str, Any]):
        # decisions["wind"] contains wind-specific decisions
        self.quantities = decisions["wind"]["q_w"]  # dict: t -> quantity
        self.prices = decisions["wind"]["p_w"]      # dict: t -> price

    def bid(self, t: int) -> Dict[str, float]:
        return {
            "quantity": self.quantities[t],
            "price": self.prices[t]
        }

    @staticmethod
    def objectives(history: List[Dict[str, Any]]) -> Dict[str, float]:
        revenues = []
        shortfalls = []

        for h in history:
            revenue = (
                h["price"] * h["dispatch"]["wind"]
                - h["wind_penalty"]
            )
            revenues.append(revenue)
            shortfalls.append(h["wind_shortfall"])

        return {
            # Objective W1
            "W1_expected_profit": float(np.mean(revenues)),
            # Objective W2
            "W2_expected_imbalance": float(np.mean(shortfalls)),
            # Objective W3
            "W3_revenue_variance": float(np.var(revenues))
        }


# ============================================================
# 3. System Regulator Perspective
# ============================================================

class SystemRegulator:
    """
    System regulator: controls the imbalance penalty and evaluates
    system-level objectives.
    """

    def __init__(self, decisions: Dict[str, Any]):
        # decisions["regulator"] contains regulator-specific decisions
        self.lambda_penalty = decisions["regulator"]["lambda_penalty"]

    @staticmethod
    def objectives(history: List[Dict[str, Any]]) -> Dict[str, float]:
        prices = [h["price"] for h in history]
        wind_dispatch = [h["dispatch"]["wind"] for h in history]
        shortfalls = [h["wind_shortfall"] for h in history]

        return {
            # Objective R1
            "R1_expected_imbalance": float(np.mean(shortfalls)),
            # Objective R2
            "R2_price_variance": float(np.var(prices)),
            # Objective R3
            "R3_expected_wind_dispatch": float(np.mean(wind_dispatch))
        }


# ============================================================
# 4. Unified Composed Model
# ============================================================

class UnifiedMarketModel:
    """
    Unified model embedding all perspectives in a shared environment.
    """

    def __init__(
        self,
        environment: MarketEnvironment,
        wind_producer: WindProducer,
        regulator: SystemRegulator
    ):
        self.env = environment
        self.wind = wind_producer
        self.regulator = regulator

    def run(self) -> List[Dict[str, Any]]:
        history = []

        for t in range(1, self.env.T + 1):
            demand = self.env.sample_demand()
            wind_realization = self.env.sample_wind()
            prices = self.env.sample_prices()

            bids = {
                "wind": self.wind.bid(t),
                "solar": {
                    "quantity": self.env.solar_quantity(t),
                    "price": prices["solar"]
                }
            }

            for c in self.env.Q_c:
                bids[c] = {
                    "quantity": self.env.Q_c[c],
                    "price": prices[c]
                }

            result = self.env.clear_market(
                bids=bids,
                demand=demand,
                wind_realization=wind_realization,
                lambda_penalty=self.regulator.lambda_penalty
            )

            history.append(result)

        return history


# ============================================================
# 5. Public Interface Function
# ============================================================

def simulate_unified_model(
    decisions: Dict[str, Any],
    uncertainties: Dict[str, Any],
    constants: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    """
    Unified simulation interface.

    Inputs
    ------
    decisions:
        {
          "wind": {
              "q_w": {t: quantity},
              "p_w": {t: price}
          },
          "regulator": {
              "lambda_penalty": float
          }
        }

    uncertainties:
        {
          "mu_c": {producer: mean},
          "sigma_c": {producer: std},
          "mu_s": float,
          "sigma_s": float
        }

    constants:
        {
          "mu_D": float,
          "sigma_D": float,
          "mu_W": float,
          "sigma_W": float,
          "Q_w": float,
          "Q_c": {producer: capacity},
          "solar_a": float,
          "solar_b": float
        }

    Returns
    -------
    Dictionary mapping perspective -> objective values
    """

    env = MarketEnvironment(constants, uncertainties)
    wind = WindProducer(decisions)
    regulator = SystemRegulator(decisions)

    model = UnifiedMarketModel(env, wind, regulator)
    history = model.run()

    return {
        "wind_producer": WindProducer.objectives(history),
        "system_regulator": SystemRegulator.objectives(history)
    }
