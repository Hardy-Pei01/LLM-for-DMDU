import numpy as np

class MarketEnvironment:
    """
    Shared stochastic environment:
    - samples uncertainty
    - clears the market
    - produces system-wide trajectories
    """

    def __init__(self, constants, uncertainties, decisions):
        self.constants = constants
        self.uncertainties = uncertainties
        self.decisions = decisions

        self.T = 24
        self.conventional_ids = [0, 1, 2]

    def sample_uncertainty(self):
        """Sample all stochastic variables for one day."""
        mu_D, sigma_D = self.constants["mu_D"], self.constants["sigma_D"]
        mu_G, sigma_G = self.constants["mu_G"], self.constants["sigma_G"]

        D = np.random.normal(mu_D, sigma_D, self.T)
        G = np.random.normal(mu_G, sigma_G, self.T)

        p_conv = {}
        for i in self.conventional_ids:
            mu_pi, sigma_pi = self.uncertainties["mu_pi"][i], self.uncertainties["sigma_pi"][i]
            p_conv[i] = np.random.normal(mu_pi, sigma_pi, self.T)

        p_solar = np.random.normal(
            self.uncertainties["mu_ps"],
            self.uncertainties["sigma_ps"],
            self.T
        )

        return D, G, p_conv, p_solar

    def solar_quantity(self, t):
        a, b = self.constants["a"], self.constants["b"]
        return max(0.0, a + b * np.cos(2 * np.pi * (t + 1) / 24))

    def clear_market(self, t, D_t, p_conv_t, p_solar_t, wind_bid):
        """
        Merit-order market clearing with uniform pricing:
        all bids with price <= clearing price are accepted.
        """

        bids = []

        # Conventional producers
        for i in self.conventional_ids:
            bids.append((
                self.constants["b_conv"][i],
                p_conv_t[i],
                f"conv_{i}"
            ))

        # Solar producer
        bids.append((
            self.solar_quantity(t),
            p_solar_t,
            "solar"
        ))

        # Wind producer
        b_w, p_w = wind_bid
        bids.append((b_w, p_w, "wind"))

        # Sort bids by increasing price
        bids.sort(key=lambda x: x[1])

        # Step 1: determine clearing price
        cumulative_supply = 0.0
        clearing_price = None

        for qty, price, _ in bids:
            cumulative_supply += qty
            if cumulative_supply >= D_t:
                clearing_price = price
                break

        # Step 2: accept all bids with price <= clearing price
        accepted = {}
        for qty, price, name in bids:
            if price <= clearing_price:
                accepted[name] = qty
            else:
                accepted[name] = 0.0

        return accepted, clearing_price


class WindProducer:
    """
    Wind producer perspective:
    evaluates profit and risk objectives
    """

    def __init__(self, decisions, constants):
        self.bids = decisions["wind_bids"]
        self.q_u = decisions["q_u"]
        self.constants = constants

    def evaluate(self, trajectory):
        profits = []
        imbalances = []

        for t in range(24):
            c_t = trajectory["prices"][t]
            x_w = trajectory["dispatch"]["wind"][t]
            G_t = trajectory["wind_generation"][t]

            under_delivery = max(0.0, x_w - G_t)
            profit = c_t * x_w - self.q_u * under_delivery

            profits.append(profit)
            imbalances.append(under_delivery)

        return {
            "expected_profit": np.sum(profits),
            "total_imbalance": np.sum(imbalances),
            "revenue_variance": np.var(profits),
        }


class SystemRegulator:
    """
    Regulator perspective:
    evaluates system-wide reliability and cost objectives
    """

    def __init__(self):
        pass

    def evaluate(self, trajectory):
        shortages = []
        total_imbalance = []
        consumer_costs = []
        wind_utilization = []

        for t in range(24):
            D_t = trajectory["demand"][t]
            total_supply = sum(
                trajectory["dispatch"][agent][t]
                for agent in trajectory["dispatch"]
            )

            shortage = max(0.0, D_t - total_supply)
            shortages.append(shortage)

            imbalance_t = sum(
                abs(trajectory["dispatch"][agent][t] - trajectory["realized_generation"].get(agent, 0.0))
                for agent in trajectory["dispatch"]
            )
            total_imbalance.append(imbalance_t)

            consumer_costs.append(trajectory["prices"][t] * D_t)
            wind_utilization.append(trajectory["dispatch"]["wind"][t])

        return {
            "expected_shortfall": np.sum(shortages),
            "total_system_imbalance": np.sum(total_imbalance),
            "total_consumer_payment": np.sum(consumer_costs),
            "total_wind_dispatch": np.sum(wind_utilization),
        }


def simulate_unified_model(decisions, uncertainties, constants):
    """
    Unified simulation interface.

    Parameters
    ----------
    decisions : dict
        - "wind_bids": list of (b_wt, p_wt) for t=1..24
        - "q_u": imbalance penalty

    uncertainties : dict
        - "mu_pi": list of means for conventional bids
        - "sigma_pi": list of std devs for conventional bids
        - "mu_ps": mean solar bid price
        - "sigma_ps": std dev solar bid price

    constants : dict
        - demand, wind, solar, and capacity parameters

    Returns
    -------
    dict
        Objective values for each perspective
    """

    env = MarketEnvironment(constants, uncertainties, decisions)

    D, G, p_conv, p_solar = env.sample_uncertainty()

    trajectory = {
        "demand": D,
        "wind_generation": G,
        "prices": [],
        "dispatch": {"wind": [], "solar": [], "conv_0": [], "conv_1": [], "conv_2": []},
        "realized_generation": {"wind": G},
    }

    for t in range(24):
        accepted, price = env.clear_market(
            t,
            D[t],
            {i: p_conv[i][t] for i in env.conventional_ids},
            p_solar[t],
            decisions["wind_bids"][t],
        )

        trajectory["prices"].append(price)

        for agent in trajectory["dispatch"]:
            trajectory["dispatch"][agent].append(accepted.get(agent, 0.0))

    wind_agent = WindProducer(decisions, constants)
    regulator = SystemRegulator()

    wind_objectives = wind_agent.evaluate(trajectory)
    regulator_objectives = regulator.evaluate(trajectory)

    return {
        "wind_producer_objectives": wind_objectives,
        "regulator_objectives": regulator_objectives,
    }
