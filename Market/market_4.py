import numpy as np

# ============================================================
# SHARED ENVIRONMENT
# ============================================================

class MarketClearingEnvironment:
    """
    Shared electricity market environment.
    Contains constants for demand and conventional producers,
    handles stochastic sampling and market clearing.
    """

    def __init__(self, bar_P):
        """
        bar_P : float
            Maximum renewable generation (MWh), treated as uncertain parameter.
        """
        # Fixed (global) constants
        self.num_conv = 3
        self.mu_D = 1000
        self.sigma_D = 100

        # Conventional producers' bid parameters
        self.mu_ib = [400, 350, 300]   # MWh
        self.sigma_ib = [50, 40, 30]
        self.mu_ip = [45, 50, 60]      # €/MWh
        self.sigma_ip = [5, 5, 5]

        self.bar_P = bar_P

    def sample_uncertainties(self, uncertain):
        """
        Draw stochastic realizations for one simulation.
        uncertain: dict to override random draws, e.g. {'D_t': value, 'P_t': value}
        """
        # Demand
        D_t = uncertain.get('D_t', np.random.normal(self.mu_D, self.sigma_D))

        # Conventional producers' bids
        b_i = [
            np.random.normal(self.mu_ib[i], self.sigma_ib[i])
            for i in range(self.num_conv)
        ]
        p_i = [
            np.random.normal(self.mu_ip[i], self.sigma_ip[i])
            for i in range(self.num_conv)
        ]

        # Renewable generation realization (bounded by uncertain bar_P)
        P_t = uncertain.get('P_t', np.random.uniform(150, self.bar_P))

        return D_t, b_i, p_i, P_t

    def market_clearing(self, D_t, b_i, p_i, b_t, p_t):
        """
        Compute market-clearing price and accepted quantities (merit-order).
        """
        bids = [(p_i[j], b_i[j]) for j in range(self.num_conv)] + [(p_t, b_t)]
        bids_sorted = sorted(bids, key=lambda x: x[0])

        supply_accum = 0.0
        c_t = bids_sorted[-1][0]  # fallback: last offer
        for price, qty in bids_sorted:
            supply_accum += qty
            if supply_accum >= D_t:
                c_t = price
                break

        accepted = [(price, qty) for price, qty in bids_sorted if price <= c_t]
        accepted_quantities = [qty for price, qty in accepted]

        accepted_renewable = b_t if p_t <= c_t else 0.0
        return c_t, accepted_quantities, accepted_renewable


# ============================================================
# RENEWABLE PRODUCER
# ============================================================

class RenewableProducer:
    """
    Renewable producer choosing (b_t, p_t) and evaluated under penalties.
    """

    def __init__(self, q_u, q_o):
        self.q_u = q_u
        self.q_o = q_o

    def profit(self, c_t, b_t, p_t, P_t):
        """
        Profit for one realization.
        """
        if p_t > c_t:
            return 0.0  # bid not accepted
        shortfall = max(0.0, b_t - P_t)
        surplus = max(0.0, P_t - b_t)
        return c_t * b_t - self.q_u * shortfall - self.q_o * surplus

    def expected_metrics(self, samples):
        """
        Compute expected profit and expected under-delivery.
        """
        profits = []
        underdeliveries = []
        for s in samples:
            profits.append(self.profit(s['c_t'], s['b_t'], s['p_t'], s['P_t']))
            underdeliveries.append(max(0.0, s['b_t'] - s['P_t']))
        return {
            'expected_profit': profits,
            'expected_under_delivery': np.mean(underdeliveries)
        }


# ============================================================
# SYSTEM REGULATOR
# ============================================================

class SystemRegulator:
    """
    Regulator controlling penalties (q_u, q_o),
    evaluating system-level cost, imbalance, and renewable share.
    """

    def __init__(self):
        pass

    def metrics(self, c_t, D_t, accepted_quantities, b_t, p_t, P_t, q_u, q_o):
        """
        Compute quantifiable system outcomes for one simulation.
        """
        total_supply = sum(accepted_quantities)
        imbalance = abs(D_t - total_supply)
        system_cost = sum([c_t * qty for qty in accepted_quantities])
        system_cost += q_u * max(0.0, b_t - P_t) + q_o * max(0.0, P_t - b_t)
        renewable_share = (b_t if p_t <= c_t else 0.0) / max(total_supply, 1e-6)
        return system_cost, imbalance, renewable_share

    def expected_metrics(self, samples):
        """
        Compute expected values of system metrics.
        """
        cost = np.mean([s['system_cost'] for s in samples])
        imbalance = np.mean([s['imbalance'] for s in samples])
        renewable_share = np.mean([s['renewable_share'] for s in samples])
        return {
            'expected_system_cost': cost,
            'expected_imbalance': imbalance,
            'expected_renewable_share': renewable_share
        }


# ============================================================
# UNIFIED MODEL INTERFACE
# ============================================================

def simulate_market(controls, uncertain_inputs, n_samples=1000):
    """
    Unified interface for simulating the renewable–regulator market model.

    Parameters
    ----------
    controls : dict
        {
            'renewable': {'b_t': float, 'p_t': float},
            'regulator': {'q_u': float, 'q_o': float}
        }

    uncertain_inputs : dict
        Must include the uncertain renewable capacity:
        {'bar_P': float, ...}
        Optional: may override other uncertain variables, e.g. {'D_t': value, 'P_t': value}.

    n_samples : int
        Number of Monte Carlo simulation runs.

    Returns
    -------
    results : dict
        {
            'renewable': {
                'expected_profit': float,
                'expected_under_delivery': float
            },
            'regulator': {
                'expected_system_cost': float,
                'expected_imbalance': float,
                'expected_renewable_share': float
            }
        }
    """

    # Extract controls
    b_t = controls['renewable']['b_t']
    p_t = controls['renewable']['p_t']
    q_u = controls['regulator']['q_u']
    q_o = controls['regulator']['q_o']

    # Extract uncertain parameter: bar_P
    if 'bar_P' not in uncertain_inputs:
        raise ValueError("uncertain_inputs must include 'bar_P' as an uncertain parameter.")
    bar_P = uncertain_inputs['bar_P']

    # Initialize shared environment and agents
    env = MarketClearingEnvironment(bar_P=bar_P)
    renewable = RenewableProducer(q_u=q_u, q_o=q_o)
    regulator = SystemRegulator()

    # Scenario results
    renewable_samples = []
    regulator_samples = []

    for _ in range(n_samples):
        # Sample stochastic elements
        D_t, b_i, p_i, P_t = env.sample_uncertainties(uncertain_inputs)

        # Market clearing
        c_t, accepted_quantities, accepted_renewable = env.market_clearing(
            D_t, b_i, p_i, b_t, p_t
        )

        # Renewable results
        renewable_samples.append({
            'c_t': c_t,
            'b_t': b_t,
            'p_t': p_t,
            'P_t': P_t
        })

        # Regulator results
        system_cost, imbalance, renewable_share = regulator.metrics(
            c_t, D_t, accepted_quantities, b_t, p_t, P_t, q_u, q_o
        )
        regulator_samples.append({
            'system_cost': system_cost,
            'imbalance': imbalance,
            'renewable_share': renewable_share
        })

    # Expected outcomes
    renewable_results = renewable.expected_metrics(renewable_samples)
    regulator_results = regulator.expected_metrics(regulator_samples)

    return {
        'renewable': renewable_results,
        'regulator': regulator_results
    }



controls = {
    'renewable': {'b_t': 220, 'p_t': 50},
    'regulator': {'q_u': 100, 'q_o': 50}
}

uncertain = {'bar_P': 250}  # renewable capacity upper bound
