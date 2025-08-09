"""Core pricing utilities including cost and revenue functions and EMSRb algorithm."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from statistics import NormalDist
from typing import Iterable, List


def marginal_cost(quantity: int, base_cost: float = 10.0, variable_cost: float = 2.0) -> float:
    """Return marginal cost for producing the next unit.

    Parameters
    ----------
    quantity: int
        Number of units already produced.
    base_cost: float
        Fixed cost component for the item.
    variable_cost: float
        Variable cost per additional unit.
    """
    return base_cost + variable_cost * quantity


def expected_revenue(price: float, expected_demand: float) -> float:
    """Calculate expected revenue for a given price and demand forecast."""
    return price * expected_demand


def emsr_b(prices: List[float], means: List[float], stds: List[float]) -> List[float]:
    """Calibrate EMSRb protection levels.

    Parameters
    ----------
    prices : list of fare class prices in descending order.
    means : list of mean demands for each class.
    stds : list of demand standard deviations for each class.

    Returns
    -------
    list
        Protection levels for each fare class.
    """
    if not (len(prices) == len(means) == len(stds)):
        raise ValueError("Input lists must have the same length")

    protection_levels: List[float] = [0.0] * len(prices)
    cumulative_mean = 0.0
    cumulative_variance = 0.0
    highest_price = prices[0]

    for i in range(len(prices) - 1):
        cumulative_mean += means[i]
        cumulative_variance += stds[i] ** 2
        nd = NormalDist(mu=cumulative_mean, sigma=math.sqrt(cumulative_variance))
        ratio = prices[i + 1] / highest_price
        protection_levels[i] = nd.inv_cdf(1 - ratio)

    protection_levels[-1] = 0.0
    return protection_levels


def choose_price(
    prices: List[float], protection_levels: List[float], remaining_capacity: int
) -> float:
    """Return the optimal price based on remaining capacity and protection levels."""
    for price, protect in zip(prices, protection_levels):
        if remaining_capacity > protect:
            return price
    return prices[-1]
