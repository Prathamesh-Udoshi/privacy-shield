"""
Laplace mechanism implementation for differential privacy.

This module provides functions to add Laplace noise to numeric values
for privacy-preserving data anonymization.
"""

import math
import random
from typing import Union, Tuple


def sample_laplace(scale: float) -> float:
    """
    Sample from the Laplace distribution using inverse CDF method.

    Args:
        scale: Scale parameter (b) of the Laplace distribution.
               For DP, scale = sensitivity / epsilon

    Returns:
        A random sample from Laplace(0, scale)
    """
    # Generate uniform random variable in (-0.5, 0.5)
    u = random.uniform(-0.5, 0.5)

    # Apply inverse CDF: F^{-1}(u) = scale * sign(u) * ln(1 - 2|u|)
    if u == 0:
        return 0.0

    noise = (1.0 / scale) * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))

    return noise


def add_laplace_noise(value: Union[int, float], sensitivity: float, epsilon: float) -> float:
    """
    Add Laplace noise to a numeric value for differential privacy.

    Args:
        value: The original numeric value
        sensitivity: The sensitivity of the query (max change in output when one record changes)
        epsilon: Privacy parameter (smaller = more privacy)

    Returns:
        The value with added Laplace noise
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")

    if sensitivity < 0:
        raise ValueError("Sensitivity must be non-negative")

    scale = sensitivity / epsilon
    noise = sample_laplace(scale)

    return float(value) + noise


def add_bounded_laplace_noise(value: Union[int, float],
                            sensitivity: float,
                            epsilon: float,
                            min_val: Union[int, float],
                            max_val: Union[int, float]) -> float:
    """
    Add bounded Laplace noise and clamp to valid range.

    This is useful for values that must stay within certain bounds (e.g., age).

    Args:
        value: The original numeric value
        sensitivity: The sensitivity of the query
        epsilon: Privacy parameter
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        The noisy value clamped to [min_val, max_val]
    """
    noisy_value = add_laplace_noise(value, sensitivity, epsilon)

    # Clamp to valid range
    return max(min_val, min(max_val, noisy_value))


def add_discrete_laplace_noise(value: int, sensitivity: int, epsilon: float) -> int:
    """
    Add discrete Laplace noise for integer values (like counts).

    Uses the continuous Laplace and rounds to nearest integer.

    Args:
        value: The original integer value
        sensitivity: The sensitivity (typically 1 for counts)
        epsilon: Privacy parameter

    Returns:
        The noisy integer value
    """
    noisy_float = add_laplace_noise(float(value), float(sensitivity), epsilon)
    return round(noisy_float)


def add_scaled_laplace_noise(value: Union[int, float],
                           sensitivity: float,
                           epsilon: float,
                           scale_factor: float = 1.0) -> float:
    """
    Add scaled Laplace noise for monetary values or other scaled data.

    Args:
        value: The original value
        sensitivity: The sensitivity
        epsilon: Privacy parameter
        scale_factor: Additional scaling factor (e.g., for currency units)

    Returns:
        The noisy value
    """
    return add_laplace_noise(value, sensitivity, epsilon) * scale_factor