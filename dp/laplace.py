"""
Laplace mechanism implementation for differential privacy.

This module provides functions to add Laplace noise to numeric values
for privacy-preserving data anonymization.
"""

import math
import random
import numpy as np
from typing import Union, Optional, List


def sample_laplace(scale: float, size: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Sample from the Laplace distribution using inverse CDF method.
    Uses NumPy for vectorized sampling if size is provided.

    Args:
        scale: Scale parameter (b) of the Laplace distribution.
               For DP, scale = sensitivity / epsilon
        size: Number of samples to generate (optional)

    Returns:
        A random sample or array of samples from Laplace(0, scale)
    """
    if size is None:
        # Generate uniform random variable in (-0.5, 0.5)
        u = random.uniform(-0.5, 0.5)
        if u == 0:
            return 0.0
        # Apply inverse CDF: F^{-1}(u) = -scale * sign(u) * ln(1 - 2|u|)
        return -scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))
    else:
        # Vectorized version using NumPy
        u = np.random.uniform(-0.5, 0.5, size)
        # Handle u=0 cases (rare with floats but for consistency)
        u = np.where(u == 0, 1e-15, u) 
        return -scale * np.sign(u) * np.log(1.0 - 2.0 * np.abs(u))


def add_laplace_noise(value: Union[int, float, List, np.ndarray], sensitivity: float, epsilon: float) -> Union[float, np.ndarray]:
    """
    Add Laplace noise to a numeric value or array for differential privacy.

    Args:
        value: The original numeric value(s)
        sensitivity: The sensitivity of the query
        epsilon: Privacy parameter

    Returns:
        The value(s) with added Laplace noise
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")

    if sensitivity < 0:
        raise ValueError("Sensitivity must be non-negative")

    scale = sensitivity / epsilon
    
    if isinstance(value, (list, np.ndarray)):
        arr = np.array(value, dtype=float)
        return arr + sample_laplace(scale, size=len(arr))
    
    return float(value) + sample_laplace(scale)


def add_bounded_laplace_noise(value: Union[int, float, List, np.ndarray],
                            sensitivity: float,
                            epsilon: float,
                            min_val: Union[int, float],
                            max_val: Union[int, float]) -> Union[float, np.ndarray]:
    """
    Add bounded Laplace noise and clamp to valid range. Vectorized.

    Args:
        value: The original numeric value(s)
        sensitivity: The sensitivity of the query
        epsilon: Privacy parameter
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        The noisy value(s) clamped to [min_val, max_val]
    """
    noisy_values = add_laplace_noise(value, sensitivity, epsilon)
    return np.clip(noisy_values, min_val, max_val)


def add_discrete_laplace_noise(value: Union[int, List[int], np.ndarray], sensitivity: int, epsilon: float) -> Union[int, np.ndarray]:
    """
    Add discrete Laplace noise for integer values. Vectorized.

    Args:
        value: The original integer value(s)
        sensitivity: The sensitivity
        epsilon: Privacy parameter

    Returns:
        The noisy integer value(s)
    """
    noisy_float = add_laplace_noise(value, float(sensitivity), epsilon)
    if isinstance(noisy_float, np.ndarray):
        return np.round(noisy_float).astype(int)
    return int(round(noisy_float))


def add_scaled_laplace_noise(value: Union[int, float, List, np.ndarray],
                           sensitivity: float,
                           epsilon: float,
                           scale_factor: float = 1.0) -> Union[float, np.ndarray]:
    """
    Add scaled Laplace noise for monetary values or other scaled data. Vectorized.

    Args:
        value: The original value(s)
        sensitivity: The sensitivity
        epsilon: Privacy parameter
        scale_factor: Additional scaling factor

    Returns:
        The noisy value(s)
    """
    return add_laplace_noise(value, sensitivity, epsilon) * scale_factor
