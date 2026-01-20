"""
Gaussian mechanism implementation for differential privacy.

This module provides functions to add Gaussian noise to numeric values
for (epsilon, delta)-differential privacy.
"""

import math
import random
import numpy as np
from typing import Union, List, Optional, Tuple


def sample_gaussian(sigma: float, size: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Sample from Gaussian distribution with mean 0 and standard deviation sigma.
    
    Args:
        sigma: Standard deviation of the Gaussian distribution
        size: Number of samples to generate
        
    Returns:
        Random sample(s) from Gaussian(0, sigma)
    """
    if size is None:
        return random.gauss(0, sigma)
    return np.random.normal(0, sigma, size)


def add_gaussian_noise(value: Union[int, float, List, np.ndarray], 
                       sensitivity: float, 
                       epsilon: float, 
                       delta: float) -> Union[float, np.ndarray]:
    """
    Add Gaussian noise for (epsilon, delta)-differential privacy.
    
    Formula for sigma: sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
    
    Args:
        value: Original numeric value(s)
        sensitivity: L2 sensitivity of the query
        epsilon: Privacy parameter
        delta: Privacy parameter (probability of privacy failure)
        
    Returns:
        Noisy value(s)
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")
    if delta <= 0 or delta >= 1:
        raise ValueError("Delta must be in (0, 1)")
    
    # Standard formula for Gaussian mechanism sigma
    sigma = (sensitivity * math.sqrt(2 * math.log(1.25 / delta))) / epsilon
    
    if isinstance(value, (list, np.ndarray)):
        arr = np.array(value, dtype=float)
        return arr + sample_gaussian(sigma, size=len(arr))
    
    return float(value) + sample_gaussian(sigma)


def add_bounded_gaussian_noise(value: Union[int, float, List, np.ndarray],
                             sensitivity: float,
                             epsilon: float,
                             delta: float,
                             min_val: Union[int, float],
                             max_val: Union[int, float]) -> Union[float, np.ndarray]:
    """
    Add bounded Gaussian noise and clamp to valid range.
    """
    noisy_values = add_gaussian_noise(value, sensitivity, epsilon, delta)
    return np.clip(noisy_values, min_val, max_val)
