"""
Laplace mechanism implementation for differential privacy.

This module provides functions to add Laplace noise to numeric values
for privacy-preserving data anonymization.
"""

import math
import random
import numpy as np
from typing import Union, Optional, List

# ---------------------------------------------------------------------------
# Seeded RNG  —  use set_seed() once per anonymization run for reproducibility
#
# Architecture: we keep a global SEED value and derive PER-COLUMN child RNGs
# using numpy SeedSequence.  This way:
#   - Column A always gets the same noise regardless of how many draws happened
#     before it (e.g. from preprocessing, AI analysis, etc.)
#   - Each column's RNG is independent from every other column's RNG
# ---------------------------------------------------------------------------
_GLOBAL_SEED: Optional[int] = None           # None  =  non-deterministic
_rng: np.random.Generator = np.random.default_rng()  # fallback for scalar callers


def set_seed(seed: Optional[int]) -> None:
    """Seed the module-level RNG for reproducible noise generation.

    When seed is not None, both the NumPy RNG and stdlib random module
    are deterministically seeded from the supplied integer.  Per-column
    child RNGs (see get_column_rng) are also derived from this seed, so
    column noise is stable regardless of the order in which other code
    consumes randomness.

    When seed is None (default), randomness is non-deterministic and the
    HMAC key remains the cryptographically-random key set at import time.
    """
    global _GLOBAL_SEED, _rng
    _GLOBAL_SEED = seed
    _rng = np.random.default_rng(seed)
    if seed is not None:
        random.seed(seed)
        # Derive a deterministic HMAC key from the seed so that hashed
        # ID columns are stable across seeded runs too.
        import hashlib as _hashlib
        seed_bytes = seed.to_bytes(8, 'big')
        derived_key = _hashlib.sha256(b'privacy-shield-hmac-v1:' + seed_bytes).digest()
        try:
            import dp.mechanisms as _mech
            _mech._SESSION_HMAC_KEY = derived_key
        except ImportError:
            pass  # mechanisms not yet imported; it will inherit the default key


def get_column_rng(column_name: str) -> np.random.Generator:
    """Return a deterministic, column-specific RNG.

    If a global seed has been set, each column gets its own child RNG
    derived from (seed, column_name_hash).  This ensures that adding or
    removing another column never shifts the noise of existing columns.

    If no seed is set, returns the shared global RNG (non-deterministic).
    """
    if _GLOBAL_SEED is None:
        return _rng
    # Derive a stable integer child-seed from the column name
    import hashlib as _hashlib
    col_hash = int(_hashlib.sha256(column_name.encode()).hexdigest(), 16) & 0xFFFF_FFFF
    child_seed = (_GLOBAL_SEED * 2654435761 + col_hash) & 0xFFFF_FFFF_FFFF_FFFF
    return np.random.default_rng(child_seed)


def get_rng() -> np.random.Generator:
    """Return the current RNG (for use in other dp modules)."""
    return _rng


def sample_laplace(scale: float, size: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Sample from the Laplace distribution using inverse CDF method.
    Routes through the module-level seeded RNG for reproducibility.

    Args:
        scale: Scale parameter (b) of the Laplace distribution.
               For DP, scale = sensitivity / epsilon
        size: Number of samples to generate (optional)

    Returns:
        A random sample or array of samples from Laplace(0, scale)
    """
    if size is None:
        # Scalar path — use seeded RNG via uniform draw
        u = _rng.uniform(-0.5 + 1e-15, 0.5 - 1e-15)
        # Apply inverse CDF: F^{-1}(u) = -scale * sign(u) * ln(1 - 2|u|)
        return -scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))
    else:
        # Vectorized path — seeded RNG
        u = _rng.uniform(-0.5 + 1e-15, 0.5 - 1e-15, size=size)
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
