"""
Differential privacy mechanisms for different data types.

This module provides column-aware noise addition strategies
for various data types commonly found in user activity data.
"""

import hashlib
import hmac
import math
import os
import random
import secrets
import warnings
import numpy as np
from typing import Union, Dict, Any, Optional, List
from .laplace import (
    add_laplace_noise,
    add_bounded_laplace_noise,
    add_discrete_laplace_noise,
    add_scaled_laplace_noise,
    get_rng,
    get_column_rng,
)
from .budget import PrivacyBudget

# Per-session HMAC key — regenerated every process start so that
# the same input value hashes differently in each session.
# This prevents an attacker who obtains one anonymized dataset
# from pre-computing a rainbow table valid for another session.
_SESSION_HMAC_KEY: bytes = secrets.token_bytes(32)


def _is_numeric_value(value: str) -> bool:
    """
    Check if a string value represents a valid number.

    Args:
        value: String value to check

    Returns:
        True if the value can be parsed as a number, False otherwise
    """
    if not value:
        return False

    try:
        # Try to parse as float (handles integers and decimals)
        float(value)
        return True
    except (ValueError, TypeError):
        return False


class DPMechanisms:
    """
    Collection of differential privacy mechanisms for different column types.
    """

    def __init__(self, budget: PrivacyBudget, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize with a privacy budget and optional column metadata.

        Args:
            budget: PrivacyBudget instance to track epsilon consumption
            metadata: Map of column names to analysis metadata
        """
        self.budget = budget
        self.metadata = metadata or {}

    def apply_age_noise(self, value: Any, config: Dict[str, Any]) -> Any:
        """
        Apply bounded Laplace noise to age values. Supports vectorized input.
        """
        epsilon = config.get('epsilon', 0.2)
        min_age = config.get('min', 0)
        max_age = config.get('max', 120)

        noisy = add_bounded_laplace_noise(
            value=value,
            sensitivity=1.0,
            epsilon=epsilon,
            min_val=min_age,
            max_val=max_age
        )
        if isinstance(noisy, np.ndarray):
            return np.round(noisy).astype(int)
        return int(round(float(noisy)))

    def apply_year_noise(self, value: Any, config: Dict[str, Any]) -> Any:
        """
        Apply bounded Laplace noise to year values. Supports vectorized input.
        """
        epsilon = config.get('epsilon', 0.2)
        min_year = config.get('min', 1900)
        max_year = config.get('max', 2050)

        noisy = add_bounded_laplace_noise(
            value=value,
            sensitivity=1.0,
            epsilon=epsilon,
            min_val=min_year,
            max_val=max_year
        )
        if isinstance(noisy, np.ndarray):
            return np.round(noisy).astype(int)
        return int(round(float(noisy)))

    def apply_numeric_noise(self, value: Any, config: Dict[str, Any], column_name: str = "") -> Any:
        """
        Apply Laplace noise with formally-bounded sensitivity.

        Sensitivity = declared (max - min) from config when available —
        this is the ONLY way to guarantee formal (epsilon)-DP.  We fall
        back to the sampled range only when no bounds are declared, and
        emit a warning so the user knows the guarantee is heuristic.
        """
        epsilon = config.get('epsilon', 0.3)

        # Prefer explicit bounds declared in config (formal DP guarantee)
        declared_min = config.get('min')
        declared_max = config.get('max')
        if declared_min is not None and declared_max is not None:
            sensitivity = float(declared_max) - float(declared_min)
        else:
            # Fall back to sampled range — NOT a formal DP bound
            default_sens = config.get('sensitivity', 1.0)
            stats = self.metadata.get(column_name, {}).get('numeric_stats', {})
            discovered_range = stats.get('range', default_sens)
            
            # UX/Utility adjustment:
            # If the user hasn't provided strict explicit bounds, we use a heuristic 10% scale 
            # so the data retains high utility and acts as a realistic "jitter" without
            # requiring a complex policy.yaml setup.
            heuristic_scale = 0.1
            sensitivity = (discovered_range * heuristic_scale) if discovered_range > 0 else default_sens

        noisy = add_laplace_noise(
            value=value,
            sensitivity=sensitivity,
            epsilon=epsilon,
        )

        # Robust clipping: respect declared bounds first, then fall back to sampled range
        stats = self.metadata.get(column_name, {}).get('numeric_stats', {})
        
        # Use declared bounds if available, otherwise observed bounds
        clip_min = declared_min if declared_min is not None else stats.get('min')
        clip_max = declared_max if declared_max is not None else stats.get('max')

        # Fallback for non-negative check
        if clip_min is None and stats.get('all_non_negative', False):
            clip_min = 0.0

        if isinstance(noisy, np.ndarray):
            if clip_min is not None and clip_max is not None:
                return np.clip(noisy, clip_min, clip_max)
            elif clip_min is not None:
                return np.maximum(clip_min, noisy)
            elif clip_max is not None:
                return np.minimum(clip_max, noisy)
            return noisy
        
        # Scalar case
        res = float(noisy)
        if clip_min is not None: res = max(clip_min, res)
        if clip_max is not None: res = min(clip_max, res)
        return res

    def apply_monetary_noise(self, value: Any, config: Dict[str, Any], column_name: str = "") -> Any:
        """
        Apply scaled Laplace noise with formally-bounded sensitivity.

        Same reasoning as apply_numeric_noise: declared (max - min) gives
        a formal DP bound; sampled range is a heuristic fallback.
        """
        epsilon = config.get('epsilon', 0.3)

        declared_min = config.get('min')
        declared_max = config.get('max')
        if declared_min is not None and declared_max is not None:
            sensitivity = float(declared_max) - float(declared_min)
        else:
            default_sens = config.get('sensitivity', 1000.0)
            stats = self.metadata.get(column_name, {}).get('numeric_stats', {})
            discovered_range = stats.get('range', default_sens)
            
            # Same utility heuristic as numeric_noise — use 10% of the range for realistic jitter
            heuristic_scale = 0.1
            sensitivity = (discovered_range * heuristic_scale) if discovered_range > 0 else default_sens
            
        noisy = add_scaled_laplace_noise(
            value=value,
            sensitivity=sensitivity,
            epsilon=epsilon,
            scale_factor=1.0,
        )

        # Robust clipping for monetary values
        stats = self.metadata.get(column_name, {}).get('numeric_stats', {})
        clip_min = declared_min if declared_min is not None else stats.get('min')
        clip_max = declared_max if declared_max is not None else stats.get('max')

        # Fallback for non-negative check (default true for monetary usually)
        if clip_min is None and stats.get('all_non_negative', True):
            clip_min = 0.0

        if isinstance(noisy, np.ndarray):
            if clip_min is not None and clip_max is not None:
                return np.clip(noisy, clip_min, clip_max)
            elif clip_min is not None:
                return np.maximum(clip_min, noisy)
            return noisy
        
        res = float(noisy)
        if clip_min is not None: res = max(clip_min, res)
        if clip_max is not None: res = min(clip_max, res)
        return res

    def apply_count_noise(self, value: Any, config: Dict[str, Any], column_name: str = "") -> Any:
        """
        Apply discrete Laplace noise with non-negative protection for counts.
        """
        epsilon = config.get('epsilon', 1.0)

        noisy = add_discrete_laplace_noise(
            value=value,
            sensitivity=1,
            epsilon=epsilon
        )
        
        # Robust clipping for counts
        stats = self.metadata.get(column_name, {}).get('numeric_stats', {})
        clip_min = config.get('min') if config.get('min') is not None else stats.get('min', 0)
        clip_max = config.get('max') if config.get('max') is not None else stats.get('max')

        if isinstance(noisy, np.ndarray):
            if clip_max is not None:
                return np.clip(noisy, clip_min, clip_max).astype(int)
            return np.maximum(clip_min, noisy).astype(int)
            
        res = int(noisy)
        res = max(int(clip_min), res)
        if clip_max is not None: res = min(int(clip_max), res)
        return res

    def apply_boolean_noise(self, value: Any, config: Dict[str, Any], column_name: str = "") -> Any:
        """
        Apply randomized response to boolean values. Supports vectorized input.
        Uses a per-column seeded RNG for reproducibility.
        """
        epsilon = config.get('epsilon', 0.5)
        p = math.exp(epsilon) / (math.exp(epsilon) + 1)
        rng = get_column_rng(column_name)

        if isinstance(value, (list, np.ndarray)):
            arr = np.array(value, dtype=bool)
            mask = rng.random(len(arr)) < p
            return np.where(mask, arr, ~arr)
        else:
            bool_value = bool(value) if isinstance(value, (int, str)) else value
            if rng.random() < p:
                return bool_value
            else:
                return not bool_value

    def apply_string_masking(self, value: Any, config: Dict[str, Any], column_name: str = "") -> Any:
        """
        Apply string protection: either Randomized Response (categories) or Hashing/Masking (PII).
        """
        epsilon = config.get('epsilon', 0.5)

        meta = self.metadata.get(column_name, {})
        unique_vals = meta.get('unique_values', [])
        unique_count = len(unique_vals)

        # Strategy 1: Categorical Swapping (Best for Gender, City, etc.)
        if 1 < unique_count <= 15:
            return self.apply_categorical_noise(value, unique_vals, epsilon, column_name)

        # Strategy 2: Hashing/masking
        if isinstance(value, (list, np.ndarray)):
            return [self._mask_single_string(str(v), config) for v in value]
        return self._mask_single_string(str(value), config)

    def apply_categorical_noise(self, value: Any, categories: List[str], epsilon: float, column_name: str = "") -> Any:
        """
        Apply generalized Randomized Response to categorical strings.
        This provides formal DP while keeping the strings clean and readable.
        Uses a per-column seeded RNG for reproducibility.
        """
        k = len(categories)
        if k <= 1:
            return value

        p_stay = math.exp(epsilon) / (math.exp(epsilon) + k - 1)
        rng = get_column_rng(column_name)

        if isinstance(value, (list, np.ndarray)):
            results = []
            for v in value:
                if rng.random() < p_stay:
                    results.append(str(v))
                else:
                    others = [c for c in categories if str(c) != str(v)]
                    results.append(rng.choice(others) if others else str(v))
            return results
        else:
            if rng.random() < p_stay:
                return str(value)
            else:
                others = [c for c in categories if str(c) != str(value)]
                return rng.choice(others) if others else str(value)

    def _mask_single_string(self, value: str, config: Dict[str, Any]) -> str:
        """Mask a single string.  ID columns use HMAC-SHA256 (not raw MD5).

        Raw MD5 is deterministic and reversible via rainbow tables.
        HMAC-SHA256 with a per-session secret key ensures:
        - The same plaintext maps to the same token WITHIN a run
          (so foreign-key relationships stay consistent)
        - But no two sessions share the same mapping
          (so pre-computed rainbow tables from one run are useless for another)
        """
        mask_type = config.get('mask_type', 'partial')

        if not value:
            return value

        if mask_type == 'full':
            return '*' * 8
        elif mask_type == 'hash':
            # HMAC-SHA256 with per-session key; truncate to 12 hex chars (48 bits)
            digest = hmac.new(_SESSION_HMAC_KEY, value.encode('utf-8'), hashlib.sha256).hexdigest()
            return digest[:12]
        else:  # partial
            if len(value) <= 4:
                return '*' * len(value)
            return value[:2] + '*' * (len(value) - 4) + value[-2:]

    def apply_mechanism(self, column_name: str, value: Any, column_type: str, config: Dict[str, Any]) -> Any:
        """
        Apply appropriate DP mechanism. Now supports vectorized input natively.
        """
        try:
            if column_type == 'age':
                return self.apply_age_noise(value, config)
            elif column_type == 'year':
                return self.apply_year_noise(value, config)
            elif column_type == 'monetary':
                return self.apply_monetary_noise(value, config, column_name)
            elif column_type == 'numeric':
                return self.apply_numeric_noise(value, config, column_name)
            elif column_type == 'count':
                return self.apply_count_noise(value, config, column_name)
            elif column_type == 'boolean':
                return self.apply_boolean_noise(value, config, column_name)
            elif column_type == 'id':
                config['mask_type'] = 'hash'
                return self.apply_string_masking(value, config)
            elif column_type == 'string':
                return self.apply_string_masking(value, config, column_name)
            else:
                return value
        except Exception as e:
            warnings.warn(f"Failed to apply DP to {column_name}: {e}. Using original value.")
            return value


def infer_column_type(column_name: str, sample_values: list) -> tuple[str, dict]:
    """
    Robustly infer column type and return metadata.
    """
    if not sample_values:
        return 'string', {}

    column_name_lower = column_name.lower()

    # Clean and analyze the sample values
    cleaned_values = []
    for val in sample_values[:200]:
        if val is not None and str(val).strip() != '':
            cleaned_values.append(val)

    if not cleaned_values:
        return 'string', {}

    # Analyze value types and patterns
    analysis = _analyze_value_patterns(cleaned_values)

    # Use statistical patterns to determine type
    col_type = _determine_type_from_analysis(column_name_lower, analysis)
    
    return col_type, analysis


def _analyze_value_patterns(values: list) -> dict:
    """
    Analyze patterns in the values to determine their characteristics.

    Returns:
        Dict with analysis results
    """
    total_count = len(values)

    # Count different types
    type_counts = {
        'numeric': 0,
        'boolean': 0,
        'string': 0,
        'null': 0
    }

    numeric_values = []
    unique_values = set()
    string_lengths = []

    for value in values:
        str_val = str(value).strip()
        unique_values.add(str_val)

        # Check for null/empty
        if not str_val or str_val.lower() in ['null', 'none', 'nan', 'n/a']:
            type_counts['null'] += 1
            continue

        # Check boolean patterns
        if str_val.lower() in ['true', 'false', '1', '0', 'yes', 'no', 'y', 'n']:
            type_counts['boolean'] += 1
            continue

        # Check numeric patterns
        if _is_numeric_value(str_val):
            type_counts['numeric'] += 1
            try:
                num_val = float(str_val)
                numeric_values.append(num_val)
            except (ValueError, TypeError):
                pass
            continue

        # Must be string
        type_counts['string'] += 1
        string_lengths.append(len(str_val))

    # Calculate ratios
    ratios = {k: v / total_count for k, v in type_counts.items() if k != 'null'}

    # Analyze numeric values if we have them
    numeric_stats = {}
    if numeric_values:
        numeric_stats = {
            'min': min(numeric_values),
            'max': max(numeric_values),
            'mean': sum(numeric_values) / len(numeric_values),
            'is_integer': all(v == int(v) for v in numeric_values),
            'range': max(numeric_values) - min(numeric_values),
            'unique_ratio': len(set(numeric_values)) / len(numeric_values),
            'all_non_negative': all(v >= 0 for v in numeric_values)
        }

    return {
        'ratios': ratios,
        'numeric_stats': numeric_stats,
        'unique_ratio': len(unique_values) / total_count,
        'unique_values': list(unique_values),
        'avg_string_length': sum(string_lengths) / len(string_lengths) if string_lengths else 0,
        'total_count': total_count
    }


def _determine_type_from_analysis(column_name: str, analysis: dict) -> str:
    """
    Determine column type based on statistical analysis and column name hints.
    """
    ratios = analysis['ratios']
    numeric_stats = analysis['numeric_stats']

    # High confidence boolean detection
    if ratios.get('boolean', 0) > 0.8:
        return 'boolean'

    # High confidence numeric detection
    if ratios.get('numeric', 0) > 0.9 and numeric_stats:  # Increased threshold for high confidence
        return _classify_numeric_type(column_name, numeric_stats)

    # Mixed data or low-confidence numeric
    if ratios.get('numeric', 0) > 0.5 and numeric_stats:
        # Some numeric content, try to classify
        return _classify_numeric_type(column_name, numeric_stats)
    elif ratios.get('string', 0) > 0.5:
        # Mostly strings
        return 'string'
    elif ratios.get('numeric', 0) > 0.3:
        # Mixed numeric/string, default to string unless strong name hints
        name_hints = ['age', 'year', 'count', 'price', 'cost', 'size', 'consumption', 'emission']
        if any(hint in column_name for hint in name_hints):
            return _classify_numeric_type(column_name, numeric_stats)
        else:
            return 'string'
    else:
        return 'string'


def _classify_numeric_type(column_name: str, stats: dict) -> str:
    """
    Classify numeric data into specific types based on patterns and name hints.
    """
    is_integer = stats['is_integer']
    min_val = stats['min']
    max_val = stats['max']
    range_val = stats['range']
    unique_ratio = stats['unique_ratio']

    # Name-based hints (high priority)
    name_hints = {
        'age': ['age', 'birth'],
        'year': ['year'],
        'count': ['count', 'number', 'num_', 'total', 'cylinder', 'smog', 'level', 'login', 'visit', 'click', 'score'],
        'monetary': ['price', 'cost', 'salary', 'income', 'amount', 'purchase', 'payment'],
        'id': ['id', 'guid', 'uid', 'identifier', 'key', 'index', 'pk'],
        'numeric': ['size', 'consumption', 'emission', 'co2', 'fuel', 'percentage', 'rate', 'ratio', 'co2_emissions']
    }

    import re
    for type_name, keywords in name_hints.items():
        for keyword in keywords:
            # Use word boundaries to avoid substring matches
            if re.search(r'\b' + re.escape(keyword) + r'\b', column_name.lower()):
                return type_name

    # Statistical pattern analysis
    if is_integer:
        # Integer analysis
        if 1900 <= min_val and max_val <= 2100:
            # Year range
            return 'year'
        elif 0 <= min_val and max_val <= 150 and range_val < 100:
            # Small range integers - could be age, count, or scores
            if 'age' in column_name or 'birth' in column_name:
                return 'age'
            elif max_val <= 12 or unique_ratio < 0.5:  # Few unique values or small max
                return 'count'
            elif max_val <= 100:  # Scores, percentages as integers
                return 'count'
            else:
                return 'age'  # Default for small integers
        elif max_val <= 50 and unique_ratio < 0.3:
            # Low unique ratio, likely categories/counts
            return 'count'
        else:
            if 'id' in column_name or 'pk' in column_name:
                return 'id'
            # Large integers, likely counts or IDs
            return 'count'
    else:
        # Float analysis
        if range_val > 1000 and ('price' in column_name or 'cost' in column_name or 'amount' in column_name):
            return 'monetary'
        elif min_val >= 0 and max_val <= 100 and ('percentage' in column_name or 'percent' in column_name):
            # Percentages are typically 0-100
            return 'numeric'
        elif not is_integer:  # If it's truly a float (not just integer represented as float)
            return 'numeric'
        elif range_val > 10:  # Has some spread, likely continuous measurement
            return 'numeric'
        else:
            # Small range floats, could be rates or ratios
            return 'numeric'