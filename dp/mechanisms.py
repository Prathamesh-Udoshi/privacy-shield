"""
Differential privacy mechanisms for different data types.

This module provides column-aware noise addition strategies
for various data types commonly found in user activity data.
"""

import math
import random
import warnings
import numpy as np
from typing import Union, Dict, Any, Optional, List
from .laplace import (
    add_laplace_noise,
    add_bounded_laplace_noise,
    add_discrete_laplace_noise,
    add_scaled_laplace_noise
)
from .budget import PrivacyBudget


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

    def __init__(self, budget: PrivacyBudget):
        """
        Initialize with a privacy budget.

        Args:
            budget: PrivacyBudget instance to track epsilon consumption
        """
        self.budget = budget

    def apply_age_noise(self, value: Any, config: Dict[str, Any]) -> Any:
        """
        Apply bounded Laplace noise to age values. Supports vectorized input.
        """
        epsilon = config.get('epsilon', 0.2)
        min_age = config.get('min', 0)
        max_age = config.get('max', 120)

        return add_bounded_laplace_noise(
            value=value,
            sensitivity=1.0,
            epsilon=epsilon,
            min_val=min_age,
            max_val=max_age
        )

    def apply_year_noise(self, value: Any, config: Dict[str, Any]) -> Any:
        """
        Apply bounded Laplace noise to year values. Supports vectorized input.
        """
        epsilon = config.get('epsilon', 0.2)
        min_year = config.get('min', 1900)
        max_year = config.get('max', 2050)

        return add_bounded_laplace_noise(
            value=value,
            sensitivity=1.0,
            epsilon=epsilon,
            min_val=min_year,
            max_val=max_year
        )

    def apply_numeric_noise(self, value: Any, config: Dict[str, Any]) -> Any:
        """
        Apply Laplace noise to continuous numeric values. Supports vectorized input.
        """
        epsilon = config.get('epsilon', 0.3)
        sensitivity = config.get('sensitivity', 1.0)

        return add_laplace_noise(
            value=value,
            sensitivity=sensitivity,
            epsilon=epsilon
        )

    def apply_monetary_noise(self, value: Any, config: Dict[str, Any]) -> Any:
        """
        Apply scaled Laplace noise to monetary values. Supports vectorized input.
        """
        epsilon = config.get('epsilon', 0.3)
        sensitivity = config.get('sensitivity', 1000.0)

        return add_scaled_laplace_noise(
            value=value,
            sensitivity=sensitivity,
            epsilon=epsilon,
            scale_factor=1.0
        )

    def apply_count_noise(self, value: Any, config: Dict[str, Any]) -> Any:
        """
        Apply discrete Laplace noise to count values. Supports vectorized input.
        """
        epsilon = config.get('epsilon', 1.0)

        return add_discrete_laplace_noise(
            value=value,
            sensitivity=1,
            epsilon=epsilon
        )

    def apply_boolean_noise(self, value: Any, config: Dict[str, Any]) -> Any:
        """
        Apply randomized response to boolean values. Supports vectorized input.
        """
        epsilon = config.get('epsilon', 0.5)
        p = math.exp(epsilon) / (math.exp(epsilon) + 1)

        if isinstance(value, (list, np.ndarray)):
            arr = np.array(value, dtype=bool)
            # Generate random mask: True with probability p
            mask = np.random.random(len(arr)) < p
            # Keep value where mask is True, flip where False
            return np.where(mask, arr, ~arr)
        else:
            bool_value = bool(value) if isinstance(value, (int, str)) else value
            if random.random() < p:
                return bool_value
            else:
                return not bool_value

    def apply_string_masking(self, value: Any, config: Dict[str, Any]) -> Any:
        """
        Apply string masking/hashing. Vectorized helper for lists.
        """
        if isinstance(value, (list, np.ndarray)):
            return [self._mask_single_string(str(v), config) for v in value]
        return self._mask_single_string(str(value), config)

    def _mask_single_string(self, value: str, config: Dict[str, Any]) -> str:
        """Internal helper for single string masking."""
        mask_type = config.get('mask_type', 'partial')
        if mask_type == 'partial':
            if len(value) <= 2:
                return '*' * len(value)
            return value[0] + '*' * (len(value) - 2) + value[-1]
        elif mask_type == 'hash':
            import hashlib
            return hashlib.md5(value.encode()).hexdigest()[:8]
        return value

    def apply_mechanism(self, column_name: str, value: Any,
                       column_type: str, config: Dict[str, Any]) -> Any:
        """
        Apply appropriate DP mechanism. Now supports vectorized input natively.
        """
        try:
            if column_type == 'age':
                return self.apply_age_noise(value, config)
            elif column_type == 'year':
                return self.apply_year_noise(value, config)
            elif column_type == 'monetary':
                return self.apply_monetary_noise(value, config)
            elif column_type == 'numeric':
                return self.apply_numeric_noise(value, config)
            elif column_type == 'count':
                return self.apply_count_noise(value, config)
            elif column_type == 'boolean':
                return self.apply_boolean_noise(value, config)
            elif column_type == 'string':
                return self.apply_string_masking(value, config)
            else:
                return value
        except Exception as e:
            warnings.warn(f"Failed to apply DP to {column_name}: {e}. Using original value.")
            return value


def infer_column_type(column_name: str, sample_values: list) -> str:
    """
    Robustly infer column type based on statistical analysis of values,
    with name-based hints as tie-breakers.

    Args:
        column_name: Name of the column
        sample_values: List of sample values from the column

    Returns:
        Inferred column type: 'age', 'year', 'monetary', 'numeric', 'count', 'boolean', 'string'
    """
    if not sample_values:
        return 'string'

    column_name_lower = column_name.lower()

    # Clean and analyze the sample values
    cleaned_values = []
    for val in sample_values[:200]:  # Use more samples for better analysis
        if val is not None and str(val).strip() != '':
            cleaned_values.append(val)

    if not cleaned_values:
        return 'string'

    # Analyze value types and patterns
    analysis = _analyze_value_patterns(cleaned_values)

    # Use statistical patterns to determine type
    return _determine_type_from_analysis(column_name_lower, analysis)


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
            'unique_ratio': len(set(numeric_values)) / len(numeric_values)
        }

    return {
        'ratios': ratios,
        'numeric_stats': numeric_stats,
        'unique_ratio': len(unique_values) / total_count,
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