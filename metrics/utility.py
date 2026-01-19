"""
Utility metrics for evaluating differential privacy anonymization quality.

This module computes metrics to measure how well the anonymization
preserves aggregate statistics while providing privacy.
"""

import math
from typing import List, Dict, Any, Tuple
from statistics import mean, stdev


def calculate_mean_preservation(original: List[float], noisy: List[float]) -> Dict[str, float]:
    """
    Calculate how well the mean is preserved after adding noise.

    Args:
        original: List of original values
        noisy: List of noisy values

    Returns:
        Dict with original_mean, noisy_mean, absolute_error, relative_error
    """
    if not original or not noisy:
        return {'original_mean': 0, 'noisy_mean': 0, 'absolute_error': 0, 'relative_error': 0}

    original_mean = mean(original)
    noisy_mean = mean(noisy)

    absolute_error = abs(original_mean - noisy_mean)
    relative_error = absolute_error / abs(original_mean) if original_mean != 0 else 0

    return {
        'original_mean': original_mean,
        'noisy_mean': noisy_mean,
        'absolute_error': absolute_error,
        'relative_error': relative_error
    }


def calculate_std_dev_change(original: List[float], noisy: List[float]) -> Dict[str, float]:
    """
    Calculate the change in standard deviation.

    Args:
        original: List of original values
        noisy: List of noisy values

    Returns:
        Dict with original_std, noisy_std, absolute_change, percentage_change
    """
    try:
        original_std = stdev(original)
        noisy_std = stdev(noisy)

        absolute_change = noisy_std - original_std
        percentage_change = (absolute_change / original_std * 100) if original_std != 0 else 0

        return {
            'original_std': original_std,
            'noisy_std': noisy_std,
            'absolute_change': absolute_change,
            'percentage_change': percentage_change
        }
    except:
        # Handle cases where stdev cannot be calculated (e.g., single value)
        return {
            'original_std': 0,
            'noisy_std': 0,
            'absolute_change': 0,
            'percentage_change': 0
        }


def calculate_mean_absolute_error(original: List[float], noisy: List[float]) -> float:
    """
    Calculate Mean Absolute Error between original and noisy values.

    Args:
        original: List of original values
        noisy: List of noisy values

    Returns:
        Mean absolute error
    """
    if len(original) != len(noisy):
        raise ValueError("Original and noisy lists must have the same length")

    if not original:
        return 0

    errors = [abs(o - n) for o, n in zip(original, noisy)]
    return mean(errors)


def calculate_utility_score(mean_preservation: Dict[str, float],
                          std_change: Dict[str, float],
                          mae: float,
                          max_expected_error: float = 1.0) -> float:
    """
    Calculate an overall utility score (0-100) based on multiple metrics.

    Higher score = better utility preservation.

    Args:
        mean_preservation: Result from calculate_mean_preservation
        std_change: Result from calculate_std_dev_change
        mae: Mean absolute error
        max_expected_error: Maximum expected error for normalization

    Returns:
        Utility score between 0 and 100
    """
    # Component scores (0-1 scale, higher is better)
    mean_score = max(0, 1 - mean_preservation['relative_error'])
    std_score = max(0, 1 - abs(std_change['percentage_change']) / 50.0)  # Allow up to 50% std change
    mae_score = max(0, 1 - mae / max_expected_error)

    # Weighted average (mean preservation is most important)
    overall_score = (0.5 * mean_score + 0.3 * std_score + 0.2 * mae_score) * 100

    return min(100, max(0, overall_score))


def generate_utility_report(original_data: Dict[str, List],
                          noisy_data: Dict[str, List],
                          numeric_columns: List[str]) -> str:
    """
    Generate a comprehensive utility report.

    Args:
        original_data: Dict mapping column names to lists of original values
        noisy_data: Dict mapping column names to lists of noisy values
        numeric_columns: List of numeric column names to analyze

    Returns:
        Formatted utility report string
    """
    report_lines = [
        "Utility Preservation Report",
        "=" * 50
    ]

    total_utility_score = 0
    analyzed_columns = 0

    for col in numeric_columns:
        if col not in original_data or col not in noisy_data:
            continue

        original_vals = original_data[col]
        noisy_vals = noisy_data[col]

        # Filter to numeric values only
        original_numeric = [v for v in original_vals if isinstance(v, (int, float)) and not math.isnan(v)]
        noisy_numeric = [v for v in noisy_vals if isinstance(v, (int, float)) and not math.isnan(v)]

        if len(original_numeric) < 2 or len(noisy_numeric) < 2:
            continue

        analyzed_columns += 1

        # Calculate metrics
        mean_metrics = calculate_mean_preservation(original_numeric, noisy_numeric)
        std_metrics = calculate_std_dev_change(original_numeric, noisy_numeric)
        mae = calculate_mean_absolute_error(original_numeric, noisy_numeric)

        # Calculate per-column utility score
        # Estimate max expected error based on data range
        if original_numeric:
            data_range = max(original_numeric) - min(original_numeric)
            max_expected_error = data_range * 0.1 if data_range > 0 else 1.0
        else:
            max_expected_error = 1.0

        utility_score = calculate_utility_score(mean_metrics, std_metrics, mae, max_expected_error)
        total_utility_score += utility_score

        # Add to report
        report_lines.extend([
            f"\nColumn: {col}",
            f"  Sample Size: {len(original_numeric)}",
            f"  Mean Preservation:",
            f"    Original: {mean_metrics['original_mean']:.3f}",
            f"    Noisy:    {mean_metrics['noisy_mean']:.3f}",
            f"    Error:    {mean_metrics['absolute_error']:.3f} ({mean_metrics['relative_error']*100:.1f}%)",
            f"  Std Deviation Change:",
            f"    Original: {std_metrics['original_std']:.3f}",
            f"    Noisy:    {std_metrics['noisy_std']:.3f}",
            f"    Change:   {std_metrics['percentage_change']:+.1f}%",
            f"  Mean Absolute Error: {mae:.3f}",
            f"  Utility Score: {utility_score:.1f}/100"
        ])

    if analyzed_columns > 0:
        avg_utility_score = total_utility_score / analyzed_columns
        report_lines.extend([
            "",
            "Overall Summary:",
            f"  Columns Analyzed: {analyzed_columns}",
            f"  Average Utility Score: {avg_utility_score:.1f}/100"
        ])

        # Interpret the score
        if avg_utility_score >= 80:
            interpretation = "EXCELLENT - Statistical properties well preserved"
        elif avg_utility_score >= 60:
            interpretation = "GOOD - Moderate statistical preservation"
        elif avg_utility_score >= 40:
            interpretation = "FAIR - Some statistical information lost"
        else:
            interpretation = "POOR - Significant statistical distortion"

        report_lines.append(f"  Interpretation: {interpretation}")

    return "\n".join(report_lines)