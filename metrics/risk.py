"""
Re-identification risk estimation for anonymized data.

This module provides heuristic measures of re-identification risk
based on uniqueness analysis and outlier preservation.
"""

from typing import List, Dict, Any, Tuple
from collections import Counter
import math


def calculate_uniqueness_score(values: List[Any]) -> float:
    """
    Calculate the uniqueness score of a dataset.

    Higher score means more unique values (higher re-identification risk).

    Args:
        values: List of values

    Returns:
        Uniqueness score between 0 and 1
    """
    if not values:
        return 0

    total_count = len(values)
    unique_count = len(set(values))

    return unique_count / total_count


def estimate_k_anonymity(values: List[Tuple], k_threshold: int = 5) -> Dict[str, Any]:
    """
    Estimate k-anonymity based on quasi-identifier combinations.

    Args:
        values: List of tuples representing quasi-identifiers
        k_threshold: Minimum k for anonymity

    Returns:
        Dict with k_estimate, anonymity_level, and risk_assessment
    """
    if not values:
        return {'k_estimate': 0, 'anonymity_level': 'UNKNOWN', 'risk_assessment': 'UNKNOWN'}

    # Count frequency of each combination
    combination_counts = Counter(values)

    # Find minimum frequency (this gives us the k value)
    min_frequency = min(combination_counts.values())
    max_frequency = max(combination_counts.values())

    # Estimate overall k-anonymity as the minimum frequency
    k_estimate = min_frequency

    # Assess anonymity level
    if k_estimate >= k_threshold:
        anonymity_level = "ANONYMOUS"
        risk_assessment = "LOW"
    elif k_estimate >= 2:
        anonymity_level = "WEAKLY_ANONYMOUS"
        risk_assessment = "MEDIUM"
    else:
        anonymity_level = "NOT_ANONYMOUS"
        risk_assessment = "HIGH"

    return {
        'k_estimate': k_estimate,
        'anonymity_level': anonymity_level,
        'risk_assessment': risk_assessment,
        'max_frequency': max_frequency,
        'unique_combinations': len(combination_counts)
    }


def detect_outlier_preservation(original: List[float], noisy: List[float],
                              outlier_threshold: float = 2.0) -> Dict[str, Any]:
    """
    Detect if outliers are preserved in noisy data.

    Outliers could be re-identification risks if they remain identifiable.

    Args:
        original: Original numeric values
        noisy: Noisy values
        outlier_threshold: Z-score threshold for outlier detection

    Returns:
        Dict with outlier statistics
    """
    if len(original) < 3 or len(noisy) < 3:
        return {'outlier_risk': 'UNKNOWN', 'details': 'Insufficient data'}

    # Calculate statistics for original data
    original_mean = sum(original) / len(original)
    original_std = math.sqrt(sum((x - original_mean) ** 2 for x in original) / len(original))

    # Find outliers in original data
    original_outliers = []
    for i, val in enumerate(original):
        if original_std > 0:
            z_score = abs(val - original_mean) / original_std
            if z_score > outlier_threshold:
                original_outliers.append((i, val, z_score))

    # Check if these outliers are still identifiable in noisy data
    preserved_outliers = 0
    for idx, orig_val, orig_z in original_outliers:
        noisy_val = noisy[idx]
        # Check if noisy value is still an outlier
        noisy_z = abs(noisy_val - original_mean) / original_std if original_std > 0 else 0
        if noisy_z > outlier_threshold:
            preserved_outliers += 1

    outlier_preservation_rate = preserved_outliers / len(original_outliers) if original_outliers else 0

    if outlier_preservation_rate > 0.5:
        outlier_risk = "HIGH"
    elif outlier_preservation_rate > 0.2:
        outlier_risk = "MEDIUM"
    else:
        outlier_risk = "LOW"

    return {
        'outlier_risk': outlier_risk,
        'original_outliers': len(original_outliers),
        'preserved_outliers': preserved_outliers,
        'preservation_rate': outlier_preservation_rate,
        'total_records': len(original)
    }


def calculate_uniqueness_reduction(original_data: Dict[str, List],
                                 noisy_data: Dict[str, List],
                                 columns: List[str]) -> float:
    """
    Calculate the reduction in uniqueness after anonymization.

    Args:
        original_data: Original column data
        noisy_data: Noisy column data
        columns: Columns to analyze

    Returns:
        Percentage reduction in uniqueness (0-100)
    """
    if not columns:
        return 0

    total_original_uniqueness = 0
    total_noisy_uniqueness = 0
    analyzed_columns = 0

    for col in columns:
        if col not in original_data or col not in noisy_data:
            continue

        orig_vals = original_data[col]
        noisy_vals = noisy_data[col]

        if not orig_vals or not noisy_vals:
            continue

        orig_uniqueness = calculate_uniqueness_score(orig_vals)
        noisy_uniqueness = calculate_uniqueness_score(noisy_vals)

        total_original_uniqueness += orig_uniqueness
        total_noisy_uniqueness += noisy_uniqueness
        analyzed_columns += 1

    if analyzed_columns == 0 or total_original_uniqueness == 0:
        return 0

    avg_original_uniqueness = total_original_uniqueness / analyzed_columns
    avg_noisy_uniqueness = total_noisy_uniqueness / analyzed_columns

    reduction = (avg_original_uniqueness - avg_noisy_uniqueness) / avg_original_uniqueness * 100
    return max(0, reduction)


def generate_risk_report(original_data: Dict[str, List],
                        noisy_data: Dict[str, List],
                        quasi_identifiers: List[str] = None) -> str:
    """
    Generate a comprehensive re-identification risk report.

    Args:
        original_data: Dict mapping column names to original values
        noisy_data: Dict mapping column names to noisy values
        quasi_identifiers: List of column names that could be quasi-identifiers

    Returns:
        Formatted risk report string
    """
    report_lines = [
        "Re-identification Risk Assessment",
        "=" * 50
    ]

    # Default quasi-identifiers if not specified
    if quasi_identifiers is None:
        quasi_identifiers = ['age', 'location', 'gender', 'occupation']

    # Find available quasi-identifier columns
    available_qi = [col for col in quasi_identifiers if col in original_data and col in noisy_data]

    # Uniqueness reduction analysis
    all_columns = list(original_data.keys())
    uniqueness_reduction = calculate_uniqueness_reduction(original_data, noisy_data, all_columns)

    report_lines.extend([
        f"Uniqueness Reduction: {uniqueness_reduction:.1f}%",
        ""
    ])

    # K-anonymity analysis
    if available_qi:
        # Create quasi-identifier combinations for original and noisy data
        orig_combinations = list(zip(*[original_data[col] for col in available_qi]))
        noisy_combinations = list(zip(*[noisy_data[col] for col in available_qi]))

        orig_k_analysis = estimate_k_anonymity(orig_combinations)
        noisy_k_analysis = estimate_k_anonymity(noisy_combinations)

        report_lines.extend([
            f"K-Anonymity Analysis (using: {', '.join(available_qi)}):",
            f"  Original Data: k={orig_k_analysis['k_estimate']} ({orig_k_analysis['anonymity_level']})",
            f"  Noisy Data:    k={noisy_k_analysis['k_estimate']} ({noisy_k_analysis['anonymity_level']})",
            ""
        ])

        # Overall risk assessment
        risk_factors = []

        # Factor 1: Uniqueness reduction
        if uniqueness_reduction < 10:
            risk_factors.append("low_uniqueness_reduction")
        elif uniqueness_reduction < 30:
            risk_factors.append("moderate_uniqueness_reduction")
        else:
            risk_factors.append("high_uniqueness_reduction")

        # Factor 2: K-anonymity
        if noisy_k_analysis['risk_assessment'] == 'HIGH':
            risk_factors.append("low_k_anonymity")
        elif noisy_k_analysis['risk_assessment'] == 'MEDIUM':
            risk_factors.append("moderate_k_anonymity")

        # Factor 3: Outlier analysis for numeric columns
        numeric_columns = [col for col in all_columns if col in original_data and
                          any(isinstance(v, (int, float)) for v in original_data[col][:10])]

        high_outlier_risk = False
        for col in numeric_columns:
            orig_vals = [v for v in original_data[col] if isinstance(v, (int, float))]
            noisy_vals = [v for v in noisy_data[col] if isinstance(v, (int, float))]

            if len(orig_vals) >= 10:  # Need minimum sample size
                outlier_analysis = detect_outlier_preservation(orig_vals, noisy_vals)
                if outlier_analysis['outlier_risk'] == 'HIGH':
                    high_outlier_risk = True
                    break

        if high_outlier_risk:
            risk_factors.append("outlier_preservation")

        # Determine overall risk level
        risk_weights = {
            'low_uniqueness_reduction': 0,
            'moderate_uniqueness_reduction': 1,
            'high_uniqueness_reduction': 2,
            'moderate_k_anonymity': 1,
            'low_k_anonymity': 3,
            'outlier_preservation': 2
        }

        total_risk_score = sum(risk_weights.get(factor, 0) for factor in risk_factors)

        if total_risk_score <= 1:
            overall_risk = "LOW"
        elif total_risk_score <= 3:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "HIGH"

        report_lines.extend([
            "Overall Risk Assessment:",
            f"  Risk Level: {overall_risk}",
            f"  Risk Factors: {', '.join(risk_factors) if risk_factors else 'None'}"
        ])

        if overall_risk == "LOW":
            report_lines.append("  Interpretation: Low risk of re-identification")
        elif overall_risk == "MEDIUM":
            report_lines.append("  Interpretation: Moderate re-identification risk - additional protections recommended")
        else:
            report_lines.append("  Interpretation: High re-identification risk - review anonymization parameters")

    else:
        report_lines.extend([
            "Limited Analysis: No quasi-identifier columns found for k-anonymity analysis",
            "Overall Risk Assessment: UNKNOWN"
        ])

    return "\n".join(report_lines)