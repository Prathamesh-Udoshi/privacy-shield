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


def simulate_membership_inference(original_data: Dict[str, List],
                                noisy_data: Dict[str, List],
                                columns: List[str]) -> Dict[str, Any]:
    """
    Simulate a membership inference attack using distance-based matching.
    
    This estimates how many original records can be correctly linked to their 
    anonymized versions based on statistical proximity.
    
    Args:
        original_data: Original column data
        noisy_data: Noisy column data
        columns: Columns to use for the attack
        
    Returns:
        Dict with simulation results
    """
    import numpy as np
    
    available_cols = [c for c in columns if c in original_data and c in noisy_data]
    if not available_cols:
        return {'success_rate': 0.0, 'risk_level': 'LOW', 'details': 'No numeric columns'}
    
    # Process only numeric columns for distance calculation
    numeric_cols = []
    for col in available_cols:
        try:
            # Check if majority of values are numeric
            sample = [v for v in original_data[col][:20] if v is not None]
            if all(isinstance(v, (int, float, complex)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit()) for v in sample):
                numeric_cols.append(col)
        except Exception:
            continue
            
    if not numeric_cols:
        return {'success_rate': 0.0, 'risk_level': 'LOW', 'details': 'No numeric columns found'}
        
    # Build matrices for comparison
    def build_matrix(data_dict, cols):
        rows = []
        for i in range(len(data_dict[cols[0]])):
            row = []
            for col in cols:
                val = data_dict[col][i]
                try:
                    row.append(float(val) if val is not None and str(val).strip() != '' else 0.0)
                except (ValueError, TypeError):
                    row.append(0.0)
            rows.append(row)
        return np.array(rows)
    
    orig_matrix = build_matrix(original_data, numeric_cols)
    noisy_matrix = build_matrix(noisy_data, numeric_cols)
    
    # Normalize matrices for fair distance calculation
    col_max = np.max(np.abs(orig_matrix), axis=0)
    col_max = np.where(col_max == 0, 1.0, col_max) # Avoid division by zero
    
    orig_norm = orig_matrix / col_max
    noisy_norm = noisy_matrix / col_max
    
    # Simple membership inference: for each noisy record, find the closest original record
    # If the closest original is the TRUE original, the attack succeeds
    success_count = 0
    num_to_test = min(len(orig_matrix), 500) # Sample to avoid O(N^2) on large datasets
    
    for i in range(num_to_test):
        # Calculate distances from noisy_norm[i] to all original records
        distances = np.linalg.norm(orig_norm - noisy_norm[i], axis=1)
        # Find index of closest record
        closest_idx = np.argmin(distances)
        
        if closest_idx == i:
            success_count += 1
            
    success_rate = (success_count / num_to_test) * 100
    
    if success_rate > 40:
        risk_level = "HIGH"
    elif success_rate > 15:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
        
    return {
        'success_rate': success_rate,
        'risk_level': risk_level,
        'attack_type': 'Nearest-Neighbor Linking',
        'columns_exploited': len(numeric_cols)
    }


def generate_risk_report(original_data: Dict[str, List],
                        noisy_data: Dict[str, List],
                        quasi_identifiers: List[str] = None) -> str:
    """
    Generate a comprehensive re-identification risk report with membership simulation.
    """
    report_lines = [
        "Re-identification Risk Assessment",
        "=" * 50
    ]

    # Default quasi-identifiers
    if quasi_identifiers is None:
        quasi_identifiers = ['age', 'location', 'gender', 'occupation']

    available_qi = [col for col in quasi_identifiers if col in original_data and col in noisy_data]
    all_columns = list(original_data.keys())
    
    # 1. Uniqueness analysis
    uniqueness_reduction = calculate_uniqueness_reduction(original_data, noisy_data, all_columns)
    
    # 2. K-Anonymity simulation
    noisy_k_analysis = {'k_estimate': 0, 'anonymity_level': 'UNKNOWN', 'risk_assessment': 'UNKNOWN'}
    if available_qi:
        noisy_combinations = list(zip(*[noisy_data[col] for col in available_qi]))
        noisy_k_analysis = estimate_k_anonymity(noisy_combinations)

    # 3. Membership Inference Simulation (NEW)
    mi_results = simulate_membership_inference(original_data, noisy_data, all_columns)

    report_lines.extend([
        f"Uniqueness Reduction:  {uniqueness_reduction:.1f}%",
        f"Membership Inference: {mi_results['success_rate']:.1f}% link success rate",
        f"Risk Level (Linking): {mi_results['risk_level']}",
        ""
    ])

    if available_qi:
        report_lines.extend([
            f"K-Anonymity Analysis (using: {', '.join(available_qi)}):",
            f"  Noisy Data: k={noisy_k_analysis['k_estimate']} ({noisy_k_analysis['anonymity_level']})",
            ""
        ])

    # Overall risk determination
    risk_score = 0
    
    # Factor 1: Membership Inference (The most realistic empirical attack)
    if mi_results['risk_level'] == "HIGH": 
        risk_score += 4
    elif mi_results['risk_level'] == "MEDIUM": 
        risk_score += 2
    
    # Factor 2: K-Anonymity (Harder to achieve with continuous noise)
    if noisy_k_analysis['risk_assessment'] == "HIGH":
        # Only penalize heavily if the dataset is large enough that k>1 should be expected
        if len(all_columns) > 0 and len(original_data[all_columns[0]]) > 100:
            risk_score += 2
        else:
            risk_score += 1 # Minor penalty for small datasets
    elif noisy_k_analysis['risk_assessment'] == "MEDIUM":
        risk_score += 1
    
    # Factor 3: Uniqueness (Continuous noise rarely reduces uniqueness)
    if uniqueness_reduction < 5: 
        risk_score += 1

    # Determination with higher thresholds
    if risk_score >= 6:
        overall_risk = "CRITICAL"
    elif risk_score >= 3:
        overall_risk = "MODERATE"
    else:
        overall_risk = "LOW"

    report_lines.extend([
        "Final Privacy Assessment:",
        f"  Overall Risk Category: {overall_risk}",
        ""
    ])

    if overall_risk == "LOW":
        report_lines.append("  Interpretation: Strong protection. Matches are likely statistical coincidences.")
    elif overall_risk == "MODERATE":
        report_lines.append("  Interpretation: Probabilistic privacy. Some records may be linkable by determined attackers.")
    else:
        report_lines.append("  Interpretation: High Linkage! Noise is too low relative to data density; records remain distinct.")

    return "\n".join(report_lines)
