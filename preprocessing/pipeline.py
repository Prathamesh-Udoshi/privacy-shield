#!/usr/bin/env python3
"""
Enhanced Data Preprocessing Pipeline for Privacy Shield

This module provides sophisticated data cleaning and validation
before differential privacy anonymization.
"""

import math
import statistics
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict
import warnings


class DPImputation:
    """
    Privacy-preserving missing value imputation using differentially private statistics.
    """

    def __init__(self, epsilon_impute: float = 0.1):
        """
        Initialize with privacy budget for imputation.

        Args:
            epsilon_impute: Privacy budget allocated for imputation operations
        """
        self.epsilon_impute = epsilon_impute

    def sample_laplace(self, scale: float) -> float:
        """Sample from Laplace distribution for DP noise."""
        import random
        # Generate uniform random variable in (-0.5, 0.5)
        u = random.uniform(-0.5, 0.5)
        if u == 0:
            return 0.0
        # Apply inverse CDF
        return (1.0 / scale) * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))

    def impute_numeric_column(self, values: List[Union[int, float, None]],
                            epsilon_budget: float) -> Tuple[List[Union[int, float]], Dict[str, Any]]:
        """
        Impute missing values in numeric column using DP statistics.

        Args:
            values: List of numeric values (with None for missing)
            epsilon_budget: Privacy budget for this imputation

        Returns:
            Tuple of (imputed_values, imputation_stats)
        """
        # Separate missing and non-missing values
        # Check for None, NaN, empty strings, and whitespace-only strings
    def is_missing(val):
        if val is None:
            return True
        if isinstance(val, float) and math.isnan(val):
            return True
        if isinstance(val, str):
            stripped = val.strip().lower()
            # Check for empty strings and various null representations
            if stripped == '' or stripped in ['none', 'null', 'nan', 'n/a', 'na']:
                return True
        return False

        non_missing = [v for v in values if not is_missing(v)]

        if not non_missing:
            # All values are missing, return zeros
            return [0.0] * len(values), {"method": "zero_fallback", "missing_count": len(values)}

        missing_indices = [i for i, v in enumerate(values) if is_missing(v)]

        if not missing_indices:
            # No missing values
            return values, {"method": "none", "missing_count": 0}

        # Calculate statistics with DP noise
        try:
            # Convert to float for calculations
            numeric_values = [float(v) for v in non_missing]
            mean_val = statistics.mean(numeric_values)
            # Add DP noise to mean (sensitivity = range / n)
            data_range = max(numeric_values) - min(numeric_values)
            sensitivity = max(data_range / len(numeric_values), 0.1)
            scale = sensitivity / epsilon_budget
            noisy_mean = mean_val + self.sample_laplace(scale)

            # For numeric columns, also consider median for robustness
            median_val = statistics.median(numeric_values)

            # Choose imputation method based on data characteristics
            if len(non_missing) < 10:
                # Small dataset, use median (more robust)
                impute_value = median_val
                method = "dp_median"
            else:
                # Larger dataset, use noisy mean
                impute_value = noisy_mean
                method = "dp_mean"

        except statistics.StatisticsError:
            # Fallback to simple approach
            impute_value = 0.0
            method = "zero_fallback"

        # Apply imputation
        imputed_values = values.copy()
        for idx in missing_indices:
            imputed_values[idx] = impute_value

        imputation_stats = {
            "method": method,
            "missing_count": len(missing_indices),
            "impute_value": impute_value,
            "original_mean": mean_val,
            "data_range": data_range,
            "epsilon_used": epsilon_budget
        }

        return imputed_values, imputation_stats

    def impute_categorical_column(self, values: List[Union[str, None]],
                                epsilon_budget: float) -> Tuple[List[Union[str, None]], Dict[str, Any]]:
        """
        Impute missing values in categorical column using mode with DP.

        Args:
            values: List of categorical values (with None for missing)
            epsilon_budget: Privacy budget for this imputation

        Returns:
            Tuple of (imputed_values, imputation_stats)
        """
        def is_missing(val):
            if val is None:
                return True
            if isinstance(val, float) and math.isnan(val):
                return True
            if isinstance(val, str):
                stripped = val.strip().lower()
                # Check for empty strings and various null representations
                if stripped == '' or stripped in ['none', 'null', 'nan', 'n/a', 'na']:
                    return True
            return False

        # Count frequencies
        value_counts = defaultdict(int)
        missing_count = 0

        for v in values:
            if is_missing(v):
                missing_count += 1
            else:
                value_counts[str(v)] += 1

        if missing_count == 0:
            return values, {"method": "none", "missing_count": 0}

        if not value_counts:
            # All values are missing
            return ["unknown"] * len(values), {"method": "unknown_fallback", "missing_count": missing_count}

        # Find mode (most frequent value)
        mode_value = max(value_counts.keys(), key=lambda k: value_counts[k])

        # For privacy, we could add noise to counts, but for simplicity:
        # Use the mode directly (low sensitivity for categorical imputation)
        impute_value = mode_value

        # Apply imputation
        imputed_values = []
        for v in values:
            if is_missing(v):
                imputed_values.append(impute_value)
            else:
                imputed_values.append(v)

        imputation_stats = {
            "method": "mode",
            "missing_count": missing_count,
            "impute_value": impute_value,
            "unique_values": len(value_counts),
            "mode_frequency": value_counts[mode_value]
        }

        return imputed_values, imputation_stats


class OutlierDetector:
    """
    Statistical outlier detection for data quality assessment.
    """

    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        """
        Initialize outlier detector.

        Args:
            z_threshold: Z-score threshold for outlier detection
            iqr_multiplier: IQR multiplier for robust outlier detection
        """
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier

    def detect_zscore_outliers(self, values: List[Union[int, float]]) -> List[Tuple[int, Union[int, float], float]]:
        """
        Detect outliers using Z-score method.

        Args:
            values: Numeric values to analyze

        Returns:
            List of (index, value, z_score) tuples for outliers
        """
        if len(values) < 3:
            return []

        try:
            mean_val = statistics.mean(values)
            stdev_val = statistics.stdev(values)

            if stdev_val == 0:
                return []  # No variation, no outliers

            outliers = []
            for i, val in enumerate(values):
                z_score = abs(val - mean_val) / stdev_val
                if z_score > self.z_threshold:
                    outliers.append((i, val, z_score))

            return outliers

        except statistics.StatisticsError:
            return []

    def detect_iqr_outliers(self, values: List[Union[int, float]]) -> List[Tuple[int, Union[int, float], str]]:
        """
        Detect outliers using IQR (Interquartile Range) method.

        Args:
            values: Numeric values to analyze

        Returns:
            List of (index, value, outlier_type) tuples for outliers
        """
        if len(values) < 4:
            return []

        try:
            sorted_values = sorted(values)
            n = len(sorted_values)

            # Calculate quartiles
            q1_idx = n // 4
            q3_idx = 3 * n // 4

            q1 = sorted_values[q1_idx]
            q3 = sorted_values[q3_idx]
            iqr = q3 - q1

            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr

            outliers = []
            for i, val in enumerate(values):
                if val < lower_bound:
                    outliers.append((i, val, "low_outlier"))
                elif val > upper_bound:
                    outliers.append((i, val, "high_outlier"))

            return outliers

        except (ValueError, IndexError):
            return []

    def detect_outliers_comprehensive(self, values: List[Union[int, float]],
                                    column_name: str = "unknown") -> Dict[str, Any]:
        """
        Comprehensive outlier detection using multiple methods.

        Args:
            values: Numeric values to analyze
            column_name: Name of the column for reporting

        Returns:
            Dictionary with outlier analysis results
        """
        numeric_values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]

        if len(numeric_values) < 3:
            return {
                "column": column_name,
                "analysis_performed": False,
                "reason": "insufficient_data",
                "zscore_outliers": [],
                "iqr_outliers": [],
                "total_outliers": 0
            }

        zscore_outliers = self.detect_zscore_outliers(numeric_values)
        iqr_outliers = self.detect_iqr_outliers(numeric_values)

        # Combine results (avoid duplicates)
        all_outlier_indices = set()
        combined_outliers = []

        for idx, val, score in zscore_outliers + iqr_outliers:
            if idx not in all_outlier_indices:
                outlier_type = "zscore" if (idx, val, score) in zscore_outliers else "iqr"
                combined_outliers.append((idx, val, outlier_type))
                all_outlier_indices.add(idx)

        return {
            "column": column_name,
            "analysis_performed": True,
            "sample_size": len(numeric_values),
            "zscore_outliers": len(zscore_outliers),
            "iqr_outliers": len(iqr_outliers),
            "total_outliers": len(combined_outliers),
            "outlier_percentage": len(combined_outliers) / len(numeric_values) * 100,
            "outlier_details": combined_outliers[:10],  # Limit details for large datasets
            "statistics": {
                "mean": statistics.mean(numeric_values),
                "median": statistics.median(numeric_values),
                "std": statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0,
                "min": min(numeric_values),
                "max": max(numeric_values)
            }
        }


class DataValidator:
    """
    Comprehensive data validation for dataset quality assessment.
    """

    def __init__(self):
        self.issues = []

    def validate_dataset(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive data quality validation.

        Args:
            data: Dataset as list of dictionaries

        Returns:
            Validation report dictionary
        """
        self.issues = []

        if not data:
            self.issues.append({"type": "critical", "message": "Empty dataset"})
            return self._generate_report(0, 0)

        # Basic structure validation
        self._validate_structure(data)

        # Data type consistency
        self._validate_data_types(data)

        # Value range validation
        self._validate_value_ranges(data)

        # Missing data analysis
        missing_analysis = self._analyze_missing_data(data)

        # Duplicate detection
        duplicate_analysis = self._detect_duplicates(data)

        return self._generate_report(len(data), len(data[0]) if data else 0,
                                   missing_analysis, duplicate_analysis)

    def _validate_structure(self, data: List[Dict[str, Any]]) -> None:
        """Validate basic data structure."""
        if not data:
            return

        first_row_keys = set(data[0].keys())
        expected_length = len(first_row_keys)

        for i, row in enumerate(data):
            if not isinstance(row, dict):
                self.issues.append({
                    "type": "critical",
                    "message": f"Row {i} is not a dictionary",
                    "row": i
                })
                continue

            row_keys = set(row.keys())
            if len(row_keys) != expected_length:
                self.issues.append({
                    "type": "error",
                    "message": f"Row {i} has inconsistent column count: expected {expected_length}, got {len(row_keys)}",
                    "row": i
                })

            if row_keys != first_row_keys:
                missing_keys = first_row_keys - row_keys
                extra_keys = row_keys - first_row_keys
                if missing_keys:
                    self.issues.append({
                        "type": "error",
                        "message": f"Row {i} missing columns: {missing_keys}",
                        "row": i
                    })
                if extra_keys:
                    self.issues.append({
                        "type": "warning",
                        "message": f"Row {i} has extra columns: {extra_keys}",
                        "row": i
                    })

    def _validate_data_types(self, data: List[Dict[str, Any]]) -> None:
        """Validate data type consistency within columns."""
        if not data:
            return

        # Sample first few rows to infer expected types
        sample_size = min(10, len(data))
        column_types = {}

        for row in data[:sample_size]:
            for key, value in row.items():
                if key not in column_types:
                    column_types[key] = type(value).__name__
                elif column_types[key] != type(value).__name__ and value is not None:
                    # Allow some flexibility for numeric strings
                    if not (column_types[key] in ['str', 'int', 'float'] and
                           type(value).__name__ in ['str', 'int', 'float']):
                        self.issues.append({
                            "type": "warning",
                            "message": f"Column '{key}' has inconsistent types: {column_types[key]} vs {type(value).__name__}",
                            "column": key
                        })

    def _validate_value_ranges(self, data: List[Dict[str, Any]]) -> None:
        """Validate value ranges for potential data quality issues."""
        if not data:
            return

        for row_idx, row in enumerate(data):
            for col_name, value in row.items():
                if isinstance(value, (int, float)) and not math.isnan(value):
                    # Check for unrealistic values
                    if abs(value) > 1e10:  # Extremely large numbers
                        self.issues.append({
                            "type": "warning",
                            "message": f"Extreme value in '{col_name}': {value}",
                            "row": row_idx,
                            "column": col_name,
                            "value": value
                        })

    def _analyze_missing_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        if not data:
            return {"total_missing": 0, "missing_by_column": {}, "missing_percentage": 0}

        total_cells = len(data) * len(data[0])
        total_missing = 0
        missing_by_column = defaultdict(int)

        def is_missing(val):
            if val is None:
                return True
            if isinstance(val, float) and math.isnan(val):
                return True
            if isinstance(val, str):
                stripped = val.strip().lower()
                # Check for empty strings and various null representations
                if stripped == '' or stripped in ['none', 'null', 'nan', 'n/a', 'na']:
                    return True
            return False

        for row in data:
            for col_name, value in row.items():
                if is_missing(value):
                    total_missing += 1
                    missing_by_column[col_name] += 1

        return {
            "total_missing": total_missing,
            "total_cells": total_cells,
            "missing_percentage": total_missing / total_cells * 100 if total_cells > 0 else 0,
            "missing_by_column": dict(missing_by_column)
        }

    def _detect_duplicates(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect duplicate rows."""
        if not data:
            return {"duplicate_rows": 0, "duplicate_percentage": 0}

        # Convert rows to tuples for hashing
        row_tuples = []
        for row in data:
            # Sort keys for consistent comparison
            row_tuple = tuple(sorted(row.items()))
            row_tuples.append(row_tuple)

        unique_rows = set(row_tuples)
        duplicate_count = len(data) - len(unique_rows)

        return {
            "duplicate_rows": duplicate_count,
            "unique_rows": len(unique_rows),
            "duplicate_percentage": duplicate_count / len(data) * 100 if data else 0
        }

    def _generate_report(self, row_count: int, col_count: int,
                        missing_analysis: Dict = None, duplicate_analysis: Dict = None) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        # Categorize issues
        critical_issues = [i for i in self.issues if i["type"] == "critical"]
        error_issues = [i for i in self.issues if i["type"] == "error"]
        warning_issues = [i for i in self.issues if i["type"] == "warning"]

        # Overall quality score (0-100)
        quality_score = 100
        if critical_issues:
            quality_score -= 50  # Critical issues severely impact quality
        if error_issues:
            quality_score -= 20 * min(len(error_issues), 3)  # Up to 60 points for errors
        if warning_issues:
            quality_score -= 5 * min(len(warning_issues), 10)  # Up to 50 points for warnings

        quality_score = max(0, min(100, quality_score))

        report = {
            "dataset_info": {
                "rows": row_count,
                "columns": col_count,
                "total_cells": row_count * col_count
            },
            "quality_score": quality_score,
            "quality_assessment": self._assess_quality(quality_score),
            "issues": {
                "critical": len(critical_issues),
                "errors": len(error_issues),
                "warnings": len(warning_issues),
                "details": self.issues[:20]  # Limit details
            }
        }

        if missing_analysis:
            report["missing_data"] = missing_analysis

        if duplicate_analysis:
            report["duplicates"] = duplicate_analysis

        return report

    def _assess_quality(self, score: float) -> str:
        """Assess overall data quality based on score."""
        if score >= 90:
            return "excellent"
        elif score >= 80:
            return "good"
        elif score >= 70:
            return "fair"
        elif score >= 60:
            return "poor"
        else:
            return "critical"


class EnhancedPreprocessingPipeline:
    """
    Complete preprocessing pipeline integrating imputation, outlier detection, and validation.
    """

    def __init__(self, imputation_epsilon: float = 0.1, outlier_z_threshold: float = 3.0):
        """
        Initialize the preprocessing pipeline.

        Args:
            imputation_epsilon: Privacy budget for imputation operations
            outlier_z_threshold: Z-score threshold for outlier detection
        """
        self.imputation = DPImputation(imputation_epsilon)
        self.outlier_detector = OutlierDetector(outlier_z_threshold)
        self.validator = DataValidator()

    def preprocess_dataset(self, data: List[Dict[str, Any]],
                          column_types: Dict[str, str],
                          epsilon_budget: float = 0.5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run complete preprocessing pipeline on dataset.

        Args:
            data: Original dataset
            column_types: Inferred column types
            epsilon_budget: Total privacy budget for preprocessing

        Returns:
            Tuple of (preprocessed_data, preprocessing_report)
        """
        preprocessing_report = {
            "stages": [],
            "original_row_count": len(data),
            "epsilon_used": 0.0,
            "issues_detected": [],
            "recommendations": []
        }

        # Stage 1: Data validation
        print("Stage 1: Validating data quality...")
        validation_report = self.validator.validate_dataset(data)
        preprocessing_report["stages"].append({
            "stage": "validation",
            "report": validation_report
        })

        if validation_report["issues"]["critical"] > 0:
            preprocessing_report["issues_detected"].append(
                f"Critical validation issues: {validation_report['issues']['critical']}"
            )
            preprocessing_report["recommendations"].append(
                "Address critical data quality issues before anonymization"
            )

        # Stage 2: Missing value imputation
        print("Stage 2: Handling missing values...")
        imputation_results = self._impute_missing_values(data, column_types, epsilon_budget)
        preprocessed_data = imputation_results["data"]
        preprocessing_report["stages"].append({
            "stage": "imputation",
            "report": imputation_results["summary"]
        })
        preprocessing_report["epsilon_used"] += imputation_results["epsilon_used"]

        # Stage 3: Outlier detection and reporting
        print("Stage 3: Analyzing data distributions...")
        outlier_report = self._analyze_outliers(preprocessed_data, column_types)
        preprocessing_report["stages"].append({
            "stage": "outlier_analysis",
            "report": outlier_report
        })

        if outlier_report["total_outliers"] > 0:
            preprocessing_report["issues_detected"].append(
                f"Potential outliers detected: {outlier_report['total_outliers']} across {len(outlier_report['columns_with_outliers'])} columns"
            )
            preprocessing_report["recommendations"].append(
                "Review outlier analysis and consider data cleaning if needed"
            )

        # Final summary
        preprocessing_report["final_row_count"] = len(preprocessed_data)
        preprocessing_report["data_quality_score"] = validation_report.get("quality_score", 0)

        return preprocessed_data, preprocessing_report

    def _impute_missing_values(self, data: List[Dict[str, Any]],
                             column_types: Dict[str, str],
                             total_epsilon: float) -> Dict[str, Any]:
        """Handle missing value imputation across all columns."""
        if not data:
            return {"data": data, "summary": {}, "epsilon_used": 0.0}

        # Convert to column-wise format for processing
        columns_data = {}
        for col_name in data[0].keys():
            columns_data[col_name] = [row.get(col_name) for row in data]

        epsilon_per_column = total_epsilon / len(columns_data) if columns_data else 0
        total_epsilon_used = 0.0
        imputation_summary = {
            "columns_processed": 0,
            "columns_with_missing": 0,
            "total_missing_values": 0,
            "imputation_methods": {},
            "epsilon_used": 0.0
        }

        # Process each column
        for col_name, col_values in columns_data.items():
            col_type = column_types.get(col_name, 'string')

            if col_type in ['numeric', 'monetary', 'count', 'age', 'year']:
                # Numeric imputation
                imputed_values, stats = self.imputation.impute_numeric_column(
                    col_values, epsilon_per_column
                )
                columns_data[col_name] = imputed_values

            elif col_type in ['string', 'boolean']:
                # Categorical imputation
                imputed_values, stats = self.imputation.impute_categorical_column(
                    col_values, epsilon_per_column
                )
                columns_data[col_name] = imputed_values
            else:
                # No imputation for other types
                stats = {"method": "none", "missing_count": 0}

            # Update summary
            imputation_summary["columns_processed"] += 1

            if stats["missing_count"] > 0:
                imputation_summary["columns_with_missing"] += 1
                imputation_summary["total_missing_values"] += stats["missing_count"]

                method = stats["method"]
                if method not in imputation_summary["imputation_methods"]:
                    imputation_summary["imputation_methods"][method] = 0
                imputation_summary["imputation_methods"][method] += 1

            total_epsilon_used += stats.get("epsilon_used", 0)

        # Convert back to row-wise format
        processed_data = []
        column_names = list(columns_data.keys())

        for i in range(len(data)):
            row = {}
            for col_name in column_names:
                row[col_name] = columns_data[col_name][i]
            processed_data.append(row)

        imputation_summary["epsilon_used"] = total_epsilon_used

        return {
            "data": processed_data,
            "summary": imputation_summary,
            "epsilon_used": total_epsilon_used
        }

    def _analyze_outliers(self, data: List[Dict[str, Any]],
                         column_types: Dict[str, str]) -> Dict[str, Any]:
        """Analyze outliers across numeric columns."""
        outlier_summary = {
            "total_outliers": 0,
            "columns_analyzed": 0,
            "columns_with_outliers": [],
            "outlier_details": {}
        }

        for col_name, col_type in column_types.items():
            if col_type in ['numeric', 'monetary', 'count', 'age', 'year']:
                col_values = [row.get(col_name) for row in data]
                numeric_values = [v for v in col_values if isinstance(v, (int, float)) and not math.isnan(v)]

                if len(numeric_values) >= 3:
                    outlier_analysis = self.outlier_detector.detect_outliers_comprehensive(
                        numeric_values, col_name
                    )

                    outlier_summary["columns_analyzed"] += 1
                    outlier_summary["outlier_details"][col_name] = outlier_analysis

                    if outlier_analysis["total_outliers"] > 0:
                        outlier_summary["columns_with_outliers"].append(col_name)
                        outlier_summary["total_outliers"] += outlier_analysis["total_outliers"]

        return outlier_summary