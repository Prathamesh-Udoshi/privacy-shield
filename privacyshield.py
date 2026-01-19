#!/usr/bin/env python3
"""
Privacy Shield - Differential Privacy Data Anonymization Tool

A lightweight tool for adding differential privacy noise to CSV files
to protect user data while preserving aggregate statistics.
"""

import argparse
import csv
import sys
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import our modules
from dp.budget import PrivacyBudget
from dp.mechanisms import DPMechanisms, infer_column_type
from config.loader import ConfigLoader
from metrics.utility import generate_utility_report
from metrics.risk import generate_risk_report
from preprocessing.pipeline import EnhancedPreprocessingPipeline


def read_csv_file(file_path: str, max_rows: Optional[int] = None) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    Read CSV file and return headers and data.

    Args:
        file_path: Path to CSV file
        max_rows: Maximum number of rows to read (for sampling)

    Returns:
        Tuple of (headers, data_rows)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to detect delimiter
            sample = f.read(1024)
            f.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            f.seek(0)

            reader = csv.DictReader(f, delimiter=delimiter)

            headers = reader.fieldnames
            if not headers:
                raise ValueError("CSV file has no headers")

            data = []
            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break
                data.append(row)

            return headers, data

    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")


def write_csv_file(file_path: str, headers: List[str], data: List[Dict[str, Any]]):
    """
    Write data to CSV file.

    Args:
        file_path: Path to output CSV file
        headers: Column headers
        data: Data rows
    """
    try:
        output_dir = os.path.dirname(file_path)
        if output_dir:  # Only create directory if there is one
            os.makedirs(output_dir, exist_ok=True)

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)

    except Exception as e:
        raise Exception(f"Error writing CSV file: {e}")


def preprocess_data(data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Convert data from list of dicts to dict of lists for easier processing.

    Args:
        data: List of row dictionaries

    Returns:
        Dict mapping column names to lists of values
    """
    if not data:
        return {}

    columns = {}
    for row in data:
        for key, value in row.items():
            if key not in columns:
                columns[key] = []
            columns[key].append(value)

    return columns


def convert_data_back(columns: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Convert data from dict of lists back to list of dicts.

    Args:
        columns: Dict mapping column names to lists of values

    Returns:
        List of row dictionaries
    """
    if not columns:
        return []

    # Get length from first column
    first_col = next(iter(columns.values()))
    num_rows = len(first_col)

    data = []
    for i in range(num_rows):
        row = {}
        for col_name, col_values in columns.items():
            row[col_name] = col_values[i] if i < len(col_values) else None
        data.append(row)

    return data


def infer_column_types(headers: List[str], sample_data: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Infer column types for all columns.

    Args:
        headers: Column headers
        sample_data: Sample data rows

    Returns:
        Dict mapping column names to inferred types
    """
    column_data = preprocess_data(sample_data)
    column_types = {}

    for header in headers:
        if header in column_data:
            sample_values = column_data[header][:100]  # Use first 100 values for inference
            column_types[header] = infer_column_type(header, sample_values)
        else:
            column_types[header] = 'string'  # Default fallback

    return column_types


def apply_anonymization(original_data: List[Dict[str, Any]],
                       config_loader: ConfigLoader) -> tuple[List[Dict[str, Any]], PrivacyBudget, Dict[str, Any], List[Dict[str, Any]], Dict[str, str]]:
    """
    Apply differential privacy anonymization to the data with preprocessing.

    Args:
        original_data: Original data rows
        config_loader: Configuration loader

    Returns:
        Tuple of (anonymized_data, privacy_budget, preprocessing_report)
    """
    if not original_data:
        return [], PrivacyBudget(config_loader.get_global_epsilon()), {}

    headers = list(original_data[0].keys())

    # Stage 1: Enhanced Data Preprocessing
    print("Running enhanced preprocessing pipeline...")

    # Allocate small portion of privacy budget for preprocessing
    total_epsilon = config_loader.get_global_epsilon()
    preprocessing_epsilon = min(0.1, total_epsilon * 0.1)  # Use up to 10% for preprocessing
    remaining_epsilon = total_epsilon - preprocessing_epsilon

    # Create preprocessing pipeline
    preprocessor = EnhancedPreprocessingPipeline(imputation_epsilon=preprocessing_epsilon * 0.7)

    # Run preprocessing
    preprocessed_data, preprocessing_report = preprocessor.preprocess_dataset(
        original_data, {}, preprocessing_epsilon  # Empty column_types initially
    )

    # Show detailed preprocessing results
    print(f"Preprocessing complete:")
    print(f"   - Rows processed: {preprocessing_report.get('original_row_count', 0)}")
    print(f"   - Data quality score: {preprocessing_report.get('data_quality_score', 0)}/100")
    print(f"   - Issues detected: {len(preprocessing_report.get('issues_detected', []))}")

    # Show missing data details
    stages = preprocessing_report.get('stages', [])
    for stage in stages:
        if stage.get('stage') == 'imputation':
            imputation_report = stage.get('report', {})
            if imputation_report.get('total_missing_values', 0) > 0:
                print(f"   - Missing values imputed: {imputation_report['total_missing_values']}")
                missing_by_col = imputation_report.get('missing_by_column', {})
                if missing_by_col:
                    print("   - Missing by column:")
                    for col, count in missing_by_col.items():
                        print(f"     - {col}: {count} missing")

        elif stage.get('stage') == 'validation':
            validation_report = stage.get('report', {})
            missing_data = validation_report.get('missing_data', {})
            if missing_data.get('total_missing', 0) > 0:
                print(f"   - Data completeness: {missing_data['missing_percentage']:.1f}% missing")
                if missing_data.get('missing_by_column'):
                    print("   - Missing breakdown:")
                    for col, count in missing_data['missing_by_column'].items():
                        percentage = (count / preprocessing_report.get('original_row_count', 1)) * 100
                        print(f"     - {col}: {count} rows ({percentage:.1f}%)")

        elif stage.get('stage') == 'outlier_analysis':
            outlier_report = stage.get('report', {})
            total_outliers = outlier_report.get('total_outliers', 0)
            if total_outliers > 0:
                print(f"   - Potential outliers detected: {total_outliers}")
                cols_with_outliers = outlier_report.get('columns_with_outliers', [])
                if cols_with_outliers:
                    print(f"   - Columns with outliers: {', '.join(cols_with_outliers[:3])}{'...' if len(cols_with_outliers) > 3 else ''}")

    if preprocessing_report.get('recommendations'):
        print(f"   - Recommendations: {len(preprocessing_report['recommendations'])}")
        for rec in preprocessing_report['recommendations'][:2]:  # Show first 2 recommendations
            print(f"     - {rec}")

    # Infer column types on preprocessed data
    print("\nInferring column types...")
    column_types = infer_column_types(headers, preprocessed_data[:min(100, len(preprocessed_data))])

    for header, col_type in column_types.items():
        print(f"  {header}: {col_type}")

    # Initialize privacy budget and mechanisms with remaining epsilon
    budget = PrivacyBudget(remaining_epsilon)
    mechanisms = DPMechanisms(budget)

    # Get configurations for each column
    column_configs = {}
    numeric_columns = []

    for header in headers:
        col_type = column_types[header]
        config = config_loader.get_column_config(header, col_type)
        column_configs[header] = config

        # Track numeric columns for utility analysis
        if col_type in ['age', 'year', 'monetary', 'numeric', 'count']:
            numeric_columns.append(header)

    # If no specific configs, auto-assign epsilon
    if not config_loader.config.get('columns'):
        print("No column-specific configuration found. Auto-assigning epsilon equally...")
        auto_configs = config_loader.auto_assign_epsilon(headers)
        column_configs.update(auto_configs)

    # Apply anonymization column by column
    column_data = preprocess_data(preprocessed_data)  # Use preprocessed data
    anonymized_columns = {}

    print("\nApplying differential privacy...")
    for header in headers:
        col_type = column_types[header]
        config = column_configs[header]
        original_values = column_data[header]

        print(f"  Processing {header} ({col_type})...")

        # Consume epsilon for this column if it uses DP
        epsilon_consumed = False
        if col_type in ['age', 'year', 'monetary', 'numeric', 'count', 'boolean']:
            epsilon = config.get('epsilon', 0.1)  # Default fallback
            operation_name = {
                'age': 'bounded_laplace',
                'year': 'bounded_laplace',
                'monetary': 'scaled_laplace',
                'numeric': 'laplace',
                'count': 'discrete_laplace',
                'boolean': 'randomized_response'
            }.get(col_type, 'laplace')

            if not budget.consume_epsilon(epsilon, operation_name, header):
                print(f"    Skipping {header} due to budget constraints")
                anonymized_columns[header] = original_values  # Keep original values
                continue
            epsilon_consumed = True

        # Apply appropriate mechanism
        anonymized_values = []
        for value in original_values:
            try:
                # Convert string values to appropriate types
                processed_value = value
                if value == '' or value is None:
                    processed_value = None
                elif col_type in ['age', 'year', 'monetary', 'numeric', 'count']:
                    try:
                        processed_value = float(value) if '.' in str(value) else int(value)
                    except (ValueError, TypeError):
                        processed_value = None
                elif col_type == 'boolean':
                    str_val = str(value).lower()
                    if str_val in ['true', '1', 'yes']:
                        processed_value = True
                    elif str_val in ['false', '0', 'no']:
                        processed_value = False
                    else:
                        processed_value = None

                # Apply DP mechanism
                if processed_value is not None:
                    anonymized_value = mechanisms.apply_mechanism(
                        header, processed_value, col_type, config
                    )
                else:
                    anonymized_value = value  # Keep missing values as-is

                anonymized_values.append(anonymized_value)

            except Exception as e:
                print(f"    Warning: Failed to process value '{value}': {e}")
                anonymized_values.append(value)  # Keep original on error

        anonymized_columns[header] = anonymized_values

    # Convert back to list of dicts
    anonymized_data = convert_data_back(anonymized_columns)

    return anonymized_data, budget, preprocessing_report, preprocessed_data, column_types


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Privacy Shield - Differential Privacy Data Anonymization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python privacyshield.py --input users.csv --output safe_users.csv
  python privacyshield.py --input data.csv --output anon.csv --config policy.yaml
  python privacyshield.py --input data.csv --output anon.csv --epsilon 0.5
        """
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file path'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output CSV file path'
    )

    parser.add_argument(
        '--config', '-c',
        help='Configuration YAML file path'
    )

    parser.add_argument(
        '--epsilon', '-e',
        type=float,
        help='Global epsilon value (overrides config file)'
    )

    parser.add_argument(
        '--max-rows',
        type=int,
        help='Maximum rows to process (for testing)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    try:
        # Validate inputs
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' does not exist")
            sys.exit(1)

        # Load configuration
        config_loader = ConfigLoader(args.config)

        # Override epsilon if specified
        if args.epsilon is not None:
            config_loader.config['global_epsilon'] = args.epsilon

        if not args.quiet:
            print(f"Privacy Shield v1.0")
            print(f"Input: {args.input}")
            print(f"Output: {args.output}")
            print(f"Global privacy budget: {config_loader.get_global_epsilon()}")
            if args.config:
                print(f"Config: {args.config}")
            print()

        # Read input data
        if not args.quiet:
            print("Reading input data...")

        headers, original_data = read_csv_file(args.input, args.max_rows)

        if not args.quiet:
            print(f"Loaded {len(original_data)} rows with {len(headers)} columns")
            print()

        if not original_data:
            print("Error: No data found in input file")
            sys.exit(1)

        # Apply anonymization with preprocessing
        anonymized_data, budget, preprocessing_report, preprocessed_data, column_types = apply_anonymization(original_data, config_loader)

        # Write output
        if not args.quiet:
            print(f"\nWriting anonymized data to {args.output}...")

        write_csv_file(args.output, headers, anonymized_data)

        # Generate reports
        original_columns = preprocess_data(preprocessed_data)  # Use preprocessed data for fair comparison
        anonymized_columns = preprocess_data(anonymized_data)

        # Column types are already inferred from preprocessed data above

        # Convert columns to appropriate types for utility analysis
        for header in headers:
            col_type = column_types[header]

            # Convert original data
            if col_type in ['age', 'year', 'monetary', 'numeric', 'count']:
                original_columns[header] = [
                    float(v) if v and str(v).replace('.', '').isdigit() else None
                    for v in original_columns[header]
                ]
                # Filter out None values
                original_columns[header] = [v for v in original_columns[header] if v is not None]

            # Convert anonymized data - these should already be numeric but ensure they are
            if col_type in ['age', 'year', 'monetary', 'numeric', 'count']:
                anonymized_columns[header] = [
                    float(v) if isinstance(v, (int, float)) else
                    (float(v) if v and str(v).replace('.', '').isdigit() else None)
                    for v in anonymized_columns[header]
                ]
                # Filter out None values
                anonymized_columns[header] = [v for v in anonymized_columns[header] if v is not None]

        # Identify numeric columns for utility analysis
        numeric_columns = []
        for header, col_type in column_types.items():
            if col_type in ['age', 'year', 'monetary', 'numeric', 'count'] and header in original_columns and header in anonymized_columns:
                # Only include if we have data
                if len(original_columns[header]) >= 2 and len(anonymized_columns[header]) >= 2:
                    numeric_columns.append(header)

        # Generate reports
        privacy_report = budget.get_budget_report()
        utility_report = generate_utility_report(original_columns, anonymized_columns, numeric_columns)
        risk_report = generate_risk_report(original_columns, anonymized_columns)

        # Print reports
        print("\n" + "="*60)
        print(privacy_report)
        print("\n" + "="*60)
        print(utility_report)
        print("\n" + "="*60)
        print(risk_report)
        print("\n" + "="*60)

        if not args.quiet:
            print("Anonymization complete!")
            print(f"Processed {len(anonymized_data)} records")
            print(f"Privacy budget used: {budget.used_epsilon:.3f}")
            print(f"Privacy budget remaining: {budget.remaining_epsilon:.3f}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()