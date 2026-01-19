#!/usr/bin/env python3
"""
ML Training Demo with Anonymized Data

This script demonstrates how to use Privacy Shield's anonymized data
for training machine learning models safely.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import our privacy shield modules
from privacyshield import read_csv_file, apply_anonymization
from config.loader import ConfigLoader


def prepare_data_for_ml(df, target_column='is_active'):
    """
    Prepare data for machine learning by handling categorical variables
    and splitting features from target.
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Handle categorical columns
    categorical_cols = []
    for col in df.columns:
        if col != target_column and df[col].dtype == 'object':
            categorical_cols.append(col)

    # Label encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Split features and target
    if target_column in df.columns:
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Convert target to numeric if needed
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))

        return X, y, label_encoders
    else:
        # No target column, return all as features
        return df, None, label_encoders


def train_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate ML models on the data.
    """
    results = {}

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model 1: Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)

    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'model': rf_model
    }

    # Model 2: Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)

    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'model': lr_model
    }

    return results


def demonstrate_ml_with_anonymized_data():
    """
    Complete demonstration of using anonymized data for ML training.
    """
    print("Privacy Shield - ML Training with Anonymized Data Demo")
    print("=" * 60)

    # Load original data
    print("Loading original data...")
    headers, original_data = read_csv_file('examples/users.csv')

    # Create config for anonymization - higher epsilon for better utility
    config_loader = ConfigLoader()
    config_loader.config['global_epsilon'] = 5.0  # Higher epsilon = less noise, better utility

    # For ML training demo, we might keep target variable unchanged
    # but anonymize features only. This is a common practice.
    print("Applying differential privacy (keeping target variable unchanged for demo)...")

    # Separate target from features for this demo
    target_column = 'is_active'
    feature_data = []
    target_data = []

    for row in original_data:
        feature_row = {k: v for k, v in row.items() if k != target_column}
        feature_data.append(feature_row)
        target_data.append({target_column: row.get(target_column, None)})

    # Anonymize only the features
    anonymized_features, budget = apply_anonymization(feature_data, config_loader)

    # Recombine with original target
    anonymized_data = []
    for feat_row, target_row in zip(anonymized_features, target_data):
        combined_row = {**feat_row, **target_row}
        anonymized_data.append(combined_row)

    # Convert to DataFrames for easier handling
    original_df = pd.DataFrame(original_data)
    anonymized_df = pd.DataFrame(anonymized_data)

    print(f"Anonymized {len(anonymized_data)} records")
    print(".3f")
    print(".3f")

    # Prepare data for ML
    print("\nPreparing data for machine learning...")

    # Original data preparation
    X_orig, y_orig, _ = prepare_data_for_ml(original_df, 'is_active')
    X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=42
    )

    # Anonymized data preparation
    X_anon, y_anon, _ = prepare_data_for_ml(anonymized_df, 'is_active')
    X_anon_train, X_anon_test, y_anon_train, y_anon_test = train_test_split(
        X_anon, y_anon, test_size=0.2, random_state=42
    )

    # Train models on both datasets
    print("Training models on original data...")
    orig_results = train_models(X_orig_train, X_orig_test, y_orig_train, y_orig_test)

    print("Training models on anonymized data...")
    anon_results = train_models(X_anon_train, X_anon_test, y_anon_train, y_anon_test)

    # Compare results
    print("\nModel Performance Comparison")
    print("=" * 50)

    for model_name in orig_results.keys():
        orig_acc = orig_results[model_name]['accuracy']
        anon_acc = anon_results[model_name]['accuracy']
        diff = abs(orig_acc - anon_acc)

        print(f"\n{model_name}:")
        print(".3f")
        print(".3f")
        print(".3f")

        if diff < 0.05:  # Less than 5% difference
            print("  Excellent: Minimal utility loss")
        elif diff < 0.10:  # Less than 10% difference
            print("  Good: Acceptable utility loss")
        else:
            print("  Poor: Significant utility loss")

    print("\nDemo Complete!")
    print("\nKey Takeaways:")
    print("• Anonymized data CAN be used for ML training with privacy protection")
    print("• Higher epsilon values preserve more utility (but less privacy)")
    print("• Models trained on anonymized data are privacy-safe for deployment")
    print("• Differential privacy enables responsible AI development")
    print("• Real-world applications often use epsilon values tailored to specific use cases")
    print("• The privacy-utility trade-off can be optimized based on requirements")


if __name__ == '__main__':
    demonstrate_ml_with_anonymized_data()