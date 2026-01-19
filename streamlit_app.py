#!/usr/bin/env python3
"""
Privacy Shield - Streamlit Web Application

A web-based interface for the Privacy Shield differential privacy tool.
Allows users to upload CSV files, configure privacy parameters, and view
anonymized results with detailed explanations.
"""

import streamlit as st
import pandas as pd
import tempfile
import os
from typing import Dict, List, Any, Optional, Tuple
import io

# Import our existing modules
from privacyshield import (
    read_csv_file, write_csv_file, apply_anonymization,
    infer_column_types, preprocess_data
)
from config.loader import ConfigLoader
from dp.budget import PrivacyBudget
from metrics.utility import generate_utility_report
from metrics.risk import generate_risk_report


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Privacy Shield",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title and description
    st.title("🛡️ Privacy Shield")
    st.markdown("""
    **Differential Privacy Data Anonymization Tool**

    Protect your sensitive data while preserving statistical utility. Upload a CSV file,
    configure privacy parameters, and get anonymized results with detailed explanations.
    """)

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Global epsilon slider
        global_epsilon = st.slider(
            "Global Privacy Budget (ε)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Lower values provide stronger privacy protection but reduce utility"
        )

        # File upload
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your CSV file to anonymize"
        )

        # Advanced options
        with st.expander("Advanced Options"):
            max_rows = st.number_input(
                "Max rows to process",
                min_value=1,
                max_value=10000,
                value=1000,
                help="Limit processing for large files (for testing)"
            )

        # Process button
        process_button = st.button("🚀 Anonymize Data", type="primary", use_container_width=True)

    # Main content area
    if uploaded_file is not None:
        # Display file info
        st.header("📊 Input Data Preview")

        try:
            # Read the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            headers, data = read_csv_file(tmp_file_path, max_rows)

            # Show data preview
            df_preview = pd.DataFrame(data[:5])  # Show first 5 rows
            st.dataframe(df_preview, use_container_width=True)

            st.caption(f"Showing preview of {len(data)} rows × {len(headers)} columns")

            # Column type inference
            st.subheader("🔍 Column Type Detection")
            column_types = infer_column_types(headers, data[:min(100, len(data))])

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Quantitative Columns**")
                quantitative_cols = [col for col, typ in column_types.items()
                                   if typ in ['age', 'year', 'monetary', 'numeric', 'count']]
                if quantitative_cols:
                    for col in quantitative_cols:
                        col_type = column_types[col]
                        # Add emoji indicators for different types
                        emoji = {
                            'age': '🎂',
                            'year': '📅',
                            'monetary': '💰',
                            'numeric': '📊',
                            'count': '🔢'
                        }.get(col_type, '📈')
                        st.write(f"{emoji} {col} ({col_type})")
                else:
                    st.write("*None detected*")

            with col2:
                st.markdown("**Categorical Columns**")
                categorical_cols = [col for col, typ in column_types.items()
                                  if typ not in ['age', 'year', 'monetary', 'numeric', 'count']]
                if categorical_cols:
                    for col in categorical_cols:
                        col_type = column_types[col]
                        emoji = '🏷️' if col_type == 'string' else '✅' if col_type == 'boolean' else '📝'
                        st.write(f"{emoji} {col} ({col_type})")
                else:
                    st.write("*All columns are quantitative*")

            # Process button action
            if process_button:
                with st.spinner("🔒 Applying differential privacy..."):
                    # Create config with user settings
                    config_loader = ConfigLoader()
                    config_loader.config['global_epsilon'] = global_epsilon

                    # Apply anonymization
                    anonymized_data, budget = apply_anonymization(data, config_loader)

                    # Display results
                    display_results(headers, anonymized_data, budget, data, column_types)

        except Exception as e:
            st.error(f"Error processing file: {e}")
        finally:
            # Clean up temp file
            if 'tmp_file_path' in locals():
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
    else:
        # Show welcome message when no file is uploaded
        st.info("👆 Upload a CSV file to get started!")

        # Example section
        st.header("📖 How It Works")
        st.markdown("""
        1. **Upload** your CSV file containing sensitive data
        2. **Configure** the privacy budget (ε) - lower values = stronger privacy
        3. **Anonymize** your data using differential privacy mechanisms
        4. **Review** the results with detailed explanations

        **Privacy Budget (ε)**: Controls the trade-off between privacy and utility:
        - ε = 0.1: Very strong privacy (high noise, low utility)
        - ε = 1.0: Balanced privacy and utility (recommended)
        - ε = 5.0: Weaker privacy (low noise, high utility)
        """)


def display_results(headers: List[str], anonymized_data: List[Dict[str, Any]],
                   budget: PrivacyBudget, original_data: List[Dict[str, Any]],
                   column_types: Dict[str, str]):
    """Display anonymization results with explanations."""

    st.header("🎯 Anonymization Results")

    # Success message
    st.success(f"✅ Successfully anonymized {len(anonymized_data)} records!")

    # Privacy budget summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Privacy Budget Used", f"ε = {budget.used_epsilon:.3f}")
    with col2:
        st.metric("Budget Remaining", f"ε = {budget.remaining_epsilon:.3f}")
    with col3:
        st.metric("Protection Level",
                 "🛡️ Strong" if budget.used_epsilon < 0.5 else
                 "🔒 Moderate" if budget.used_epsilon < 2.0 else "⚠️ Light")

    # Anonymized data preview
    st.subheader("📋 Anonymized Data Preview")
    df_anonymized = pd.DataFrame(anonymized_data[:10])  # Show first 10 rows
    st.dataframe(df_anonymized, use_container_width=True)

    if len(anonymized_data) > 10:
        st.caption(f"Showing first 10 of {len(anonymized_data)} anonymized rows")

    # Download button
    st.subheader("💾 Download Results")
    csv_buffer = io.StringIO()
    df_full = pd.DataFrame(anonymized_data)
    df_full.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="📥 Download Anonymized CSV",
        data=csv_data,
        file_name="anonymized_data.csv",
        mime="text/csv",
        help="Download the fully anonymized dataset"
    )

    # Detailed explanations
    with st.expander("📚 Understanding Your Results", expanded=True):
        st.markdown("""
        ### 🔒 Privacy Protection Applied

        **Differential Privacy Mechanisms Used:**
        """)

        # Explain each column's protection
        for header, col_type in column_types.items():
            if col_type in ['age', 'year', 'monetary', 'numeric', 'count', 'boolean']:
                mechanism = {
                    'age': 'Bounded Laplace',
                    'year': 'Bounded Laplace',
                    'monetary': 'Scaled Laplace',
                    'numeric': 'Laplace',
                    'count': 'Discrete Laplace',
                    'boolean': 'Randomized Response'
                }.get(col_type, 'Laplace')

                if col_type == 'age':
                    description = "• Bounded noise for personal ages (0-120 range)"
                elif col_type == 'year':
                    description = "• Bounded noise for years (1900-2050 range)"
                elif col_type == 'monetary':
                    description = "• Scaled noise for currency/financial values"
                elif col_type == 'numeric':
                    description = "• Standard noise for continuous measurements"
                elif col_type == 'count':
                    description = "• Discrete noise preserving integer values"
                elif col_type == 'boolean':
                    description = "• Randomly flips some true/false answers"
                else:
                    description = "• Calibrated noise for privacy protection"

                st.markdown(f"""
                **{header}** ({col_type}): *{mechanism} Mechanism*
                {description}
                - Privacy guarantee: ε = {budget.used_epsilon:.3f} per query
                - Trade-off: Some accuracy lost for privacy protection
                """)
            else:
                st.markdown(f"""
                **{header}** ({col_type}): *No noise added*
                - Categorical/string data kept as-is
                - Consider additional protection if sensitive
                """)

    # Metrics and analysis
    display_metrics(original_data, anonymized_data, column_types)


def display_metrics(original_data: List[Dict[str, Any]],
                   anonymized_data: List[Dict[str, Any]],
                   column_types: Dict[str, str]):
    """Display utility and risk metrics with explanations."""

    st.header("📊 Privacy vs. Utility Analysis")

    # Prepare data for analysis
    original_columns = preprocess_data(original_data)
    anonymized_columns = preprocess_data(anonymized_data)

    # Convert to numeric for analysis
    for header in original_columns:
        col_type = column_types[header]
        if col_type in ['age', 'year', 'monetary', 'numeric', 'count']:
            # Convert original data
            original_columns[header] = [
                float(v) if v and str(v).replace('.', '').isdigit() else None
                for v in original_columns[header]
            ]
            original_columns[header] = [v for v in original_columns[header] if v is not None]

            # Convert anonymized data
            anonymized_columns[header] = [
                float(v) if isinstance(v, (int, float)) else
                (float(v) if v and str(v).replace('.', '').isdigit() else None)
                for v in anonymized_columns[header]
            ]
            anonymized_columns[header] = [v for v in anonymized_columns[header] if v is not None]

    # Identify quantitative columns for analysis
    quantitative_columns = []
    for header, col_type in column_types.items():
        if (col_type in ['age', 'year', 'monetary', 'numeric', 'count'] and
            header in original_columns and header in anonymized_columns and
            len(original_columns[header]) >= 2 and len(anonymized_columns[header]) >= 2):
            quantitative_columns.append(header)

    if quantitative_columns:
        # Generate reports
        utility_report = generate_utility_report(original_columns, anonymized_columns, quantitative_columns)
        risk_report = generate_risk_report(original_columns, anonymized_columns)

        # Display metrics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Utility Analysis")
            st.caption(f"Analyzing {len(quantitative_columns)} quantitative columns")
            st.code(utility_report, language="text")

            with st.expander("💡 What do these numbers mean?"):
                st.markdown("""
                **Mean Squared Error (MSE)**: Average squared difference between original and anonymized values
                - Lower = Better utility (data is more accurate)
                - Higher = More privacy protection (but less accurate)

                **Mean Absolute Error (MAE)**: Average absolute difference
                - Similar to MSE but less sensitive to large outliers

                **Relative Error**: Percentage difference from original values
                - < 10%: Good utility preservation
                - 10-25%: Moderate utility loss
                - > 25%: Significant utility loss for privacy
                """)

        with col2:
            st.subheader("🔍 Risk Assessment")
            st.code(risk_report, language="text")

            with st.expander("💡 Understanding Risk Metrics"):
                st.markdown("""
                **Re-identification Risk**: Probability that an anonymized record can be linked back to the original
                - < 1%: Very low risk
                - 1-5%: Low risk
                - 5-10%: Moderate risk
                - > 10%: High risk (consider stronger privacy)

                **Attribute Disclosure Risk**: Risk of inferring sensitive attributes
                - Based on correlation analysis between anonymized and original data

                **Uniqueness**: How unique each record is in the dataset
                - Higher uniqueness = Higher re-identification risk
                """)
    else:
        st.warning("⚠️ No quantitative columns found for detailed analysis. Only categorical data was processed.")


if __name__ == '__main__':
    main()