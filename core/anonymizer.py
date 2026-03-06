"""
core/anonymizer.py
──────────────────
Pure-Python DP core logic. No CLI, no file I/O — just the anonymization
pipeline. Consumed by the FastAPI backend (backend/routers/anonymize.py).
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from dp.budget import PrivacyBudget
from dp.mechanisms import DPMechanisms, infer_column_type
from dp.laplace import set_seed
from config.loader import ConfigLoader
from preprocessing.pipeline import EnhancedPreprocessingPipeline
from ai.semantic_analyzer import SemanticAnalyzer
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_data(data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Convert list-of-row-dicts → column-wise dict-of-lists."""
    if not data:
        return {}
    columns: Dict[str, List[Any]] = {}
    for row in data:
        for key, value in row.items():
            columns.setdefault(key, []).append(value)
    return columns


def _convert_data_back(columns: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Convert column-wise dict-of-lists → list-of-row-dicts."""
    if not columns:
        return []
    num_rows = len(next(iter(columns.values())))
    return [
        {col: vals[i] if i < len(vals) else None for col, vals in columns.items()}
        for i in range(num_rows)
    ]


def infer_column_types(
    headers: List[str],
    sample_data: List[Dict[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, dict]]:
    """
    Infer column types using AI if OPENAI_API_KEY is set, then statistical
    heuristics for the remainder.
    """
    column_data = preprocess_data(sample_data)
    column_types: Dict[str, str] = {}
    metadata: Dict[str, dict] = {}

    ai_analyzer = SemanticAnalyzer()
    ai_types: Dict[str, str] = (
        ai_analyzer.analyze_columns(headers, sample_data) if ai_analyzer.client else {}
    )

    for header in headers:
        if header in column_data:
            sample_values = column_data[header][:100]
            col_type, col_meta = infer_column_type(header, sample_values)
            column_types[header] = ai_types.get(header, col_type)
            metadata[header] = col_meta
        else:
            column_types[header] = "string"
            metadata[header] = {}

    return column_types, metadata


def _detect_and_recompute_relationships(
    anonymized_columns: Dict[str, list],
    original_columns: Dict[str, list],
    column_types: Dict[str, str]
):
    """
    Heuristically detect and recompute derived columns to preserve consistency.
    Example: Total = Price * Quantity
    """
    headers = list(anonymized_columns.keys())
    numeric_cols = [h for h in headers if column_types.get(h) in ("numeric", "monetary", "count", "age")]
    
    if len(numeric_cols) < 3:
        return

    # Check for Product Relationships: A * B = C
    for i, a in enumerate(numeric_cols):
        for j, b in enumerate(numeric_cols):
            if i == j: continue
            for k, c in enumerate(numeric_cols):
                if k == i or k == j: continue
                
                # Sample 100 rows to verify relationship
                orig_a = np.array(pd.to_numeric(original_columns[a], errors='coerce'))
                orig_b = np.array(pd.to_numeric(original_columns[b], errors='coerce'))
                orig_c = np.array(pd.to_numeric(original_columns[c], errors='coerce'))
                
                mask = ~np.isnan(orig_a) & ~np.isnan(orig_b) & ~np.isnan(orig_c)
                if mask.sum() < 10: continue
                
                # Check for A * B = C
                diff = np.abs((orig_a[mask] * orig_b[mask]) - orig_c[mask])
                if np.all(diff < 0.01): # Tolerance for floating point
                    # Recompute C in the anonymized data
                    anon_a = np.array(pd.to_numeric(anonymized_columns[a], errors='coerce'))
                    anon_b = np.array(pd.to_numeric(anonymized_columns[b], errors='coerce'))
                    anonymized_columns[c] = (anon_a * anon_b).tolist()
                    continue

                # Check for A + B = C
                diff_sum = np.abs((orig_a[mask] + orig_b[mask]) - orig_c[mask])
                if np.all(diff_sum < 0.01):
                    anon_a = np.array(pd.to_numeric(anonymized_columns[a], errors='coerce'))
                    anon_b = np.array(pd.to_numeric(anonymized_columns[b], errors='coerce'))
                    anonymized_columns[c] = (anon_a + anon_b).tolist()
                    continue


# ─────────────────────────────────────────────────────────────────────────────
# Main anonymization pipeline
# ─────────────────────────────────────────────────────────────────────────────

def apply_anonymization(
    original_data: List[Dict[str, Any]],
    config_loader: ConfigLoader,
    excluded_columns: Optional[List[str]] = None,
    type_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], PrivacyBudget, Dict[str, Any], List[Dict[str, Any]], Dict[str, str], bool]:
    """
    Apply differential privacy anonymization to the dataset.

    Returns
    -------
    (anonymized_data, budget, preprocessing_report, preprocessed_data,
     column_types, ai_was_active)
    """
    if not original_data:
        return [], PrivacyBudget(config_loader.get_global_epsilon()), {}, [], {}, False

    # ── Seed for reproducibility ─────────────────────────────────────────────
    seed = config_loader.get_random_seed()
    set_seed(seed)

    headers = list(original_data[0].keys())
    row_count = len(original_data)

    # Small-dataset auto-adjustment removed to strictly respect user epsilon Choice
    # if row_count < 500 and current_epsilon < 2.0:
    #     config_loader.config["global_epsilon"] = 4.0

    # ── Stage 1: Preprocessing ───────────────────────────────────────────────
    total_epsilon = config_loader.get_global_epsilon()
    preprocessing_epsilon = min(0.1, total_epsilon * 0.1)
    remaining_epsilon = total_epsilon - preprocessing_epsilon

    preprocessor = EnhancedPreprocessingPipeline(
        imputation_epsilon=preprocessing_epsilon * 0.7
    )
    preprocessed_data, preprocessing_report = preprocessor.preprocess_dataset(
        original_data, {}, preprocessing_epsilon
    )

    # ── Stage 2: Column type inference ──────────────────────────────────────
    column_types, metadata = infer_column_types(
        headers, preprocessed_data[:min(100, len(preprocessed_data))]
    )
    ai_active = bool(os.getenv("OPENAI_API_KEY"))

    # Apply user type overrides from the per-column config dashboard
    if type_overrides:
        for col_name, override_type in type_overrides.items():
            if col_name in column_types and override_type:
                column_types[col_name] = override_type

    # ── Stage 3: Budget + mechanisms setup ──────────────────────────────────
    budget = PrivacyBudget(remaining_epsilon)
    mechanisms = DPMechanisms(budget, metadata=metadata)

    # ── Stage 4: Per-column configs ─────────────────────────────────────────
    column_configs: Dict[str, dict] = {}
    for header in headers:
        col_type = column_types[header]
        column_configs[header] = config_loader.get_column_config(header, col_type)

    # Dynamic allocation happens below in Stage 5

    # ── Stage 5: Apply DP mechanism per column ───────────────────────────────
    column_data = preprocess_data(preprocessed_data)
    anonymized_columns: Dict[str, list] = {}

    # Detect original data types for restoration
    orig_types = {}
    if preprocessed_data:
        first_row = preprocessed_data[0]
        for col in headers:
            val = first_row.get(col)
            if isinstance(val, bool): orig_types[col] = bool
            elif isinstance(val, int): orig_types[col] = int
            elif isinstance(val, (float, np.float64, np.float32)): orig_types[col] = float
            else: orig_types[col] = str

    sensitive_cols = [
        h for h in headers
        if (not excluded_columns or h not in excluded_columns) and
        column_types[h] in ["age", "year", "monetary", "numeric", "count", "boolean"]
    ]

    if sensitive_cols:
        # Check for user-defined epsilons in column_configs
        user_defined_sum = 0
        user_defined_cols = []
        auto_cols = []

        for h in sensitive_cols:
            # We consider it user-defined if it's explicitly in the config loader's 'columns' config
            # OR if it was passed via the dashboard (which sets config['epsilon'])
            # But wait, config_loader.get_column_config adds defaults.
            # Let's check if the header was in config_loader.config['columns']
            if config_loader.config.get('columns', {}).get(h, {}).get('epsilon'):
                user_defined_cols.append(h)
                user_defined_sum += column_configs[h]['epsilon']
            else:
                auto_cols.append(h)

        # Calculate budget pool
        available_epsilon = budget.remaining_epsilon
        
        if auto_cols:
            # Distribute remaining budget among auto_cols
            # ONLY sensitive columns get a share of the DP budget
            remaining_for_auto = max(0, available_epsilon - user_defined_sum)
            epsilon_per_auto = remaining_for_auto / len(auto_cols)
            for h in auto_cols:
                column_configs[h]['epsilon'] = epsilon_per_auto
        
        # Final safety check: if total sum > available, scale proportionally
        total_requested = sum(column_configs[h].get('epsilon', 0) for h in sensitive_cols)
        if total_requested > (available_epsilon + 1e-9) and available_epsilon > 0:
            scale_factor = (available_epsilon - 0.0001) / total_requested
            for h in sensitive_cols:
                column_configs[h]['epsilon'] *= scale_factor

    # ── Stage 6: Apply DP mechanism per column ───────────────────────────────
    for header in headers:
        # Excluded / target columns stay unchanged
        if excluded_columns and header in excluded_columns:
            anonymized_columns[header] = column_data[header]
            continue

        col_type = column_types[header]
        config = column_configs[header]
        original_values = column_data[header]

        # Consume privacy budget for mechanisms that require it
        if col_type in ["age", "year", "monetary", "numeric", "count", "boolean"]:
            epsilon = config.get("epsilon", 0.1)
            op_name = {
                "age": "bounded_laplace",
                "year": "bounded_laplace",
                "monetary": "scaled_laplace",
                "numeric": "laplace",
                "count": "discrete_laplace",
                "boolean": "randomized_response",
            }.get(col_type, "laplace")

            if not budget.consume_epsilon(epsilon, op_name, header):
                anonymized_columns[header] = original_values
                continue

        # Numeric types — vectorised
        if col_type in ["age", "year", "monetary", "numeric", "count"]:
            processed: list = []
            for v in original_values:
                if v == "" or v is None:
                    processed.append(np.nan)
                else:
                    try:
                        processed.append(float(v))
                    except (ValueError, TypeError):
                        processed.append(np.nan)

            anon = mechanisms.apply_mechanism(header, processed, col_type, config)
            if isinstance(anon, np.ndarray):
                final: list = []
                for i, v in enumerate(anon):
                    if np.isnan(processed[i]):
                        final.append(None)
                    else:
                        # Restore original type (int vs float)
                        val = float(v)
                        if orig_types.get(header) == int:
                            final.append(int(round(val)))
                        else:
                            final.append(val)
                anonymized_columns[header] = final
            else:
                anonymized_columns[header] = anon

        # Boolean — vectorised
        elif col_type == "boolean":
            bool_col: list = []
            for v in original_values:
                s = str(v).lower()
                if s in ("true", "1", "yes"):
                    bool_col.append(True)
                elif s in ("false", "0", "no"):
                    bool_col.append(False)
                else:
                    bool_col.append(None)

            anon = mechanisms.apply_mechanism(header, bool_col, col_type, config)
            anonymized_columns[header] = [
                original_values[i] if bool_col[i] is None else bool(v)
                for i, v in enumerate(anon)
            ]

        # String / ID / unknown
        else:
            anonymized_columns[header] = mechanisms.apply_mechanism(
                header, original_values, col_type, config
            )

    # ── Stage 7: Relationship Preservation ──────────────────────────────────
    original_columns = preprocess_data(preprocessed_data)
    _detect_and_recompute_relationships(anonymized_columns, original_columns, column_types)

    anonymized_data = _convert_data_back(anonymized_columns)
    return anonymized_data, budget, preprocessing_report, preprocessed_data, column_types, ai_active
