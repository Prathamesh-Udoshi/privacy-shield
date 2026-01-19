"""
Privacy budget management for differential privacy.

This module tracks epsilon consumption across multiple privacy mechanisms
to ensure the total privacy budget is not exceeded.
"""

from typing import Dict, List, Optional
import warnings


class PrivacyBudget:
    """
    Manages global privacy budget (epsilon) across multiple DP operations.

    Tracks epsilon consumption and prevents budget overruns.
    """

    def __init__(self, total_epsilon: float):
        """
        Initialize privacy budget.

        Args:
            total_epsilon: Total privacy budget available
        """
        if total_epsilon <= 0:
            raise ValueError("Total epsilon must be positive")

        self.total_epsilon = total_epsilon
        self.used_epsilon = 0.0
        self.consumptions: List[Dict] = []  # Track individual consumptions

    @property
    def remaining_epsilon(self) -> float:
        """Get remaining unused epsilon."""
        return self.total_epsilon - self.used_epsilon

    def consume_epsilon(self, epsilon: float, operation: str, column: str) -> bool:
        """
        Attempt to consume epsilon for a privacy operation.

        Args:
            epsilon: Amount of epsilon to consume
            operation: Description of the operation
            column: Column name being processed

        Returns:
            True if consumption was successful, False if budget exceeded
        """
        if epsilon <= 0:
            raise ValueError("Epsilon to consume must be positive")

        if self.used_epsilon + epsilon > self.total_epsilon:
            warnings.warn(
            f"Privacy budget exceeded! Operation '{operation}' on column '{column}' "
            f"requires e={epsilon:.3f}, but only e={self.remaining_epsilon:.3f} remains. "
            f"Operation will be skipped.",
                UserWarning
            )
            return False

        # Record the consumption
        self.consumptions.append({
            'operation': operation,
            'column': column,
            'epsilon': epsilon
        })

        self.used_epsilon += epsilon
        return True

    def can_consume(self, epsilon: float) -> bool:
        """
        Check if epsilon can be consumed without exceeding budget.

        Args:
            epsilon: Amount of epsilon to check

        Returns:
            True if consumption is possible
        """
        return self.used_epsilon + epsilon <= self.total_epsilon

    def get_budget_report(self) -> str:
        """
        Generate a human-readable budget report.

        Returns:
            Formatted string with budget usage details
        """
        report_lines = [
            "Privacy Budget Report",
            "=" * 50,
            f"Total Budget: e = {self.total_epsilon:.3f}",
            f"Used Budget:  e = {self.used_epsilon:.3f}",
            f"Remaining:    e = {self.remaining_epsilon:.3f}",
            f"Utilization:  {self.used_epsilon/self.total_epsilon*100:.1f}%",
            "",
            "Consumption Details:"
        ]

        if not self.consumptions:
            report_lines.append("  No operations performed yet.")
        else:
            for i, consumption in enumerate(self.consumptions, 1):
                report_lines.append(
                    f"  {i}. {consumption['operation']} on '{consumption['column']}': "
                    f"e = {consumption['epsilon']:.3f}"
                )

        return "\n".join(report_lines)

    def reset(self):
        """Reset the budget (for testing purposes)."""
        self.used_epsilon = 0.0
        self.consumptions.clear()