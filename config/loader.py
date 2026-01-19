"""
Configuration loader for Privacy Shield.

This module handles loading and validation of YAML configuration files
that specify privacy parameters for different columns.
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path


class ConfigLoader:
    """
    Loads and validates Privacy Shield configuration from YAML files.
    """

    DEFAULT_CONFIG = {
        'global_epsilon': 1.0,
        'columns': {}
    }

    REQUIRED_FIELDS = ['global_epsilon']
    VALID_METHODS = ['laplace', 'bounded_laplace', 'discrete_laplace', 'randomized_response']

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()

        if config_path and os.path.exists(config_path):
            self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If configuration is invalid
        """
        if not self.config_path:
            return self.config

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)

            if loaded_config is None:
                loaded_config = {}

            # Validate and merge with defaults
            self.config = self._validate_and_merge_config(loaded_config)
            return self.config

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")

    def _validate_and_merge_config(self, loaded_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and merge loaded config with defaults.

        Args:
            loaded_config: Configuration loaded from YAML

        Returns:
            Validated and merged configuration

        Raises:
            ValueError: If configuration is invalid
        """
        config = self.DEFAULT_CONFIG.copy()

        # Validate required fields
        for field in self.REQUIRED_FIELDS:
            if field in loaded_config:
                config[field] = loaded_config[field]
            elif field not in config:
                raise ValueError(f"Required field '{field}' missing from configuration")

        # Validate global epsilon
        global_epsilon = config['global_epsilon']
        if not isinstance(global_epsilon, (int, float)) or global_epsilon <= 0:
            raise ValueError(f"global_epsilon must be a positive number, got: {global_epsilon}")

        # Validate column configurations
        if 'columns' in loaded_config:
            config['columns'] = self._validate_column_configs(loaded_config['columns'])

        return config

    def _validate_column_configs(self, column_configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate column-specific configurations.

        Args:
            column_configs: Column configuration dictionary

        Returns:
            Validated column configurations

        Raises:
            ValueError: If column configuration is invalid
        """
        validated_configs = {}

        for column_name, column_config in column_configs.items():
            if not isinstance(column_config, dict):
                raise ValueError(f"Column '{column_name}' configuration must be a dictionary")

            validated_config = {}

            # Validate method
            method = column_config.get('method', 'laplace')
            if method not in self.VALID_METHODS:
                raise ValueError(f"Invalid method '{method}' for column '{column_name}'. "
                               f"Valid methods: {self.VALID_METHODS}")
            validated_config['method'] = method

            # Validate epsilon
            epsilon = column_config.get('epsilon')
            if epsilon is not None:
                if not isinstance(epsilon, (int, float)) or epsilon <= 0:
                    raise ValueError(f"epsilon for column '{column_name}' must be positive, got: {epsilon}")
                validated_config['epsilon'] = float(epsilon)

            # Validate bounds for bounded methods
            if method == 'bounded_laplace':
                min_val = column_config.get('min')
                max_val = column_config.get('max')

                if min_val is not None:
                    if not isinstance(min_val, (int, float)):
                        raise ValueError(f"min for column '{column_name}' must be numeric, got: {min_val}")
                    validated_config['min'] = float(min_val)

                if max_val is not None:
                    if not isinstance(max_val, (int, float)):
                        raise ValueError(f"max for column '{column_name}' must be numeric, got: {max_val}")
                    validated_config['max'] = float(max_val)

                    if 'min' in validated_config and validated_config['max'] <= validated_config['min']:
                        raise ValueError(f"max must be greater than min for column '{column_name}'")

            # Validate sensitivity for laplace methods
            sensitivity = column_config.get('sensitivity')
            if sensitivity is not None:
                if not isinstance(sensitivity, (int, float)) or sensitivity <= 0:
                    raise ValueError(f"sensitivity for column '{column_name}' must be positive, got: {sensitivity}")
                validated_config['sensitivity'] = float(sensitivity)

            # Validate mask_type for string columns (though we don't use method for strings yet)
            mask_type = column_config.get('mask_type')
            if mask_type is not None:
                valid_mask_types = ['partial', 'hash']
                if mask_type not in valid_mask_types:
                    raise ValueError(f"mask_type for column '{column_name}' must be one of {valid_mask_types}, got: {mask_type}")
                validated_config['mask_type'] = mask_type

            validated_configs[column_name] = validated_config

        return validated_configs

    def get_column_config(self, column_name: str, column_type: str = 'numeric') -> Dict[str, Any]:
        """
        Get configuration for a specific column.

        If no specific config exists, returns default config based on column type.

        Args:
            column_name: Name of the column
            column_type: Inferred column type ('age', 'monetary', 'count', 'boolean', 'string')

        Returns:
            Column configuration dictionary
        """
        # Check for column-specific config
        if column_name in self.config['columns']:
            return self.config['columns'][column_name].copy()

        # Return default config based on column type
        defaults = {
            'age': {'method': 'bounded_laplace', 'epsilon': 0.2, 'min': 18, 'max': 90},
            'monetary': {'method': 'laplace', 'epsilon': 0.3, 'sensitivity': 1000.0},
            'count': {'method': 'discrete_laplace', 'epsilon': 0.2},
            'boolean': {'method': 'randomized_response', 'epsilon': 0.5},
            'string': {'mask_type': 'partial'}
        }

        return defaults.get(column_type, {'method': 'laplace', 'epsilon': 0.2}).copy()

    def get_global_epsilon(self) -> float:
        """Get the global epsilon budget."""
        return self.config['global_epsilon']

    def auto_assign_epsilon(self, columns: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Auto-assign epsilon values equally across columns when no config is provided.

        Args:
            columns: List of column names

        Returns:
            Dictionary mapping column names to configurations
        """
        if not columns:
            return {}

        # Calculate equal epsilon split
        num_columns = len(columns)
        epsilon_per_column = self.get_global_epsilon() / num_columns

        assignments = {}
        for column in columns:
            assignments[column] = {
                'method': 'laplace',
                'epsilon': epsilon_per_column
            }

        return assignments

    def save_config(self, output_path: str):
        """
        Save current configuration to a YAML file.

        Args:
            output_path: Path to save configuration
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"ConfigLoader(config_path='{self.config_path}', global_epsilon={self.get_global_epsilon()})"