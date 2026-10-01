"""
Weather Data Loader Base Classes and Adapters

This module provides:
1. BaseWeatherDataLoader: Abstract base class for all weather data loaders
2. WeatherDataAdapter: Adapter to unify different weather data formats

Author: Generated for Pangu V2 project
Date: 2025-12-08
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict, Any
import numpy as np


class BaseWeatherDataLoader(ABC):
    """
    Abstract base class for weather data loaders

    All weather data loaders (Pangu, Fengwu, ERA5, etc.) should inherit from this class
    and implement the required methods.
    """

    @abstractmethod
    def load_dataset(self, dataset_name: str) -> Dict:
        """
        Load weather dataset

        Args:
            dataset_name: Name of the dataset (e.g., 'wf1', 'wf2', 'wf3', 'lhb')

        Returns:
            Dict: Dictionary containing train/val/test splits
        """
        pass

    @abstractmethod
    def get_weather_data(self, dataset_name: str, split: str):
        """
        Get weather data for specific split

        Args:
            dataset_name: Name of the dataset
            split: 'train', 'val', 'test', or 'full'

        Returns:
            Weather data (format depends on implementation)
        """
        pass

    @abstractmethod
    def get_output_shapes(self) -> Dict[str, Any]:
        """
        Get output shape information

        Returns:
            Dict: Shape information for model initialization
        """
        pass


class WeatherDataAdapter:
    """
    Weather data adapter to unify different weather data formats

    Unified return format: (weather_data1, weather_data2, metadata)
    - Pangu: (upper, surface, {'type': 'pangu', 'shape': (...)})
    - Fengwu: (fengwu, None, {'type': 'fengwu', 'shape': (...)})
    - ERA5: (era5, None, {'type': 'era5', 'shape': (...)})

    This adapter allows seamless switching between different weather data sources
    without modifying downstream code.

    Example:
        >>> from data.loader_utils import PanguWeatherDataLoader
        >>> loader = PanguWeatherDataLoader('data/processed')
        >>> adapter = WeatherDataAdapter(loader, 'pangu')
        >>> data1, data2, metadata = adapter.get_weather_data('wf1', 'train')
        >>> print(metadata['type'])  # Output: 'pangu'
    """

    def __init__(self, loader: BaseWeatherDataLoader, weather_type: str):
        """
        Initialize the adapter

        Args:
            loader: Weather data loader instance
            weather_type: Type of weather data ('pangu', 'fengwu', etc.)
        """
        self.loader = loader
        self.weather_type = weather_type.lower()

        # Validate weather type
        valid_types = ['pangu', 'pangu_random', 'pangu_5x5', 'pangu_11x11', 'pangu_21x21', 'pangu_31x31', 'fengwu', 'fengwu_random', 'era5']
        if self.weather_type not in valid_types:
            raise ValueError(
                f"Unknown weather_type: {weather_type}. "
                f"Valid types: {', '.join(valid_types)}"
            )

    def get_weather_data(self, dataset_name: str, split: str) -> Tuple:
        """
        Get weather data in unified format

        Args:
            dataset_name: Name of the dataset
            split: 'train', 'val', 'test', or 'full'

        Returns:
            Tuple: (weather_data1, weather_data2, metadata)
                - For Pangu: (upper, surface, metadata)
                - For Fengwu: (fengwu, None, metadata)
                - For ERA5: (era5, None, metadata)
        """
        # Check if it's a pangu-type loader (including spatial variants)
        pangu_types = ['pangu', 'pangu_random', 'pangu_5x5', 'pangu_11x11', 'pangu_21x21', 'pangu_31x31']
        if self.weather_type in pangu_types:
            # Pangu (real, random, or spatial variants) returns tuple: (upper, surface)
            upper, surface = self.loader.get_weather_data(dataset_name, split)

            metadata = {
                'type': self.weather_type,
                'shape': {
                    'upper': upper.shape,
                    'surface': surface.shape
                },
                'description': f'{self.weather_type.capitalize()} weather data with upper and surface layers'
            }

            return upper, surface, metadata

        elif self.weather_type == 'fengwu' or self.weather_type == 'fengwu_random':
            # Fengwu (real or random) returns single array
            fengwu = self.loader.get_weather_data(dataset_name, split)

            metadata = {
                'type': self.weather_type,
                'shape': {
                    'fengwu': fengwu.shape
                },
                'description': f'{self.weather_type.capitalize()} weather data with unified features'
            }

            return fengwu, None, metadata

        elif self.weather_type == 'era5':
            # ERA5 ground truth returns single array (same structure as Fengwu)
            era5 = self.loader.get_weather_data(dataset_name, split)

            metadata = {
                'type': self.weather_type,
                'shape': {
                    'era5': era5.shape
                },
                'description': 'ERA5 ground truth weather data with unified features'
            }

            return era5, None, metadata

        else:
            raise ValueError(f"Unsupported weather_type: {self.weather_type}")

    def get_type(self) -> str:
        """
        Get weather data type

        Returns:
            str: Weather type ('pangu', 'fengwu', etc.)
        """
        return self.weather_type

    def load_dataset(self, dataset_name: str) -> Dict:
        """
        Load dataset (delegates to underlying loader)

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dict: Dataset dictionary
        """
        return self.loader.load_dataset(dataset_name)

    def get_output_shapes(self) -> Dict[str, Any]:
        """
        Get output shape information (delegates to underlying loader)

        Returns:
            Dict: Shape information
        """
        return self.loader.get_output_shapes()

    def __repr__(self) -> str:
        """String representation"""
        return f"WeatherDataAdapter(type='{self.weather_type}', loader={self.loader.__class__.__name__})"
