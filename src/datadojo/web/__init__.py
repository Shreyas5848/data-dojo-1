"""
DataDojo Web Module
Web interface components for the DataDojo platform.
"""

__version__ = "1.0.0"

from .visualizations import DataVisualizationEngine, create_data_quality_summary_card
from .config import (
    DATADOJO_THEME, LAYOUT_CONFIG, DATA_CONFIG, VIZ_CONFIG, 
    QUALITY_THRESHOLDS, apply_theme, get_quality_class, format_metric_delta
)

__all__ = [
    'DataVisualizationEngine',
    'create_data_quality_summary_card',
    'DATADOJO_THEME',
    'LAYOUT_CONFIG',
    'DATA_CONFIG',
    'VIZ_CONFIG',
    'QUALITY_THRESHOLDS',
    'apply_theme',
    'get_quality_class',
    'format_metric_delta'
]