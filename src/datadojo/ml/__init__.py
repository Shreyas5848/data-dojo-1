"""
ML Pipeline Module - Machine Learning Pipeline Builder
Simple, educational AutoML system for DataDojo.
"""

from .automl_engine import AutoMLEngine, MLPipelineStep, create_ml_templates

__all__ = ["AutoMLEngine", "MLPipelineStep", "create_ml_templates"]