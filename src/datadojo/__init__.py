"""
DataDojo: AI-Powered Data Preparation Learning Framework

An educational framework that teaches data preprocessing skills through hands-on learning
with real datasets, combining production-ready pipeline tools with interactive guidance.
"""

__version__ = "0.1.0"
__author__ = "DataDojo Team"

# Factory function for easy instantiation
def create_dojo(educational_mode: bool = True, config_path: str = None):
    """Create a DataDojo learning environment

    Args:
        educational_mode: Enable step-by-step guidance and explanations
        config_path: Path to custom configuration (optional)

    Returns:
        Dojo instance ready for learning
    """
    from .core.dojo import Dojo
    return Dojo(educational_mode=educational_mode)

__all__ = ["create_dojo", "__version__"]