"""CLI command for generating Jupyter notebooks from datasets."""

import os
from pathlib import Path
from typing import Optional
from datetime import datetime


class CLIResult:
    def __init__(self, success: bool, output: str, exit_code: int, error_message: Optional[str] = None):
        self.success = success
        self.output = output
        self.exit_code = exit_code
        self.error_message = error_message


def generate_notebook(
    dataset_path: str,
    template_type: str = "exploratory_data_analysis",
    output_dir: Optional[str] = None,
    open_notebook: bool = False
) -> CLIResult:
    """Generate a Jupyter notebook from a dataset file.
    
    Args:
        dataset_path: Path to the CSV dataset file
        template_type: Type of notebook template to generate
        output_dir: Directory to save the notebook (default: generated_notebooks/)
        open_notebook: Whether to open the notebook after creation
        
    Returns:
        CLIResult with notebook creation status
    """
    try:
        import pandas as pd
        import nbformat as nbf
        from ..notebook.template_engine import NotebookTemplateEngine
        from ..utils.intelligent_profiler import IntelligentProfiler
        
        # Validate dataset path
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message=f"Dataset file not found: {dataset_path}"
            )
        
        if not dataset_file.suffix.lower() == '.csv':
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message=f"Only CSV files are supported. Got: {dataset_file.suffix}"
            )
        
        # Available templates
        available_templates = [
            'exploratory_data_analysis',
            'data_cleaning',
            'classification_analysis',
            'regression_analysis',
            'time_series_analysis',
            'clustering_analysis',
            'dimensionality_reduction',
            'feature_engineering'
        ]
        
        if template_type not in available_templates:
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message=f"Unknown template: {template_type}\nAvailable: {', '.join(available_templates)}"
            )
        
        # Load dataset
        try:
            df = pd.read_csv(dataset_file)
        except Exception as e:
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message=f"Failed to read CSV file: {str(e)}"
            )
        
        dataset_name = dataset_file.stem
        
        # Profile the dataset
        profiler = IntelligentProfiler()
        profile = profiler.profile_dataset(df, dataset_name)
        
        # Create profile results dict for template engine
        profile_results = {
            'total_rows': df.shape[0],
            'total_columns': df.shape[1],
            'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
            'overall_quality_score': profile.overall_quality_score * 100,
            'total_missing': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Generate notebook
        engine = NotebookTemplateEngine()
        
        # Use absolute path for dataset in notebook
        absolute_dataset_path = str(dataset_file.resolve())
        
        notebook = engine.generate_notebook(
            profile_results=profile_results,
            template_type=template_type,
            dataset_name=dataset_name,
            dataset_path=absolute_dataset_path
        )
        
        # Determine output directory
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = Path("generated_notebooks") / template_type
        
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        notebook_filename = f"{dataset_name}_{template_type}_{timestamp}.ipynb"
        notebook_path = out_path / notebook_filename
        
        # Save notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbf.write(notebook, f)
        
        # Build output
        output_lines = []
        output_lines.append("=" * 70)
        output_lines.append("📓 NOTEBOOK GENERATED SUCCESSFULLY")
        output_lines.append("=" * 70)
        output_lines.append("")
        output_lines.append(f"📂 Dataset: {dataset_name}")
        output_lines.append(f"   Rows: {df.shape[0]:,}")
        output_lines.append(f"   Columns: {df.shape[1]}")
        output_lines.append(f"   Quality Score: {profile.overall_quality_score:.1%}")
        output_lines.append("")
        output_lines.append(f"📝 Template: {template_type.replace('_', ' ').title()}")
        output_lines.append("")
        output_lines.append(f"💾 Saved to: {notebook_path.resolve()}")
        output_lines.append("")
        output_lines.append("─" * 70)
        output_lines.append("🚀 NEXT STEPS:")
        output_lines.append("")
        output_lines.append(f"   Open in VS Code:")
        output_lines.append(f"   code \"{notebook_path.resolve()}\"")
        output_lines.append("")
        output_lines.append(f"   Or run with Jupyter:")
        output_lines.append(f"   jupyter notebook \"{notebook_path.resolve()}\"")
        output_lines.append("")
        output_lines.append("─" * 70)
        output_lines.append("")
        output_lines.append("💡 TIP: The notebook includes:")
        
        template_contents = {
            'exploratory_data_analysis': ['Data loading', 'Overview statistics', 'Distribution plots', 'Correlation analysis', 'Missing value analysis'],
            'data_cleaning': ['Missing value handling', 'Duplicate removal', 'Data type fixes', 'Outlier detection', 'Export cleaned data'],
            'classification_analysis': ['Target analysis', 'Feature engineering', 'Train/test split', 'Multiple model training', 'Evaluation metrics'],
            'regression_analysis': ['Target correlation', 'Feature selection', 'Model training', 'Prediction evaluation', 'Residual analysis'],
            'time_series_analysis': ['Time series decomposition', 'Trend analysis', 'Seasonality detection', 'Forecasting models', 'Visualization'],
            'clustering_analysis': ['Feature scaling', 'K-means clustering', 'Elbow method', 'Cluster visualization', 'Cluster profiling'],
            'dimensionality_reduction': ['PCA analysis', 'Feature importance', 'Variance explained', '2D/3D visualization', 'Component selection'],
            'feature_engineering': ['Feature creation', 'Encoding strategies', 'Scaling techniques', 'Feature selection', 'Pipeline building']
        }
        
        for item in template_contents.get(template_type, []):
            output_lines.append(f"   • {item}")
        
        output_lines.append("")
        output_lines.append("=" * 70)
        
        # Optionally open the notebook
        if open_notebook:
            try:
                import subprocess
                import sys
                if sys.platform == 'win32':
                    os.startfile(str(notebook_path.resolve()))
                elif sys.platform == 'darwin':
                    subprocess.run(['open', str(notebook_path.resolve())])
                else:
                    subprocess.run(['xdg-open', str(notebook_path.resolve())])
                output_lines.append("📖 Opening notebook...")
            except Exception:
                output_lines.append("⚠️  Could not auto-open notebook. Please open manually.")
        
        return CLIResult(
            success=True,
            output="\n".join(output_lines),
            exit_code=0
        )
        
    except ImportError as e:
        return CLIResult(
            success=False,
            output="",
            exit_code=1,
            error_message=f"Missing dependency: {str(e)}\nTry: pip install pandas nbformat"
        )
    except Exception as e:
        return CLIResult(
            success=False,
            output="",
            exit_code=1,
            error_message=f"Failed to generate notebook: {str(e)}"
        )


def list_templates() -> CLIResult:
    """List all available notebook templates."""
    templates = {
        'exploratory_data_analysis': 'Comprehensive EDA with visualizations and statistics',
        'data_cleaning': 'Step-by-step data cleaning workflow',
        'classification_analysis': 'ML classification with multiple models',
        'regression_analysis': 'Regression modeling and evaluation',
        'time_series_analysis': 'Time series decomposition and forecasting',
        'clustering_analysis': 'Unsupervised clustering with K-means',
        'dimensionality_reduction': 'PCA and feature reduction techniques',
        'feature_engineering': 'Feature creation and transformation pipeline'
    }
    
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("📓 AVAILABLE NOTEBOOK TEMPLATES")
    output_lines.append("=" * 70)
    output_lines.append("")
    
    for template_id, description in templates.items():
        output_lines.append(f"  {template_id}")
        output_lines.append(f"      {description}")
        output_lines.append("")
    
    output_lines.append("─" * 70)
    output_lines.append("Usage: datadojo notebook <dataset.csv> --template <template_name>")
    output_lines.append("─" * 70)
    
    return CLIResult(
        success=True,
        output="\n".join(output_lines),
        exit_code=0
    )
