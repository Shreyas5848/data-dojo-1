"""
List Datasets Command for DataDojo CLI
Discovers and displays available datasets in the workspace.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

class CLIResult:
    def __init__(self, success: bool, output: str, exit_code: int, error_message: Optional[str] = None):
        self.success = success
        self.output = output
        self.exit_code = exit_code
        self.error_message = error_message

class DatasetInfo:
    def __init__(self, name: str, path: str, domain: str, size_mb: float, rows: int, columns: int):
        self.name = name
        self.path = path
        self.domain = domain
        self.size_mb = size_mb
        self.rows = rows
        self.columns = columns

def discover_datasets(search_paths: List[str] = None) -> List[DatasetInfo]:
    """Discover all CSV datasets in specified paths (optimized for speed)."""
    
    if search_paths is None:
        search_paths = ["datasets", "test_datasets", "generated_datasets", "demo_datasets"]
    
    datasets = []
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        for root, dirs, files in os.walk(search_path):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        # Get file info - fast operation
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        
                        # Determine domain from path
                        path_parts = Path(file_path).parts
                        domain = None
                        for part in path_parts:
                            if part.lower() in ["healthcare", "finance", "ecommerce", "financial"]:
                                domain = part.lower()
                                if domain == "financial":
                                    domain = "finance"
                                break
                        
                        # If no domain from path, try to infer from filename
                        if domain is None:
                            filename_lower = file.lower()
                            if any(term in filename_lower for term in ["patient", "lab", "health", "medical", "ehr", "diagnosis"]):
                                domain = "healthcare"
                            elif any(term in filename_lower for term in ["bank", "transaction", "credit", "loan", "stock", "finance"]):
                                domain = "finance"
                            elif any(term in filename_lower for term in ["customer", "order", "product", "sales", "ecommerce", "shop"]):
                                domain = "ecommerce"
                            else:
                                domain = "other"
                        
                        # FAST: Estimate rows from file size instead of counting lines
                        # Average CSV row is ~100-200 bytes
                        estimated_rows = max(1, int(size_mb * 1024 * 1024 / 150))  # ~150 bytes per row estimate
                        
                        # FAST: Get column count by reading just the header
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                header = f.readline()
                                columns = len(header.split(','))
                        except Exception:
                            columns = 0
                        
                        dataset_info = DatasetInfo(
                            name=file,
                            path=file_path,
                            domain=domain,
                            size_mb=size_mb,
                            rows=estimated_rows,
                            columns=columns
                        )
                        
                        datasets.append(dataset_info)
                        
                    except Exception as e:
                        # Skip files that can't be processed
                        continue
    
    return sorted(datasets, key=lambda x: (x.domain, x.name))

def list_datasets(
    search_paths: List[str] = None,
    domain_filter: Optional[str] = None,
    min_size_mb: Optional[float] = None,
    format_type: str = "table"
) -> CLIResult:
    """
    List all available datasets with their information.
    
    Args:
        search_paths: List of directories to search for datasets
        domain_filter: Filter by domain (healthcare, finance, ecommerce)
        min_size_mb: Minimum file size in MB
        format_type: Output format (table, json, paths)
        
    Returns:
        CLIResult: Success/failure with dataset list
    """
    try:
        datasets = discover_datasets(search_paths)
        
        if not datasets:
            return CLIResult(
                success=True,
                output="No datasets found in the specified locations.",
                exit_code=0
            )
        
        # Apply filters
        if domain_filter:
            datasets = [d for d in datasets if d.domain.lower() == domain_filter.lower()]
            
        if min_size_mb is not None:
            datasets = [d for d in datasets if d.size_mb >= min_size_mb]
        
        if not datasets:
            return CLIResult(
                success=True,
                output=f"No datasets found matching the specified criteria.",
                exit_code=0
            )
        
        # Format output
        if format_type == "json":
            import json
            dataset_list = []
            for ds in datasets:
                dataset_list.append({
                    "name": ds.name,
                    "path": ds.path,
                    "domain": ds.domain,
                    "size_mb": round(ds.size_mb, 2),
                    "rows": ds.rows,
                    "columns": ds.columns
                })
            output = json.dumps(dataset_list, indent=2)
            
        elif format_type == "paths":
            output = "\n".join([ds.path for ds in datasets])
            
        else:  # table format
            output = format_dataset_table(datasets)
        
        return CLIResult(
            success=True,
            output=output,
            exit_code=0
        )
        
    except Exception as e:
        return CLIResult(
            success=False,
            output="",
            exit_code=1,
            error_message=f"Failed to list datasets: {str(e)}"
        )

def format_dataset_table(datasets: List[DatasetInfo]) -> str:
    """Format datasets as a nice table."""
    
    if not datasets:
        return "No datasets available."
    
    lines = []
    
    # Header
    lines.append("=" * 120)
    lines.append("📊 AVAILABLE DATASETS")
    lines.append("=" * 120)
    lines.append(f"{'Name':<35} {'Domain':<12} {'Rows':<10} {'Cols':<6} {'Size(MB)':<10} {'Path':<40}")
    lines.append("-" * 120)
    
    # Group by domain for better organization
    current_domain = None
    total_datasets = 0
    total_size_mb = 0
    
    for dataset in datasets:
        if dataset.domain != current_domain:
            if current_domain is not None:
                lines.append("")  # Add space between domains
            current_domain = dataset.domain
            lines.append(f"🏷️  {dataset.domain.upper()} DOMAIN")
            lines.append("-" * 120)
        
        # Format size nicely
        if dataset.size_mb < 1:
            size_str = f"{dataset.size_mb*1024:.0f}KB"
        else:
            size_str = f"{dataset.size_mb:.1f}MB"
            
        lines.append(
            f"{dataset.name[:34]:<35} {dataset.domain:<12} {dataset.rows:>9,} "
            f"{dataset.columns:>5} {size_str:<10} {dataset.path[:39]:<40}"
        )
        
        total_datasets += 1
        total_size_mb += dataset.size_mb
    
    lines.append("=" * 120)
    lines.append(f"📈 Summary: {total_datasets} datasets found, {total_size_mb:.1f}MB total")
    lines.append("")
    lines.append("💡 Usage:")
    lines.append("   • Profile a dataset: profile-data --file <path>")
    lines.append("   • Generate new data: generate-data --domain <domain> --size <size>")
    lines.append("   • Filter by domain: list-datasets --domain <healthcare|finance|ecommerce>")
    
    return "\n".join(lines)

def get_dataset_quick_stats(file_path: str) -> Dict:
    """Get quick statistics for a dataset without full profiling."""
    try:
        # Read sample of data for quick analysis
        df_sample = pd.read_csv(file_path, nrows=1000)
        
        stats = {
            "total_rows": len(pd.read_csv(file_path, usecols=[0])),  # Count rows efficiently
            "columns": len(df_sample.columns),
            "missing_percentage": (df_sample.isnull().sum().sum() / (len(df_sample) * len(df_sample.columns))) * 100,
            "numeric_columns": len(df_sample.select_dtypes(include=['int64', 'float64']).columns),
            "text_columns": len(df_sample.select_dtypes(include=['object']).columns),
            "memory_mb": df_sample.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        return stats
    except Exception as e:
        return {"error": str(e)}