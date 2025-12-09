"""
Data Processing Limits and Smart Sampling Utilities
Prevents UI freezing by enforcing sensible limits on data operations.
"""

import os
from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path


@dataclass
class DataLimits:
    """Configuration for data processing limits."""
    
    # Row limits
    QUICK_PROFILE_MAX_ROWS: int = 5_000
    FULL_PROFILE_MAX_ROWS: int = 50_000
    VISUALIZATION_MAX_ROWS: int = 10_000
    NOTEBOOK_PREVIEW_MAX_ROWS: int = 5_000
    DATAFRAME_DISPLAY_MAX_ROWS: int = 1_000
    
    # File size limits (MB)
    LARGE_FILE_THRESHOLD_MB: float = 50.0
    MAX_FILE_SIZE_MB: float = 500.0
    
    # Memory limits
    MAX_MEMORY_MB: float = 500.0
    
    # Timeout limits (seconds)
    PROFILE_TIMEOUT_SECONDS: int = 60
    GENERATION_TIMEOUT_SECONDS: int = 120


# Global instance
LIMITS = DataLimits()


def get_file_size_mb(path: str) -> float:
    """Get file size in megabytes."""
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except:
        return 0.0


def estimate_rows_from_size(path: str, avg_row_bytes: int = 200) -> int:
    """Estimate number of rows from file size."""
    size_bytes = os.path.getsize(path)
    return max(1, size_bytes // avg_row_bytes)


def get_sample_size(total_rows: int, operation: str = "profile") -> Tuple[int, bool]:
    """
    Determine appropriate sample size for an operation.
    
    Returns:
        Tuple of (sample_size, is_sampled)
    """
    limits = {
        "quick_profile": LIMITS.QUICK_PROFILE_MAX_ROWS,
        "full_profile": LIMITS.FULL_PROFILE_MAX_ROWS,
        "visualization": LIMITS.VISUALIZATION_MAX_ROWS,
        "notebook": LIMITS.NOTEBOOK_PREVIEW_MAX_ROWS,
        "display": LIMITS.DATAFRAME_DISPLAY_MAX_ROWS,
    }
    
    max_rows = limits.get(operation, LIMITS.FULL_PROFILE_MAX_ROWS)
    
    if total_rows <= max_rows:
        return total_rows, False
    else:
        return max_rows, True


def check_file_processable(path: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a file can be processed.
    
    Returns:
        Tuple of (can_process, warning_message)
    """
    size_mb = get_file_size_mb(path)
    
    if size_mb > LIMITS.MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f} MB). Maximum supported: {LIMITS.MAX_FILE_SIZE_MB} MB"
    
    if size_mb > LIMITS.LARGE_FILE_THRESHOLD_MB:
        return True, f"Large file detected ({size_mb:.1f} MB). Processing will use sampling for better performance."
    
    return True, None


def get_chunked_reader_params(path: str) -> dict:
    """Get parameters for chunked CSV reading."""
    size_mb = get_file_size_mb(path)
    
    if size_mb > 100:
        chunksize = 10_000
    elif size_mb > 50:
        chunksize = 25_000
    elif size_mb > 10:
        chunksize = 50_000
    else:
        chunksize = None  # Read all at once
    
    return {"chunksize": chunksize}


def smart_read_csv(pd, path: str, max_rows: Optional[int] = None, show_progress=None):
    """
    Smart CSV reader that handles large files gracefully.
    
    Args:
        pd: pandas module
        path: Path to CSV file
        max_rows: Maximum rows to read (None = use defaults)
        show_progress: Optional Streamlit progress bar
    
    Returns:
        DataFrame (possibly sampled)
    """
    size_mb = get_file_size_mb(path)
    
    # For small files, just read directly
    if size_mb < 10:
        df = pd.read_csv(path)
        if max_rows and len(df) > max_rows:
            return df.sample(n=max_rows, random_state=42)
        return df
    
    # For larger files, use nrows or chunked reading
    if max_rows:
        # Read only what we need plus some extra for random sampling
        read_rows = min(max_rows * 2, 100_000)
        df = pd.read_csv(path, nrows=read_rows)
        if len(df) > max_rows:
            return df.sample(n=max_rows, random_state=42)
        return df
    
    # Default: read with limit
    df = pd.read_csv(path, nrows=LIMITS.FULL_PROFILE_MAX_ROWS)
    return df


def format_processing_message(total_rows: int, sample_size: int, operation: str) -> str:
    """Generate a user-friendly message about sampling."""
    if total_rows <= sample_size:
        return f"Processing all {total_rows:,} rows"
    
    percentage = (sample_size / total_rows) * 100
    return f"Using {sample_size:,} row sample ({percentage:.1f}%) from {total_rows:,} total rows for faster {operation}"
