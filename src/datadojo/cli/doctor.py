import sys
from pathlib import Path

class CLIResult:
    def __init__(self, success: bool, output: str, exit_code: int, error_message: Optional[str] = None):
        self.success = success
        self.output = output
        self.exit_code = exit_code
        self.error_message = error_message

def doctor():
    """Checks the DataDojo environment and reports any issues."""
    output = []
    errors = 0

    # Check Python version
    output.append("Checking Python version...")
    if sys.version_info.major == 3 and sys.version_info.minor >= 11:
        output.append(f"[OK] Python version {sys.version_info.major}.{sys.version_info.minor} is compatible (>=3.11).")
    else:
        output.append(f"[ERROR] Python version {sys.version_info.major}.{sys.version_info.minor} is not compatible. DataDojo requires Python >= 3.11.")
        errors += 1

    # Check if datadojo is installed
    output.append("\nChecking DataDojo installation...")
    try:
        import datadojo
        output.append("[OK] DataDojo package is installed and accessible.")
    except ImportError:
        output.append("[ERROR] DataDojo package is not installed or not in the Python path.")
        errors += 1

    # Check if datasets directory exists
    output.append("\nChecking for datasets directory...")
    datasets_path = Path("datasets")
    if datasets_path.exists() and datasets_path.is_dir():
        output.append("[OK] 'datasets' directory found.")
    else:
        output.append("[ERROR] 'datasets' directory not found. Please create it in the project root.")
        errors += 1

    # Summary
    output.append("\n" + "="*30)
    if errors == 0:
        output.append("DataDojo environment looks good!")
        return CLIResult(success=True, output="\n".join(output), exit_code=0)
    else:
        output.append(f"Found {errors} issue(s). Please resolve them and run 'datadojo doctor' again.")
        return CLIResult(success=False, output="\n".join(output), exit_code=1)
