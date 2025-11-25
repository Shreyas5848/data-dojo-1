"""CLI command for validating dataset quality."""

import json
from typing import Optional
from pathlib import Path
import pandas as pd

class CLIResult:
    def __init__(self, success: bool, output: str, exit_code: int, error_message: Optional[str] = None):
        self.success = success
        self.output = output
        self.exit_code = exit_code
        self.error_message = error_message


def validate_data(
    dojo,
    data_file: str,
    validation_rules: Optional[str] = None,
    report_format: str = "summary"
) -> CLIResult:
    """Validate dataset quality.

    Args:
        dojo: Dojo instance
        data_file: Path to dataset for validation
        validation_rules: Custom validation rules file (optional)
        report_format: Output format (summary|detailed|json)

    Returns:
        CLI result with validation report
    """
    try:
        # Validate input file exists
        data_path = Path(data_file)
        if not data_path.exists():
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message=f"Data file not found: {data_file}"
            )

        # Load data
        try:
            if data_path.suffix == '.csv':
                data = pd.read_csv(data_path)
            elif data_path.suffix == '.json':
                data = pd.read_json(data_path)
            elif data_path.suffix == '.parquet':
                data = pd.read_parquet(data_path)
            else:
                return CLIResult(
                    success=False,
                    output="",
                    exit_code=1,
                    error_message=f"Unsupported file format: {data_path.suffix}"
                )
        except Exception as e:
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message=f"Failed to load data file: {str(e)}"
            )

        # Load custom validation rules if provided
        custom_rules = {}
        if validation_rules:
            rules_path = Path(validation_rules)
            if rules_path.exists():
                try:
                    with open(rules_path, 'r') as f:
                        custom_rules = json.load(f)
                except Exception as e:
                    return CLIResult(
                        success=False,
                        output="",
                        exit_code=1,
                        error_message=f"Failed to load validation rules: {str(e)}"
                    )

        # Perform validation
        validation_results = _perform_validation(data, custom_rules)

        # Format output
        if report_format == "json":
            output = json.dumps(validation_results, indent=2)
        elif report_format == "detailed":
            output = _format_detailed_report(data_file, validation_results)
        else:  # summary
            output = _format_summary_report(data_file, validation_results)

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
            error_message=f"Failed to validate data: {str(e)}"
        )


def _perform_validation(data: pd.DataFrame, custom_rules: dict) -> dict:
    """Perform data validation checks.

    Args:
        data: DataFrame to validate
        custom_rules: Custom validation rules

    Returns:
        Dictionary with validation results
    """
    results = {
        "total_rows": len(data),
        "total_columns": len(data.columns),
        "issues": [],
        "warnings": [],
        "info": []
    }

    # Check for missing values
    missing = data.isnull().sum()
    if missing.any():
        for col in missing[missing > 0].index:
            count = missing[col]
            percentage = (count / len(data)) * 100
            results["issues"].append({
                "type": "missing_values",
                "column": col,
                "count": int(count),
                "percentage": round(percentage, 2),
                "message": f"Column '{col}' has {count} missing values ({percentage:.2f}%)"
            })

    # Check for duplicates
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        results["warnings"].append({
            "type": "duplicates",
            "count": int(duplicates),
            "message": f"Found {duplicates} duplicate rows"
        })

    # Check data types
    for col in data.columns:
        dtype = str(data[col].dtype)
        results["info"].append({
            "column": col,
            "dtype": dtype,
            "unique_values": int(data[col].nunique())
        })

    # Check for outliers in numerical columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        if outliers > 0:
            percentage = (outliers / len(data)) * 100
            results["warnings"].append({
                "type": "outliers",
                "column": col,
                "count": int(outliers),
                "percentage": round(percentage, 2),
                "message": f"Column '{col}' has {outliers} potential outliers ({percentage:.2f}%)"
            })

    # Check skewness in numerical columns
    for col in numeric_cols:
        try:
            skewness = data[col].skew()
            if abs(skewness) > 1:
                results["warnings"].append({
                    "type": "skewness",
                    "column": col,
                    "skewness": round(float(skewness), 2),
                    "message": f"Column '{col}' has high skewness: {skewness:.2f}"
                })
        except:
            pass

    # Apply custom validation rules
    if custom_rules:
        for rule_name, rule_spec in custom_rules.items():
            # Placeholder for custom rule execution
            pass

    return results


def _format_summary_report(data_file: str, results: dict) -> str:
    """Format validation results as summary report."""
    lines = []

    lines.append("=" * 80)
    lines.append("Data Validation Summary")
    lines.append("=" * 80)
    lines.append(f"File: {data_file}")
    lines.append(f"Rows: {results['total_rows']:,}")
    lines.append(f"Columns: {results['total_columns']}")
    lines.append("")

    # Issues
    issue_count = len(results['issues'])
    warning_count = len(results['warnings'])

    if issue_count == 0 and warning_count == 0:
        lines.append("✓ No data quality issues found!")
    else:
        if issue_count > 0:
            lines.append(f"Issues Found: {issue_count}")
            for issue in results['issues'][:5]:  # Show first 5
                lines.append(f"  ✗ {issue['message']}")
            if issue_count > 5:
                lines.append(f"  ... and {issue_count - 5} more issues")
            lines.append("")

        if warning_count > 0:
            lines.append(f"Warnings: {warning_count}")
            for warning in results['warnings'][:5]:  # Show first 5
                lines.append(f"  ⚠ {warning['message']}")
            if warning_count > 5:
                lines.append(f"  ... and {warning_count - 5} more warnings")
            lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def _format_detailed_report(data_file: str, results: dict) -> str:
    """Format validation results as detailed report."""
    lines = []

    lines.append("=" * 80)
    lines.append("Detailed Data Validation Report")
    lines.append("=" * 80)
    lines.append(f"File: {data_file}")
    lines.append(f"Total Rows: {results['total_rows']:,}")
    lines.append(f"Total Columns: {results['total_columns']}")
    lines.append("")

    # Column information
    lines.append("Column Information:")
    lines.append(f"{'Column':<30} {'Type':<15} {'Unique Values':<15}")
    lines.append("-" * 80)
    for info in results['info']:
        lines.append(f"{info['column']:<30} {info['dtype']:<15} {info['unique_values']:<15}")
    lines.append("")

    # Issues
    if results['issues']:
        lines.append("Data Quality Issues:")
        for issue in results['issues']:
            lines.append(f"  ✗ [{issue['type']}] {issue['message']}")
        lines.append("")

    # Warnings
    if results['warnings']:
        lines.append("Warnings:")
        for warning in results['warnings']:
            lines.append(f"  ⚠ [{warning['type']}] {warning['message']}")
        lines.append("")

    if not results['issues'] and not results['warnings']:
        lines.append("✓ No data quality issues or warnings found!")
        lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)
