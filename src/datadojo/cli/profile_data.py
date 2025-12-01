"""
Profile Data Command for DataDojo CLI  
Provides intelligent data profiling capabilities through the command line.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from ..utils.intelligent_profiler import IntelligentProfiler, quick_profile

class CLIResult:
    def __init__(self, success: bool, output: str, exit_code: int, error_message: Optional[str] = None):
        self.success = success
        self.output = output
        self.exit_code = exit_code
        self.error_message = error_message


def profile_data(
    file_path: str,
    output_path: Optional[str] = None,
    format_type: str = "text",
    dataset_name: Optional[str] = None
) -> CLIResult:
    """
    Profile a dataset with AI-powered analysis and recommendations.
    
    Args:
        file_path: Path to the CSV file to profile
        output_path: Optional output path for the report
        format_type: Report format ('text' or 'json')
        dataset_name: Custom name for the dataset (defaults to filename)
    
    Returns:
        CLIResult: Success/failure status with profile summary
    """
    try:
        file_path_obj = Path(file_path)
        
        # Validate file exists
        if not file_path_obj.exists():
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message=f"File not found: {file_path}"
            )
        
        # Validate file type
        if not file_path_obj.suffix.lower() == '.csv':
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message="Only CSV files are supported for profiling"
            )
        
        # Load the dataset
        try:
            df = pd.read_csv(file_path_obj)
        except Exception as e:
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message=f"Failed to load CSV file: {str(e)}"
            )
        
        if df.empty:
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message="The dataset is empty"
            )
        
        # Profile the dataset
        dataset_name = dataset_name or file_path_obj.stem
        profiler = IntelligentProfiler()
        profile = profiler.profile_dataset(df, dataset_name)
        
        # Generate output based on format
        if format_type == "json":
            if not output_path:
                output_path = f"{file_path_obj.stem}_profile.json"
            
            profiler.export_profile_json(profile, output_path)
            
            summary = f"📊 Profile exported to JSON: {output_path}\n"
            summary += f"🔍 Analyzed {profile.shape[0]:,} rows × {profile.shape[1]} columns\n"
            summary += f"🎯 Overall Quality Score: {profile.overall_quality_score:.1%}"
            
        else:  # text format
            report = profiler.generate_report(profile, output_path)
            
            if output_path:
                summary = f"📄 Profile report saved to: {output_path}\n"
                summary += f"🔍 Analyzed {profile.shape[0]:,} rows × {profile.shape[1]} columns\n"
                summary += f"🎯 Overall Quality Score: {profile.overall_quality_score:.1%}"
            else:
                summary = report
        
        # Add key insights to summary
        if format_type != "text" or output_path:
            if profile.recommendations:
                summary += "\n\n🚀 Top Recommendations:\n"
                for i, rec in enumerate(profile.recommendations[:3], 1):
                    summary += f"  {i}. {rec}\n"
            
            if profile.business_insights:
                summary += "\n💡 Key Insights:\n"
                for insight in profile.business_insights[:3]:
                    summary += f"  • {insight}\n"
        
        return CLIResult(
            success=True,
            output=summary,
            exit_code=0
        )
        
    except Exception as e:
        return CLIResult(
            success=False,
            output="",
            exit_code=1,
            error_message=f"Failed to profile data: {str(e)}"
        )


def validate_data(
    file_path: str,
    rules: Optional[dict] = None
) -> CLIResult:
    """
    Validate data against specific quality rules.
    
    Args:
        file_path: Path to the CSV file to validate
        rules: Dictionary of validation rules
        
    Returns:
        Response: Validation results
    """
    try:
        # Quick validation using the profiler
        profile_result = profile_data(file_path, format_type="text")
        
        if not profile_result.success:
            return profile_result
        
        # Load data for validation
        df = pd.read_csv(file_path)
        profiler = IntelligentProfiler()
        profile = profiler.profile_dataset(df, Path(file_path).stem)
        
        # Basic validation checks
        validation_results = []
        issues_found = 0
        
        # Check for missing data
        high_missing_cols = [
            name for name, col_profile in profile.column_profiles.items() 
            if col_profile.null_percentage > 20
        ]
        
        if high_missing_cols:
            issues_found += len(high_missing_cols)
            validation_results.append(f"❌ High missing data in columns: {', '.join(high_missing_cols)}")
        else:
            validation_results.append("✅ Missing data levels are acceptable")
        
        # Check for duplicates
        if profile.duplicate_percentage > 5:
            issues_found += 1
            validation_results.append(f"❌ High duplicate rate: {profile.duplicate_percentage:.1f}%")
        else:
            validation_results.append("✅ Duplicate levels are acceptable")
        
        # Check overall quality
        if profile.overall_quality_score < 0.7:
            issues_found += 1
            validation_results.append(f"❌ Overall quality score is low: {profile.overall_quality_score:.1%}")
        else:
            validation_results.append(f"✅ Good overall quality score: {profile.overall_quality_score:.1%}")
        
        # Summary
        summary = f"🔍 Validation Results for {Path(file_path).name}\n"
        summary += f"📊 Dataset: {profile.shape[0]:,} rows × {profile.shape[1]} columns\n"
        summary += f"⚠️  Issues found: {issues_found}\n\n"
        summary += "\n".join(validation_results)
        
        return CLIResult(
            success=True,
            output=summary,
            exit_code=0
        )
        
    except Exception as e:
        return CLIResult(
            success=False,
            output="",
            exit_code=1,
            error_message=f"Validation failed: {str(e)}"
        )