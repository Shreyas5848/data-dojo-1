"""Custom exceptions with educational context for DataDojo.

Provides informative error messages with learning guidance to help
students understand and fix issues.
"""

from typing import Optional, List


class DataDojoException(Exception):
    """Base exception for all DataDojo errors with educational context."""

    def __init__(
        self,
        message: str,
        educational_hint: Optional[str] = None,
        suggested_actions: Optional[List[str]] = None,
        related_concepts: Optional[List[str]] = None
    ):
        """Initialize DataDojo exception.

        Args:
            message: Error message
            educational_hint: Educational explanation of the error
            suggested_actions: List of suggested fixes
            related_concepts: Related concepts to learn
        """
        self.message = message
        self.educational_hint = educational_hint
        self.suggested_actions = suggested_actions or []
        self.related_concepts = related_concepts or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the complete error message with educational context."""
        parts = [f"Error: {self.message}"]

        if self.educational_hint:
            parts.append(f"\nWhy this happened: {self.educational_hint}")

        if self.suggested_actions:
            parts.append("\nSuggested actions:")
            for i, action in enumerate(self.suggested_actions, 1):
                parts.append(f"  {i}. {action}")

        if self.related_concepts:
            parts.append(f"\nRelated concepts: {', '.join(self.related_concepts)}")

        return "\n".join(parts)


class ProjectNotFoundError(DataDojoException):
    """Raised when a requested project cannot be found."""

    def __init__(self, project_id: str):
        super().__init__(
            message=f"Project '{project_id}' not found",
            educational_hint=(
                "The project you're trying to load doesn't exist in the system. "
                "This could be because the project ID is incorrect or the project "
                "hasn't been created yet."
            ),
            suggested_actions=[
                "Use dojo.list_projects() to see all available projects",
                "Check the project ID for typos",
                "Create the project first if it doesn't exist"
            ]
        )


class DataValidationError(DataDojoException):
    """Raised when data fails validation checks."""

    def __init__(
        self,
        validation_name: str,
        details: str,
        column: Optional[str] = None
    ):
        column_info = f" in column '{column}'" if column else ""
        super().__init__(
            message=f"Data validation failed: {validation_name}{column_info}",
            educational_hint=(
                f"{details}. Data quality is crucial for reliable analysis. "
                "This validation ensures your data meets expected standards."
            ),
            suggested_actions=[
                "Review the validation rule requirements",
                "Check for missing or incorrect values",
                "Consider data cleaning operations",
                "Use df.describe() and df.info() to understand your data"
            ],
            related_concepts=["data_quality", "data_validation", "data_cleaning"]
        )


class PipelineExecutionError(DataDojoException):
    """Raised when pipeline execution fails."""

    def __init__(self, step_name: str, error_details: str):
        super().__init__(
            message=f"Pipeline failed at step '{step_name}': {error_details}",
            educational_hint=(
                "A pipeline consists of sequential data processing steps. When one "
                "step fails, the entire pipeline stops to prevent cascading errors."
            ),
            suggested_actions=[
                "Check the input data for this step",
                "Verify step parameters are correct",
                "Review previous step outputs",
                "Use pipeline.preview_next_step() to see what's coming",
                "Run steps individually for debugging"
            ],
            related_concepts=["data_pipeline", "error_handling", "debugging"]
        )


class MissingDataError(DataDojoException):
    """Raised when required data is missing."""

    def __init__(self, column: str, percentage: float):
        super().__init__(
            message=f"Column '{column}' has {percentage:.1f}% missing values",
            educational_hint=(
                "Missing data is common in real-world datasets. The approach to "
                "handle it depends on why the data is missing and how much is missing. "
                "Common strategies include deletion, imputation, or using algorithms "
                "that handle missing values."
            ),
            suggested_actions=[
                "Decide if this level of missingness is acceptable",
                "Consider removing the column if >50% missing",
                "Impute values using mean, median, or mode",
                "Use forward/backward fill for time series",
                "Investigate why data is missing (MCAR, MAR, or MNAR)"
            ],
            related_concepts=["missing_data", "imputation", "data_quality"]
        )


class InvalidParameterError(DataDojoException):
    """Raised when invalid parameters are provided."""

    def __init__(self, parameter: str, value: any, expected: str):
        super().__init__(
            message=f"Invalid parameter '{parameter}' = {value}",
            educational_hint=(
                f"Expected {expected}. Parameters control how operations behave. "
                "Using correct parameters ensures the operation does what you intend."
            ),
            suggested_actions=[
                "Check the parameter documentation",
                "Verify the parameter type and format",
                "Look at example usage",
                "Use default values if unsure"
            ]
        )


class DataTypeError(DataDojoException):
    """Raised when data has incorrect types."""

    def __init__(self, column: str, expected_type: str, actual_type: str):
        super().__init__(
            message=f"Column '{column}' has type '{actual_type}', expected '{expected_type}'",
            educational_hint=(
                "Data types determine what operations can be performed on data. "
                "For example, you can't calculate mean of text data. Converting "
                "types (type casting) is often necessary during data preparation."
            ),
            suggested_actions=[
                f"Convert column to {expected_type}: df['{column}'].astype('{expected_type}')",
                "Check for non-numeric values if converting to numeric",
                "Use pd.to_numeric() with errors='coerce' for safe conversion",
                "Investigate why the type is unexpected"
            ],
            related_concepts=["data_types", "type_conversion", "data_preparation"]
        )


class ProgressNotFoundError(DataDojoException):
    """Raised when student progress cannot be found."""

    def __init__(self, student_id: str, project_id: str):
        super().__init__(
            message=f"No progress found for student '{student_id}' on project '{project_id}'",
            educational_hint=(
                "Progress tracking helps you resume your learning journey. If no "
                "progress exists, you may be starting fresh or the IDs might be incorrect."
            ),
            suggested_actions=[
                "Verify student ID and project ID are correct",
                "Start the project to create initial progress",
                "Check if progress was saved properly"
            ]
        )


class ConceptNotFoundError(DataDojoException):
    """Raised when an educational concept is not found."""

    def __init__(self, concept_id: str):
        super().__init__(
            message=f"Concept '{concept_id}' not found in educational database",
            educational_hint=(
                "Educational concepts provide explanations and examples for data "
                "science topics. This concept may not be available yet."
            ),
            suggested_actions=[
                "Check the concept ID for typos",
                "Browse available concepts",
                "Search online documentation for this topic",
                "Ask for help or contribute this concept to the database"
            ]
        )


class StorageError(DataDojoException):
    """Raised when storage operations fail."""

    def __init__(self, operation: str, details: str):
        super().__init__(
            message=f"Storage {operation} failed: {details}",
            educational_hint=(
                "DataDojo stores your progress and projects on disk. Storage errors "
                "can occur due to permissions, disk space, or file corruption."
            ),
            suggested_actions=[
                "Check file system permissions",
                "Verify disk space is available",
                "Check if storage directory exists",
                "Try with a different storage location"
            ]
        )


class ConfigurationError(DataDojoException):
    """Raised when configuration is invalid."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            educational_hint=(
                "Configuration settings control how DataDojo behaves. Invalid "
                "configuration can prevent the system from working properly."
            ),
            suggested_actions=[
                "Check configuration file syntax",
                "Verify all required settings are present",
                "Review environment variables",
                "Use default configuration as reference"
            ]
        )
