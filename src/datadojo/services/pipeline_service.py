"""PipelineService for DataDojo framework.

Executes data preprocessing pipelines with educational guidance.
"""

from typing import Optional, Callable, Dict, Any, List
import time
import pandas as pd

from ..models.pipeline import Pipeline, ExecutionState, GuidanceLevel
from ..models.processing_step import ProcessingStep, CompletionStatus
from sklearn.preprocessing import MinMaxScaler


class PipelineExecutionError(Exception):
    """Raised when pipeline execution fails."""
    pass



class PipelineService:
    """Service for executing data preprocessing pipelines.

    Manages pipeline execution with step-by-step processing, educational
    guidance, and error handling.
    """

    def __init__(self, educational_service=None):
        """Initialize the PipelineService.

        Args:
            educational_service: Optional EducationalService for guidance
        """
        self.educational_service = educational_service
        self._operation_handlers: Dict[str, Callable] = {}
        self._register_default_operations()

    def _register_default_operations(self) -> None:
        """Register default preprocessing operations."""
        # These are placeholder implementations - real operations would be more complex
        self._operation_handlers["data_cleaning"] = self._handle_data_cleaning
        self._operation_handlers["feature_engineering"] = self._handle_feature_engineering
        self._operation_handlers["transformation"] = self._handle_transformation
        self._operation_handlers["validation"] = self._handle_validation

    def register_operation(self, operation_type: str, handler: Callable) -> None:
        """Register a custom operation handler.

        Args:
            operation_type: Type of operation
            handler: Function to handle the operation (receives data, parameters)
        """
        self._operation_handlers[operation_type] = handler

    def _handle_data_cleaning(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Handle data cleaning operation.

        Args:
            data: Input DataFrame
            parameters: Operation parameters

        Returns:
            Cleaned DataFrame
        """
        result = data.copy()

        if "remove_duplicates" in parameters:
            subset = parameters["remove_duplicates"].get("subset")
            result = result.drop_duplicates(subset=subset)

        if "handle_missing" in parameters:
            strategy = parameters["handle_missing"].get("strategy", "drop")
            if strategy == "drop":
                result = result.dropna()
            elif strategy == "fill":
                fill_value = parameters["handle_missing"].get("fill_value", 0)
                result = result.fillna(fill_value)
            elif strategy == "median":
                for col in result.columns:
                    if result[col].isnull().any():
                        median = result[col].median()
                        result[col] = result[col].fillna(median)

        return result

    def _handle_feature_engineering(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Handle feature engineering operation.

        Args:
            data: Input DataFrame
            parameters: Operation parameters

        Returns:
            DataFrame with engineered features
        """
        result = data.copy()

        if "interaction_features" in parameters:
            for new_col, (col1, col2) in parameters["interaction_features"].items():
                if col1 in result.columns and col2 in result.columns:
                    result[new_col] = result[col1] * result[col2]

        return result

    def _handle_transformation(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Handle data transformation operation.

        Args:
            data: Input DataFrame
            parameters: Operation parameters

        Returns:
            Transformed DataFrame
        """
        result = data.copy()

        if "normalize" in parameters:
            cols_to_normalize = parameters["normalize"].get("columns")
            if cols_to_normalize:
                scaler = MinMaxScaler()
                result[cols_to_normalize] = scaler.fit_transform(result[cols_to_normalize])

        if "encode_categorical" in parameters:
            cols_to_encode = parameters["encode_categorical"].get("columns")
            if cols_to_encode:
                result = pd.get_dummies(result, columns=cols_to_encode)

        return result

    def _handle_validation(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Handle data validation operation.

        Args:
            data: Input DataFrame
            parameters: Operation parameters

        Returns:
            Validated DataFrame

        Raises:
            PipelineExecutionError: If validation fails
        """
        if "check_schema" in parameters:
            expected_cols = parameters["check_schema"].get("columns")
            if expected_cols:
                missing_cols = set(expected_cols) - set(data.columns)
                if missing_cols:
                    raise PipelineExecutionError(f"Missing columns: {', '.join(missing_cols)}")

        if "check_quality" in parameters:
            # Placeholder for more complex quality checks
            pass

        return data

    def execute_pipeline(
        self,
        pipeline: Pipeline,
        data: pd.DataFrame,
        interactive: bool = False
    ) -> pd.DataFrame:
        """Execute a complete pipeline.

        Args:
            pipeline: Pipeline to execute
            data: Input data
            interactive: Whether to pause for user input on interactive steps

        Returns:
            Processed data

        Raises:
            PipelineExecutionError: If execution fails
        """
        try:
            pipeline.start_execution()
            result = data.copy()

            for step in pipeline.steps:
                if not self._can_execute_step(step, pipeline):
                    continue

                result = self.execute_step(step, result, pipeline, interactive)

            pipeline.complete_execution()
            return result

        except Exception as e:
            pipeline.fail_execution()
            raise PipelineExecutionError(f"Pipeline execution failed: {str(e)}") from e

    def _can_execute_step(self, step: ProcessingStep, pipeline: Pipeline) -> bool:
        """Check if a step can be executed.

        Args:
            step: ProcessingStep to check
            pipeline: Parent pipeline

        Returns:
            True if step can execute, False otherwise
        """
        # Check if step is already completed or skipped
        if step.completion_status in [CompletionStatus.COMPLETED, CompletionStatus.SKIPPED]:
            return False

        # Check prerequisites
        completed_step_ids = [
            s.id for s in pipeline.steps
            if s.completion_status == CompletionStatus.COMPLETED
        ]
        return step.is_ready(completed_step_ids)

    def execute_step(
        self,
        step: ProcessingStep,
        data: pd.DataFrame,
        pipeline: Pipeline,
        interactive: bool = False
    ) -> pd.DataFrame:
        """Execute a single processing step.

        Args:
            step: ProcessingStep to execute
            data: Input data
            pipeline: Parent pipeline
            interactive: Whether to pause for user input

        Returns:
            Processed data

        Raises:
            PipelineExecutionError: If step execution fails
        """
        try:
            step.start()
            start_time = time.time()

            # Provide educational guidance if available
            if pipeline.educational_mode and self.educational_service:
                self._provide_guidance(step, pipeline)

            # Handle interactive steps
            if step.interactive and interactive:
                self._handle_interactive_step(step, data, pipeline)

            # Execute the operation
            handler = self._operation_handlers.get(step.operation_type.value)
            if not handler:
                raise PipelineExecutionError(
                    f"No handler registered for operation type: {step.operation_type}"
                )

            result = handler(data, step.parameters)

            # Mark step as completed
            execution_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
            step.complete(execution_time)

            return result

        except Exception as e:
            step.fail()
            raise PipelineExecutionError(f"Step '{step.name}' failed: {str(e)}") from e

    def _provide_guidance(self, step: ProcessingStep, pipeline: Pipeline) -> None:
        """Provide educational guidance for a step.

        Args:
            step: ProcessingStep being executed
            pipeline: Parent pipeline
        """
        if not self.educational_service:
            return

        # Request guidance based on pipeline settings
        for concept in step.learned_concepts:
            try:
                # This would call the educational service to display guidance
                # Placeholder for now
                pass
            except Exception:
                # Don't fail execution if guidance fails
                pass

    def _handle_interactive_step(
        self,
        step: ProcessingStep,
        data: pd.DataFrame,
        pipeline: Pipeline
    ) -> None:
        """Handle user interaction for interactive steps.

        Args:
            step: Interactive ProcessingStep
            data: Current data state
            pipeline: Parent pipeline
        """
        # Placeholder for interactive handling
        # Would prompt user for input, display data preview, etc.
        pass

    def validate_pipeline(self, pipeline: Pipeline) -> List[str]:
        """Validate a pipeline configuration.

        Args:
            pipeline: Pipeline to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not pipeline.steps:
            errors.append("Pipeline has no steps")

        # Check for circular dependencies
        visited = set()
        for step in pipeline.steps:
            if not self._check_dependencies(step, pipeline.steps, visited):
                errors.append(f"Circular dependency detected for step: {step.id}")

        # Check that all operation types have handlers
        for step in pipeline.steps:
            if step.operation_type.value not in self._operation_handlers:
                errors.append(
                    f"No handler for operation type '{step.operation_type}' in step '{step.id}'"
                )

        return errors

    def _check_dependencies(
        self,
        step: ProcessingStep,
        all_steps: List[ProcessingStep],
        visited: set
    ) -> bool:
        """Check for circular dependencies.

        Args:
            step: Step to check
            all_steps: All steps in pipeline
            visited: Set of visited step IDs

        Returns:
            True if no circular dependency, False otherwise
        """
        if step.id in visited:
            return False

        visited.add(step.id)

        for prereq_id in step.prerequisites:
            prereq = next((s for s in all_steps if s.id == prereq_id), None)
            if prereq and not self._check_dependencies(prereq, all_steps, visited.copy()):
                return False

        return True
