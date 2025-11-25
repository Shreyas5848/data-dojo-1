"""PipelineInterface implementation for DataDojo framework.

Interface for building and executing preprocessing pipelines.
"""

from typing import List, Dict, Any
import pandas as pd
import time

from datadojo.dojo_api import PipelineInterface, ExecutionResult
from ..models.pipeline import Pipeline
from ..models.processing_step import ProcessingStep, OperationType, CompletionStatus
from ..services.educational_service import EducationalService
from ..services.pipeline_service import PipelineService


class PipelineImpl(PipelineInterface):
    """Interface for building and executing preprocessing pipelines.

    Provides method chaining for pipeline construction and execution
    with educational guidance.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        educational_service: EducationalService
    ):
        """Initialize PipelineImpl.

        Args:
            pipeline: Pipeline model instance
            educational_service: Service for educational features
        """
        self._pipeline = pipeline
        self._educational_service = educational_service
        self._pipeline_service = PipelineService(educational_service)
        self._step_counter = 0

    def _convert_operation_type(self, operation_type: str) -> OperationType:
        """Convert string operation type to OperationType enum.

        Args:
            operation_type: String operation type

        Returns:
            OperationType enum

        Raises:
            ValueError: If operation type is not supported
        """
        operation_map = {
            "data_cleaning": OperationType.DATA_CLEANING,
            "feature_engineering": OperationType.FEATURE_ENGINEERING,
            "transformation": OperationType.TRANSFORMATION,
            "validation": OperationType.VALIDATION
        }

        if operation_type not in operation_map:
            raise ValueError(f"Unsupported operation type: {operation_type}")

        return operation_map[operation_type]

    def add_step(
        self,
        operation_type: str,
        interactive: bool = False,
        **kwargs
    ) -> 'PipelineImpl':
        """Add a preprocessing step to the pipeline.

        Args:
            operation_type: Type of operation (data_cleaning, feature_engineering, etc.)
            interactive: Whether step requires user interaction
            **kwargs: Operation-specific parameters

        Returns:
            Self for method chaining

        Raises:
            ValueError: If operation_type is not supported
        """
        # Convert operation type
        op_type = self._convert_operation_type(operation_type)

        # Generate step ID
        self._step_counter += 1
        step_id = f"{self._pipeline.id}_step_{self._step_counter}"

        # Create processing step
        step = ProcessingStep(
            id=step_id,
            operation_type=op_type,
            name=kwargs.pop("name", f"{operation_type.replace('_', ' ').title()} Step"),
            description=kwargs.pop("description", f"Perform {operation_type}"),
            parameters=kwargs,
            interactive=interactive
        )

        # Add to pipeline
        self._pipeline.add_step(step)

        return self

    def execute(self, data: Any) -> ExecutionResult:
        """Execute the complete pipeline.

        Args:
            data: Input dataset to process

        Returns:
            Execution result with processed data and metrics

        Raises:
            RuntimeError: If pipeline execution fails
        """
        try:
            # Ensure data is a DataFrame
            if not isinstance(data, pd.DataFrame):
                if hasattr(data, 'to_dataframe'):
                    data = data.to_dataframe()
                else:
                    raise ValueError("Data must be a pandas DataFrame")

            # Track execution time
            start_time = time.time()

            # Execute pipeline through service
            result_data = self._pipeline_service.execute_pipeline(
                pipeline=self._pipeline,
                data=data,
                interactive=False  # For now, non-interactive execution
            )

            # Calculate metrics
            execution_time_ms = int((time.time() - start_time) * 1000)
            completed_steps = self._pipeline.get_completed_steps()
            steps_completed = len(completed_steps)

            # Collect learned concepts
            concepts_learned = []
            for step in completed_steps:
                concepts_learned.extend(step.learned_concepts)
            concepts_learned = list(set(concepts_learned))  # Remove duplicates

            return ExecutionResult(
                success=True,
                processed_data=result_data,
                execution_time_ms=execution_time_ms,
                steps_completed=steps_completed,
                concepts_learned=concepts_learned,
                error_message=None
            )

        except Exception as e:
            # Return error result
            return ExecutionResult(
                success=False,
                processed_data=None,
                execution_time_ms=0,
                steps_completed=0,
                concepts_learned=[],
                error_message=str(e)
            )

    def get_available_operations(self) -> List[str]:
        """Get list of supported preprocessing operations.

        Returns:
            List of operation type strings
        """
        return [
            "data_cleaning",
            "feature_engineering",
            "transformation",
            "validation"
        ]

    def preview_next_step(self) -> Dict[str, Any]:
        """Preview the next step without executing.

        Returns:
            Information about the next operation including guidance
        """
        # Find next pending step
        next_step = None
        for step in self._pipeline.steps:
            if step.completion_status == CompletionStatus.PENDING:
                # Check if prerequisites are met
                completed_ids = [
                    s.id for s in self._pipeline.steps
                    if s.completion_status == CompletionStatus.COMPLETED
                ]
                if step.is_ready(completed_ids):
                    next_step = step
                    break

        if next_step is None:
            return {
                "has_next": False,
                "message": "No more steps to execute"
            }

        # Build preview information
        preview = {
            "has_next": True,
            "step_id": next_step.id,
            "operation_type": next_step.operation_type.value,
            "name": next_step.name,
            "description": next_step.description,
            "interactive": next_step.interactive,
            "parameters": next_step.parameters,
            "learned_concepts": next_step.learned_concepts
        }

        # Add educational guidance if available
        if self._pipeline.educational_mode and next_step.learned_concepts:
            guidance = []
            for concept in next_step.learned_concepts:
                content = self._educational_service.get_concept_explanation(concept)
                if content:
                    guidance.append({
                        "concept_id": content.concept_id,
                        "title": content.title,
                        "summary": content.get_summary()
                    })
            preview["guidance"] = guidance

        return preview
