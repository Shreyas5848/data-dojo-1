"""CLI command for executing preprocessing pipelines."""

import json
from typing import List, Optional
from pathlib import Path
import pandas as pd

class CLIResult:
    def __init__(self, success: bool, output: str, exit_code: int, error_message: Optional[str] = None):
        self.success = success
        self.output = output
        self.exit_code = exit_code
        self.error_message = error_message


def pipeline_cmd(
    dojo,
    operations: List[str],
    input_file: str,
    output_file: Optional[str] = None,
    config_file: Optional[str] = None,
    educational_mode: bool = False
) -> CLIResult:
    """Execute a preprocessing pipeline.

    Args:
        dojo: Dojo instance
        operations: List of preprocessing operations
        input_file: Path to input dataset
        output_file: Path for processed output (optional)
        config_file: Pipeline configuration (optional)
        educational_mode: Enable educational guidance

    Returns:
        CLI result with pipeline execution summary
    """
    try:
        # Validate input file
        input_path = Path(input_file)
        if not input_path.exists():
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message=f"Input file not found: {input_file}"
            )

        # Load input data
        try:
            if input_path.suffix == '.csv':
                data = pd.read_csv(input_path)
            elif input_path.suffix == '.json':
                data = pd.read_json(input_path)
            elif input_path.suffix == '.parquet':
                data = pd.read_parquet(input_path)
            else:
                return CLIResult(
                    success=False,
                    output="",
                    exit_code=1,
                    error_message=f"Unsupported file format: {input_path.suffix}"
                )
        except Exception as e:
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message=f"Failed to load input file: {str(e)}"
            )

        # Load configuration if provided
        config = {}
        if config_file:
            config_path = Path(config_file)
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                except Exception as e:
                    return CLIResult(
                        success=False,
                        output="",
                        exit_code=1,
                        error_message=f"Failed to load config file: {str(e)}"
                    )

        # Create a temporary project for pipeline execution
        from ..core.dojo import Dojo
        from ..models.pipeline import Pipeline, GuidanceLevel
        from ..services.pipeline_service import PipelineService
        from ..services.educational_service import EducationalService

        educational_service = EducationalService()
        pipeline_service = PipelineService(educational_service if educational_mode else None)

        # Create pipeline
        guidance = GuidanceLevel.DETAILED if educational_mode else GuidanceLevel.NONE
        pipeline = Pipeline(
            id="cli_pipeline",
            name="CLI Pipeline",
            educational_mode=educational_mode,
            guidance_level=guidance
        )

        # Add steps from operations
        from ..models.processing_step import ProcessingStep, OperationType

        operation_type_map = {
            "data_cleaning": OperationType.DATA_CLEANING,
            "feature_engineering": OperationType.FEATURE_ENGINEERING,
            "transformation": OperationType.TRANSFORMATION,
            "validation": OperationType.VALIDATION
        }

        for i, op in enumerate(operations):
            # Get operation config from config file
            op_config = config.get(op, {})

            # Map operation string to type
            op_type = operation_type_map.get(op)
            if not op_type:
                return CLIResult(
                    success=False,
                    output="",
                    exit_code=1,
                    error_message=f"Unsupported operation: {op}. Must be one of: {', '.join(operation_type_map.keys())}"
                )

            step = ProcessingStep(
                id=f"step_{i+1}",
                operation_type=op_type,
                name=f"{op.replace('_', ' ').title()} Step",
                description=f"Perform {op}",
                parameters=op_config
            )
            pipeline.add_step(step)

        # Execute pipeline
        try:
            result_data = pipeline_service.execute_pipeline(
                pipeline=pipeline,
                data=data,
                interactive=False
            )

            # Get execution metrics
            completed_steps = pipeline.get_completed_steps()
            concepts_learned = []
            for step in completed_steps:
                concepts_learned.extend(step.learned_concepts)

            # Save output if specified
            if output_file:
                output_path = Path(output_file)
                try:
                    if output_path.suffix == '.csv':
                        result_data.to_csv(output_path, index=False)
                    elif output_path.suffix == '.json':
                        result_data.to_json(output_path, orient='records', indent=2)
                    elif output_path.suffix == '.parquet':
                        result_data.to_parquet(output_path, index=False)
                    else:
                        return CLIResult(
                            success=False,
                            output="",
                            exit_code=1,
                            error_message=f"Unsupported output format: {output_path.suffix}"
                        )
                except Exception as e:
                    return CLIResult(
                        success=False,
                        output="",
                        exit_code=1,
                        error_message=f"Failed to save output: {str(e)}"
                    )

            # Build success output
            output_lines = []
            output_lines.append("=" * 80)
            output_lines.append("Pipeline Execution Summary")
            output_lines.append("=" * 80)
            output_lines.append(f"Input: {input_file}")
            output_lines.append(f"Output: {output_file if output_file else 'Not saved'}")
            output_lines.append("")
            output_lines.append(f"Operations Executed: {len(completed_steps)}")
            for step in completed_steps:
                output_lines.append(f"  ✓ {step.name} ({step.execution_time}ms)")
            output_lines.append("")
            output_lines.append(f"Input Rows: {len(data)}")
            output_lines.append(f"Output Rows: {len(result_data)}")
            output_lines.append(f"Input Columns: {len(data.columns)}")
            output_lines.append(f"Output Columns: {len(result_data.columns)}")
            output_lines.append("")

            if concepts_learned:
                output_lines.append("Concepts Covered:")
                for concept in set(concepts_learned):
                    output_lines.append(f"  • {concept}")
                output_lines.append("")

            output_lines.append("Pipeline execution completed successfully!")
            output_lines.append("=" * 80)

            return CLIResult(
                success=True,
                output="\n".join(output_lines),
                exit_code=0
            )

        except Exception as e:
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message=f"Pipeline execution failed: {str(e)}"
            )

    except Exception as e:
        return CLIResult(
            success=False,
            output="",
            exit_code=1,
            error_message=f"Failed to execute pipeline: {str(e)}"
        )
