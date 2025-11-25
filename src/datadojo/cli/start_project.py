"""CLI command for starting a learning project."""

from typing import Optional

from datadojo.dojo_api import GuidanceLevel

class CLIResult:
    def __init__(self, success: bool, output: str, exit_code: int, error_message: Optional[str] = None):
        self.success = success
        self.output = output
        self.exit_code = exit_code
        self.error_message = error_message


def start_project(
    dojo,
    project_id: str,
    student_id: str,
    guidance_level: str = "detailed",
    interactive: bool = True
) -> CLIResult:
    """Start a learning project.

    Args:
        dojo: Dojo instance
        project_id: Unique project identifier
        student_id: Learner identifier for progress tracking
        guidance_level: Amount of educational assistance
        interactive: Enable interactive mode

    Returns:
        CLI result with project startup information
    """
    try:
        if not project_id:
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message="Project ID is required"
            )

        if not student_id:
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message="Student ID is required"
            )

        # Convert guidance level string to enum
        guidance_map = {
            "none": GuidanceLevel.NONE,
            "basic": GuidanceLevel.BASIC,
            "detailed": GuidanceLevel.DETAILED
        }
        guidance_enum = guidance_map.get(guidance_level.lower())
        if not guidance_enum:
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message=f"Invalid guidance level: {guidance_level}. Must be one of: none, basic, detailed"
            )

        # Load project
        project = dojo.load_project(project_id)

        # Get project info
        info = project.info

        # Track progress
        progress = project.get_progress(student_id)

        # Build output
        output_lines = []
        output_lines.append("=" * 80)
        output_lines.append(f"Starting Project: {info.name}")
        output_lines.append("=" * 80)
        output_lines.append(f"ID: {info.id}")
        output_lines.append(f"Domain: {info.domain.value}")
        output_lines.append(f"Difficulty: {info.difficulty.value}")
        output_lines.append(f"Estimated Time: {info.estimated_time_minutes} minutes")
        output_lines.append("")
        output_lines.append("Description:")
        output_lines.append(f"  {info.description}")
        output_lines.append("")
        output_lines.append("Learning Objectives:")
        for i, objective in enumerate(info.learning_objectives, 1):
            output_lines.append(f"  {i}. {objective}")
        output_lines.append("")
        output_lines.append("=" * 80)
        output_lines.append(f"Student: {student_id}")
        output_lines.append(f"Guidance Level: {guidance_level}")
        output_lines.append(f"Interactive Mode: {'Enabled' if interactive else 'Disabled'}")
        output_lines.append("=" * 80)
        output_lines.append("")

        # Show progress if any
        if progress.get("completed_steps"):
            output_lines.append("Previous Progress:")
            output_lines.append(f"  Completed Steps: {len(progress['completed_steps'])}")
            output_lines.append(f"  Learned Concepts: {len(progress['learned_concepts'])}")
            output_lines.append("")

        # Create pipeline for demonstration
        pipeline = project.create_pipeline(guidance_level=guidance_enum)

        output_lines.append("Next Steps:")
        output_lines.append("  1. Load your dataset using: project.dataset")
        output_lines.append("  2. Create a pipeline: pipeline = project.create_pipeline()")
        output_lines.append("  3. Add preprocessing steps: pipeline.add_step('data_cleaning', ...)")
        output_lines.append("  4. Execute pipeline: result = pipeline.execute(data)")
        output_lines.append("")
        output_lines.append("Available operations:")
        for op in pipeline.get_available_operations():
            output_lines.append(f"  - {op}")
        output_lines.append("")
        output_lines.append("Project started successfully! Ready for learning.")
        output_lines.append("=" * 80)

        return CLIResult(
            success=True,
            output="\n".join(output_lines),
            exit_code=0
        )

    except ValueError as e:
        return CLIResult(
            success=False,
            output="",
            exit_code=1,
            error_message=str(e)
        )
    except Exception as e:
        return CLIResult(
            success=False,
            output="",
            exit_code=1,
            error_message=f"Failed to start project: {str(e)}"
        )
