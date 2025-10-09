"""CLI command for showing student progress."""

import json
import sys
import os
from typing import Optional
from datetime import datetime

# Add contracts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'specs/001-use-the-requirements/contracts'))

from cli_interface import CLIResult


def show_progress(
    dojo,
    student_id: str,
    project_id: Optional[str] = None,
    format_output: str = "summary"
) -> CLIResult:
    """Show learning progress for a student.

    Args:
        dojo: Dojo instance
        student_id: Learner identifier
        project_id: Specific project (optional, shows all if not specified)
        format_output: Output format (summary|detailed|json)

    Returns:
        CLI result with progress information
    """
    try:
        if not student_id:
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message="Student ID is required"
            )

        if project_id:
            # Show progress for specific project
            try:
                project = dojo.load_project(project_id)
                progress = project.get_progress(student_id)

                if format_output == "json":
                    output = json.dumps(progress, indent=2)
                else:
                    output = _format_single_project_progress(project.info, progress, format_output)

                return CLIResult(
                    success=True,
                    output=output,
                    exit_code=0
                )
            except ValueError as e:
                return CLIResult(
                    success=False,
                    output="",
                    exit_code=1,
                    error_message=str(e)
                )
        else:
            # Show progress for all projects
            all_projects = dojo.list_projects()
            progress_data = []

            for project_info in all_projects:
                try:
                    project = dojo.load_project(project_info.id)
                    progress = project.get_progress(student_id)
                    progress_data.append({
                        "project_info": project_info,
                        "progress": progress
                    })
                except:
                    # Skip projects with no progress
                    continue

            if not progress_data:
                output = f"No progress found for student: {student_id}"
                return CLIResult(
                    success=True,
                    output=output,
                    exit_code=0
                )

            if format_output == "json":
                output = _format_all_progress_json(student_id, progress_data)
            else:
                output = _format_all_progress(student_id, progress_data, format_output)

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
            error_message=f"Failed to retrieve progress: {str(e)}"
        )


def _format_single_project_progress(project_info, progress, detail_level: str) -> str:
    """Format progress for a single project."""
    lines = []

    lines.append("=" * 80)
    lines.append(f"Progress Report: {project_info.name}")
    lines.append("=" * 80)
    lines.append(f"Student: {progress['student_id']}")
    lines.append(f"Project: {project_info.id}")
    lines.append(f"Domain: {project_info.domain.value}")
    lines.append(f"Difficulty: {project_info.difficulty.value}")
    lines.append("")

    # Parse timestamps
    started = datetime.fromisoformat(progress['started_at'])
    last = datetime.fromisoformat(progress['last_activity'])

    lines.append(f"Started: {started.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Last Activity: {last.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Progress metrics
    lines.append("Progress Metrics:")
    lines.append(f"  Completed Steps: {progress['completed_steps_count']}")
    lines.append(f"  Learned Concepts: {len(progress['learned_concepts'])}")
    lines.append(f"  Average Skill Score: {progress['average_skill_score']:.1f}%")
    lines.append("")

    if detail_level == "detailed":
        # Show completed steps
        if progress['completed_steps']:
            lines.append("Completed Steps:")
            for step_id in progress['completed_steps']:
                lines.append(f"  ✓ {step_id}")
            lines.append("")

        # Show learned concepts
        if progress['learned_concepts']:
            lines.append("Learned Concepts:")
            for concept in progress['learned_concepts']:
                lines.append(f"  • {concept}")
            lines.append("")

        # Show skill assessments
        if progress['skill_assessments']:
            lines.append("Skill Assessments:")
            for skill, score in progress['skill_assessments'].items():
                lines.append(f"  {skill}: {score:.1f}%")
            lines.append("")

    # Current status
    if progress['current_step']:
        lines.append(f"Current Step: {progress['current_step']}")
    else:
        lines.append("Status: Not currently active")

    lines.append("=" * 80)

    return "\n".join(lines)


def _format_all_progress(student_id: str, progress_data, detail_level: str) -> str:
    """Format progress for all projects."""
    lines = []

    lines.append("=" * 80)
    lines.append(f"Progress Overview for Student: {student_id}")
    lines.append("=" * 80)
    lines.append(f"Total Projects: {len(progress_data)}")
    lines.append("")

    for data in progress_data:
        project_info = data['project_info']
        progress = data['progress']

        lines.append(f"Project: {project_info.name} ({project_info.id})")
        lines.append(f"  Domain: {project_info.domain.value} | Difficulty: {project_info.difficulty.value}")
        lines.append(f"  Completed Steps: {progress['completed_steps_count']}")
        lines.append(f"  Learned Concepts: {len(progress['learned_concepts'])}")
        lines.append(f"  Average Score: {progress['average_skill_score']:.1f}%")

        if progress['current_step']:
            lines.append(f"  Status: In Progress (Current: {progress['current_step']})")
        else:
            lines.append(f"  Status: Not Active")

        lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def _format_all_progress_json(student_id: str, progress_data) -> str:
    """Format all progress as JSON."""
    result = {
        "student_id": student_id,
        "total_projects": len(progress_data),
        "projects": []
    }

    for data in progress_data:
        project_info = data['project_info']
        progress = data['progress']

        result['projects'].append({
            "project_id": project_info.id,
            "project_name": project_info.name,
            "domain": project_info.domain.value,
            "difficulty": project_info.difficulty.value,
            "progress": progress
        })

    return json.dumps(result, indent=2)
