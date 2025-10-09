"""CLI command for listing available projects."""

import json
import sys
import os
from typing import Optional

# Add contracts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'specs/001-use-the-requirements/contracts'))

from cli_interface import CLIResult
from dojo_api import Domain, DifficultyLevel


def list_projects(
    dojo,
    domain: Optional[str] = None,
    difficulty: Optional[str] = None,
    format_output: str = "table"
) -> CLIResult:
    """List available learning projects.

    Args:
        dojo: Dojo instance
        domain: Filter by subject area
        difficulty: Filter by skill level
        format_output: Output format (table|json|csv)

    Returns:
        CLI result with formatted project list
    """
    try:
        # Convert string arguments to enums
        domain_filter = None
        if domain:
            domain_map = {
                "ecommerce": Domain.ECOMMERCE,
                "healthcare": Domain.HEALTHCARE,
                "finance": Domain.FINANCE
            }
            domain_filter = domain_map.get(domain.lower())
            if not domain_filter:
                return CLIResult(
                    success=False,
                    output="",
                    exit_code=1,
                    error_message=f"Invalid domain: {domain}. Must be one of: ecommerce, healthcare, finance"
                )

        difficulty_filter = None
        if difficulty:
            difficulty_map = {
                "beginner": DifficultyLevel.BEGINNER,
                "intermediate": DifficultyLevel.INTERMEDIATE,
                "advanced": DifficultyLevel.ADVANCED
            }
            difficulty_filter = difficulty_map.get(difficulty.lower())
            if not difficulty_filter:
                return CLIResult(
                    success=False,
                    output="",
                    exit_code=1,
                    error_message=f"Invalid difficulty: {difficulty}. Must be one of: beginner, intermediate, advanced"
                )

        # Get projects from dojo
        projects = dojo.list_projects(domain=domain_filter, difficulty=difficulty_filter)

        if not projects:
            output = "No projects found matching criteria."
            return CLIResult(
                success=True,
                output=output,
                exit_code=0
            )

        # Format output
        if format_output == "json":
            output = _format_json(projects)
        elif format_output == "csv":
            output = _format_csv(projects)
        else:  # table
            output = _format_table(projects)

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
            error_message=f"Failed to list projects: {str(e)}"
        )


def _format_table(projects) -> str:
    """Format projects as a table."""
    if not projects:
        return "No projects available."

    lines = []

    # Header
    lines.append("=" * 100)
    lines.append(f"{'ID':<20} {'Name':<25} {'Domain':<12} {'Difficulty':<12} {'Time (min)':<10}")
    lines.append("=" * 100)

    # Project rows
    for project in projects:
        lines.append(
            f"{project.id:<20} {project.name:<25} "
            f"{project.domain.value:<12} {project.difficulty.value:<12} "
            f"{project.estimated_time_minutes:<10}"
        )

    lines.append("=" * 100)
    lines.append(f"\nTotal projects: {len(projects)}")

    return "\n".join(lines)


def _format_json(projects) -> str:
    """Format projects as JSON."""
    projects_data = [
        {
            "id": p.id,
            "name": p.name,
            "domain": p.domain.value,
            "difficulty": p.difficulty.value,
            "description": p.description,
            "estimated_time_minutes": p.estimated_time_minutes,
            "learning_objectives": p.learning_objectives
        }
        for p in projects
    ]
    return json.dumps(projects_data, indent=2)


def _format_csv(projects) -> str:
    """Format projects as CSV."""
    lines = []

    # Header
    lines.append("ID,Name,Domain,Difficulty,EstimatedTimeMinutes,Description")

    # Project rows
    for project in projects:
        # Escape commas and quotes in description
        desc = project.description.replace('"', '""')
        lines.append(
            f'{project.id},{project.name},{project.domain.value},'
            f'{project.difficulty.value},{project.estimated_time_minutes},"{desc}"'
        )

    return "\n".join(lines)
