"""CLI command for listing available projects."""

import json
from typing import Optional

from datadojo.dojo_api import Domain, DifficultyLevel

class CLIResult:
    def __init__(self, success: bool, output: str, exit_code: int, error_message: Optional[str] = None):
        self.success = success
        self.output = output
        self.exit_code = exit_code
        self.error_message = error_message



def list_projects(
    dojo,
    domain: Optional[str] = None,
    difficulty: Optional[str] = None
) -> CLIResult:
    """Gets a list of available learning projects."""
    try:
        domain_filter = Domain(domain) if domain else None
        difficulty_filter = DifficultyLevel(difficulty) if difficulty else None
        
        projects = dojo.list_projects(domain=domain_filter, difficulty=difficulty_filter)
        
        if not projects:
            return CLIResult(success=True, output=[], exit_code=0, error_message="No projects found matching criteria.")
        
        # Return the raw list of projects
        return CLIResult(success=True, output=projects, exit_code=0)

    except ValueError as e: # Handles invalid domain/difficulty strings
        return CLIResult(success=False, output=None, exit_code=1, error_message=str(e))
    except Exception as e:
        return CLIResult(success=False, output=None, exit_code=1, error_message=f"Failed to list projects: {str(e)}")



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
