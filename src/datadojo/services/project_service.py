"""ProjectService for DataDojo framework.

Provides CRUD operations for learning projects with file-based storage.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import json

from ..models.learning_project import LearningProject, Domain, Difficulty


class ProjectService:
    """Service for managing learning projects.

    Provides CRUD operations for LearningProject entities with file-based
    persistence and filtering capabilities.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the ProjectService.

        Args:
            storage_path: Path to store project data (defaults to ~/.datadojo/projects)
        """
        if storage_path is None:
            storage_path = str(Path.home() / ".datadojo" / "projects")

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._projects_cache: Dict[str, LearningProject] = {}
        self._load_projects()

    def _load_projects(self) -> None:
        """Load all projects from storage into cache."""
        self._projects_cache.clear()

        if not self.storage_path.exists():
            return

        for project_file in self.storage_path.glob("*.json"):
            try:
                with open(project_file, 'r') as f:
                    data = json.load(f)
                    project = LearningProject.from_dict(data)
                    self._projects_cache[project.id] = project
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Log error but continue loading other projects
                print(f"Warning: Failed to load project from {project_file}: {e}")

    def _save_project(self, project: LearningProject) -> None:
        """Save a project to storage.

        Args:
            project: LearningProject to save
        """
        project_file = self.storage_path / f"{project.id}.json"
        with open(project_file, 'w') as f:
            json.dump(project.to_dict(), f, indent=2)

    def _delete_project_file(self, project_id: str) -> None:
        """Delete a project file from storage.

        Args:
            project_id: ID of the project to delete
        """
        project_file = self.storage_path / f"{project_id}.json"
        if project_file.exists():
            project_file.unlink()

    def create_project(self, project: LearningProject) -> LearningProject:
        """Create a new learning project.

        Args:
            project: LearningProject to create

        Returns:
            Created LearningProject

        Raises:
            ValueError: If project with same ID already exists
        """
        if project.id in self._projects_cache:
            raise ValueError(f"Project with ID '{project.id}' already exists")

        self._projects_cache[project.id] = project
        self._save_project(project)
        return project

    def get_project(self, project_id: str) -> Optional[LearningProject]:
        """Get a project by ID.

        Args:
            project_id: ID of the project to retrieve

        Returns:
            LearningProject if found, None otherwise
        """
        return self._projects_cache.get(project_id)

    def update_project(self, project: LearningProject) -> LearningProject:
        """Update an existing project.

        Args:
            project: LearningProject with updated data

        Returns:
            Updated LearningProject

        Raises:
            ValueError: If project does not exist
        """
        if project.id not in self._projects_cache:
            raise ValueError(f"Project with ID '{project.id}' does not exist")

        self._projects_cache[project.id] = project
        self._save_project(project)
        return project

    def delete_project(self, project_id: str) -> bool:
        """Delete a project by ID.

        Args:
            project_id: ID of the project to delete

        Returns:
            True if project was deleted, False if not found
        """
        if project_id not in self._projects_cache:
            return False

        del self._projects_cache[project_id]
        self._delete_project_file(project_id)
        return True

    def list_projects(
        self,
        domain: Optional[Domain] = None,
        difficulty: Optional[Difficulty] = None
    ) -> List[LearningProject]:
        """List all projects with optional filtering.

        Args:
            domain: Filter by domain (optional)
            difficulty: Filter by difficulty (optional)

        Returns:
            List of LearningProjects matching filters
        """
        projects = list(self._projects_cache.values())

        if domain is not None:
            projects = [p for p in projects if p.domain == domain]

        if difficulty is not None:
            projects = [p for p in projects if p.difficulty == difficulty]

        return projects

    def get_projects_by_domain(self, domain: Domain) -> List[LearningProject]:
        """Get all projects in a specific domain.

        Args:
            domain: Domain to filter by

        Returns:
            List of LearningProjects in the domain
        """
        return self.list_projects(domain=domain)

    def get_projects_by_difficulty(self, difficulty: Difficulty) -> List[LearningProject]:
        """Get all projects at a specific difficulty level.

        Args:
            difficulty: Difficulty level to filter by

        Returns:
            List of LearningProjects at the difficulty level
        """
        return self.list_projects(difficulty=difficulty)

    def project_exists(self, project_id: str) -> bool:
        """Check if a project exists.

        Args:
            project_id: ID of the project to check

        Returns:
            True if project exists, False otherwise
        """
        return project_id in self._projects_cache

    def get_project_count(self) -> int:
        """Get the total number of projects.

        Returns:
            Number of projects in storage
        """
        return len(self._projects_cache)

    def clear_all_projects(self) -> None:
        """Delete all projects from storage.

        WARNING: This operation cannot be undone.
        """
        for project_id in list(self._projects_cache.keys()):
            self.delete_project(project_id)
