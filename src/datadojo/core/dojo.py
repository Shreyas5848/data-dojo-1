"""DojoInterface implementation for DataDojo framework.

Main entry point for the DataDojo learning framework.
"""

from typing import List, Optional
import sys
import os

# Add contracts to path for interface imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'specs/001-use-the-requirements/contracts'))

from dojo_api import DojoInterface, ProjectInfo, Domain, DifficultyLevel
from ..services.project_service import ProjectService
from ..services.educational_service import EducationalService
from ..services.domain_service import DomainService
from ..models.learning_project import Domain as ModelDomain, Difficulty as ModelDifficulty
from .project import Project


class Dojo(DojoInterface):
    """Main entry point for DataDojo learning framework.

    Provides access to learning projects across multiple domains and
    difficulty levels with optional educational guidance.
    """

    def __init__(self, educational_mode: bool = True):
        """Initialize DataDojo learning environment.

        Args:
            educational_mode: Enable step-by-step guidance and explanations
        """
        self.educational_mode = educational_mode
        self.project_service = ProjectService()
        self.educational_service = EducationalService()
        self.domain_service = DomainService()

        # Initialize default domains if needed
        self.domain_service.initialize_default_domains()

    def _convert_domain_to_model(self, domain: Optional[Domain]) -> Optional[ModelDomain]:
        """Convert contract Domain to model Domain.

        Args:
            domain: Contract Domain enum

        Returns:
            Model Domain enum or None
        """
        if domain is None:
            return None

        domain_map = {
            Domain.ECOMMERCE: ModelDomain.ECOMMERCE,
            Domain.HEALTHCARE: ModelDomain.HEALTHCARE,
            Domain.FINANCE: ModelDomain.FINANCE
        }
        return domain_map.get(domain)

    def _convert_difficulty_to_model(self, difficulty: Optional[DifficultyLevel]) -> Optional[ModelDifficulty]:
        """Convert contract DifficultyLevel to model Difficulty.

        Args:
            difficulty: Contract DifficultyLevel enum

        Returns:
            Model Difficulty enum or None
        """
        if difficulty is None:
            return None

        difficulty_map = {
            DifficultyLevel.BEGINNER: ModelDifficulty.BEGINNER,
            DifficultyLevel.INTERMEDIATE: ModelDifficulty.INTERMEDIATE,
            DifficultyLevel.ADVANCED: ModelDifficulty.ADVANCED
        }
        return difficulty_map.get(difficulty)

    def _convert_domain_from_model(self, domain: ModelDomain) -> Domain:
        """Convert model Domain to contract Domain.

        Args:
            domain: Model Domain enum

        Returns:
            Contract Domain enum
        """
        domain_map = {
            ModelDomain.ECOMMERCE: Domain.ECOMMERCE,
            ModelDomain.HEALTHCARE: Domain.HEALTHCARE,
            ModelDomain.FINANCE: Domain.FINANCE
        }
        return domain_map[domain]

    def _convert_difficulty_from_model(self, difficulty: ModelDifficulty) -> DifficultyLevel:
        """Convert model Difficulty to contract DifficultyLevel.

        Args:
            difficulty: Model Difficulty enum

        Returns:
            Contract DifficultyLevel enum
        """
        difficulty_map = {
            ModelDifficulty.BEGINNER: DifficultyLevel.BEGINNER,
            ModelDifficulty.INTERMEDIATE: DifficultyLevel.INTERMEDIATE,
            ModelDifficulty.ADVANCED: DifficultyLevel.ADVANCED
        }
        return difficulty_map[difficulty]

    def list_projects(
        self,
        domain: Optional[Domain] = None,
        difficulty: Optional[DifficultyLevel] = None
    ) -> List[ProjectInfo]:
        """Get available learning projects.

        Args:
            domain: Filter by subject area (optional)
            difficulty: Filter by skill level (optional)

        Returns:
            List of available projects matching criteria
        """
        # Convert contract enums to model enums
        model_domain = self._convert_domain_to_model(domain)
        model_difficulty = self._convert_difficulty_to_model(difficulty)

        # Get projects from service
        projects = self.project_service.list_projects(
            domain=model_domain,
            difficulty=model_difficulty
        )

        # Convert to ProjectInfo
        project_infos = []
        for project in projects:
            project_info = ProjectInfo(
                id=project.id,
                name=project.name,
                domain=self._convert_domain_from_model(project.domain),
                difficulty=self._convert_difficulty_from_model(project.difficulty),
                description=project.description,
                estimated_time_minutes=60,  # Default, could be stored in project
                learning_objectives=project.expected_outcomes
            )
            project_infos.append(project_info)

        return project_infos

    def load_project(
        self,
        project_id: str,
        difficulty: Optional[DifficultyLevel] = None
    ) -> 'Project':
        """Load a specific learning project.

        Args:
            project_id: Unique project identifier
            difficulty: Override project difficulty level

        Returns:
            Project interface for learning activities

        Raises:
            ValueError: If project_id is not found or invalid
        """
        if not project_id:
            raise ValueError("Project ID cannot be empty")

        # Get project from service
        project = self.project_service.get_project(project_id)
        if not project:
            raise ValueError(f"Project with ID '{project_id}' not found")

        # Override difficulty if specified
        if difficulty is not None:
            model_difficulty = self._convert_difficulty_to_model(difficulty)
            project.difficulty = model_difficulty

        # Create and return Project implementation
        return Project(
            project=project,
            educational_service=self.educational_service,
            educational_mode=self.educational_mode
        )
