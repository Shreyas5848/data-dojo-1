"""ProjectInterface implementation for DataDojo framework.

Interface for working with specific learning projects.
"""

from typing import Dict, Any, Optional
import sys
import os
import pandas as pd
from pathlib import Path

# Add contracts to path for interface imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'specs/001-use-the-requirements/contracts'))

from dojo_api import ProjectInterface, ProjectInfo, GuidanceLevel, DifficultyLevel, Domain
from ..models.learning_project import LearningProject, Domain as ModelDomain, Difficulty as ModelDifficulty
from ..services.educational_service import EducationalService
from .pipeline import PipelineImpl


class Project(ProjectInterface):
    """Interface for working with a specific learning project.

    Provides access to project data, pipeline creation, and progress tracking.
    """

    def __init__(
        self,
        project: LearningProject,
        educational_service: EducationalService,
        educational_mode: bool = True
    ):
        """Initialize Project.

        Args:
            project: LearningProject model instance
            educational_service: Service for educational features
            educational_mode: Whether educational features are enabled
        """
        self._project = project
        self._educational_service = educational_service
        self._educational_mode = educational_mode
        self._dataset_cache: Optional[pd.DataFrame] = None

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

    @property
    def info(self) -> ProjectInfo:
        """Get project information.

        Returns:
            ProjectInfo with project details
        """
        return ProjectInfo(
            id=self._project.id,
            name=self._project.name,
            domain=self._convert_domain_from_model(self._project.domain),
            difficulty=self._convert_difficulty_from_model(self._project.difficulty),
            description=self._project.description,
            estimated_time_minutes=60,  # Default estimate
            learning_objectives=self._project.expected_outcomes
        )

    @property
    def dataset(self) -> Any:
        """Get the raw dataset for this project.

        Returns:
            DataFrame with raw project data

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset cannot be loaded
        """
        if self._dataset_cache is not None:
            return self._dataset_cache

        # Load dataset from file
        dataset_path = Path(self._project.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self._project.dataset_path}")

        try:
            # Determine file type and load accordingly
            if dataset_path.suffix == '.csv':
                self._dataset_cache = pd.read_csv(dataset_path)
            elif dataset_path.suffix == '.json':
                self._dataset_cache = pd.read_json(dataset_path)
            elif dataset_path.suffix == '.parquet':
                self._dataset_cache = pd.read_parquet(dataset_path)
            elif dataset_path.suffix in ['.xlsx', '.xls']:
                self._dataset_cache = pd.read_excel(dataset_path)
            else:
                raise ValueError(f"Unsupported file format: {dataset_path.suffix}")

            return self._dataset_cache

        except Exception as e:
            raise ValueError(f"Failed to load dataset: {str(e)}") from e

    def create_pipeline(self, guidance_level: GuidanceLevel = GuidanceLevel.DETAILED) -> 'PipelineImpl':
        """Create a new preprocessing pipeline.

        Args:
            guidance_level: Amount of educational assistance

        Returns:
            Pipeline interface for chaining operations
        """
        from ..models.pipeline import Pipeline, GuidanceLevel as ModelGuidanceLevel

        # Convert contract GuidanceLevel to model GuidanceLevel
        guidance_map = {
            GuidanceLevel.NONE: ModelGuidanceLevel.NONE,
            GuidanceLevel.BASIC: ModelGuidanceLevel.BASIC,
            GuidanceLevel.DETAILED: ModelGuidanceLevel.DETAILED
        }
        model_guidance = guidance_map.get(guidance_level, ModelGuidanceLevel.BASIC)

        # Create pipeline model
        pipeline = Pipeline(
            id=f"{self._project.id}_pipeline_{id(self)}",
            name=f"{self._project.name} Pipeline",
            educational_mode=self._educational_mode,
            guidance_level=model_guidance,
            project_id=self._project.id
        )

        # Create and return PipelineImpl
        return PipelineImpl(
            pipeline=pipeline,
            educational_service=self._educational_service
        )

    def get_progress(self, student_id: str) -> Dict[str, Any]:
        """Get learning progress for a student.

        Args:
            student_id: Unique learner identifier

        Returns:
            Progress information including completed steps and concepts
        """
        if not student_id:
            raise ValueError("Student ID cannot be empty")

        # Get progress tracker from educational service
        tracker = self._educational_service.track_progress(
            student_id=student_id,
            project_id=self._project.id
        )

        # Return progress as dictionary
        return {
            "student_id": tracker.student_id,
            "project_id": tracker.project_id,
            "started_at": tracker.started_at.isoformat(),
            "last_activity": tracker.last_activity.isoformat(),
            "completed_steps": tracker.completed_steps,
            "learned_concepts": tracker.learned_concepts,
            "skill_assessments": tracker.skill_assessments,
            "current_step": tracker.current_step,
            "average_skill_score": tracker.get_average_skill_score()
        }
