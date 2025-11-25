"""EducationalService for DataDojo framework.

Provides educational guidance and progress tracking for learners.
"""

from typing import List, Optional, Dict
from pathlib import Path
import json
from datetime import datetime

from ..models.educational_content import EducationalContent, DifficultyLevel
from ..models.progress_tracker import ProgressTracker
from ..models.processing_step import ProcessingStep


class EducationalService:
    """Service for educational guidance and progress tracking.

    Manages concept explanations, progress tracking, and personalized
    learning assistance.
    """

    def __init__(self, content_path: Optional[str] = None, progress_path: Optional[str] = None):
        """Initialize the EducationalService.

        Args:
            content_path: Path to educational content storage
            progress_path: Path to progress tracking storage
        """
        if content_path is None:
            content_path = str(Path.home() / ".datadojo" / "content")
        if progress_path is None:
            progress_path = str(Path.home() / ".datadojo" / "progress")

        self.content_path = Path(content_path)
        self.progress_path = Path(progress_path)
        self.content_path.mkdir(parents=True, exist_ok=True)
        self.progress_path.mkdir(parents=True, exist_ok=True)

        self._content_cache: Dict[str, EducationalContent] = {}
        self._progress_cache: Dict[str, ProgressTracker] = {}
        self._load_content()

    def _load_content(self) -> None:
        """Load educational content from storage."""
        from ..educational.concepts import get_concept_database
        concept_db = get_concept_database()
        for concept in concept_db.list_concepts():
            self._content_cache[concept.concept_id] = concept


    def _save_content(self, content: EducationalContent) -> None:
        """Save educational content to storage.

        Args:
            content: EducationalContent to save
        """
        content_file = self.content_path / f"{content.concept_id}.json"
        with open(content_file, 'w') as f:
            json.dump(content.to_dict(), f, indent=2)

    def _save_progress(self, progress: ProgressTracker) -> None:
        """Save progress tracker to storage.

        Args:
            progress: ProgressTracker to save
        """
        progress_file = self.progress_path / f"{progress.student_id}_{progress.project_id}.json"
        with open(progress_file, 'w') as f:
            json.dump(progress.to_dict(), f, indent=2)

    def _load_progress(self, student_id: str, project_id: str) -> Optional[ProgressTracker]:
        """Load progress tracker from storage.

        Args:
            student_id: Student identifier
            project_id: Project identifier

        Returns:
            ProgressTracker if found, None otherwise
        """
        progress_file = self.progress_path / f"{student_id}_{project_id}.json"
        if not progress_file.exists():
            return None

        try:
            with open(progress_file, 'r') as f:
                data = json.load(f)
                return ProgressTracker.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def get_concept_explanation(
        self,
        concept_id: str,
        student_level: Optional[DifficultyLevel] = None
    ) -> Optional[EducationalContent]:
        """Get explanation for a concept.

        Args:
            concept_id: Concept identifier
            student_level: Optional student level for appropriateness check

        Returns:
            EducationalContent if found and appropriate, None otherwise
        """
        content = self._content_cache.get(concept_id)

        if content and student_level:
            if not content.is_appropriate_for_level(student_level):
                return None

        return content

    def add_concept(self, content: EducationalContent) -> None:
        """Add educational content for a concept.

        Args:
            content: EducationalContent to add
        """
        self._content_cache[content.concept_id] = content
        self._save_content(content)

    def get_related_concepts(self, concept_id: str) -> List[EducationalContent]:
        """Get related concepts for a given concept.

        Args:
            concept_id: Concept identifier

        Returns:
            List of related EducationalContent
        """
        content = self._content_cache.get(concept_id)
        if not content:
            return []

        related = []
        for related_id in content.related_concepts:
            related_content = self._content_cache.get(related_id)
            if related_content:
                related.append(related_content)

        return related

    def track_progress(
        self,
        student_id: str,
        project_id: str,
        step_id: Optional[str] = None
    ) -> ProgressTracker:
        """Get or create progress tracker for a student and project.

        Args:
            student_id: Student identifier
            project_id: Project identifier
            step_id: Optional current step ID

        Returns:
            ProgressTracker for the student/project
        """
        cache_key = f"{student_id}_{project_id}"

        # Check cache first
        if cache_key in self._progress_cache:
            tracker = self._progress_cache[cache_key]
        else:
            # Try to load from storage
            tracker = self._load_progress(student_id, project_id)

            if tracker is None:
                # Create new tracker
                tracker = ProgressTracker(
                    student_id=student_id,
                    project_id=project_id,
                    started_at=datetime.now(),
                    last_activity=datetime.now()
                )

            self._progress_cache[cache_key] = tracker

        # Update current step if provided
        if step_id is not None:
            tracker.set_current_step(step_id)

        self._save_progress(tracker)
        return tracker

    def record_step_completion(
        self,
        student_id: str,
        project_id: str,
        step: ProcessingStep
    ) -> ProgressTracker:
        """Record completion of a processing step.

        Args:
            student_id: Student identifier
            project_id: Project identifier
            step: Completed ProcessingStep

        Returns:
            Updated ProgressTracker
        """
        tracker = self.track_progress(student_id, project_id)
        tracker.complete_step(step.id)

        # Add learned concepts
        for concept in step.learned_concepts:
            tracker.add_learned_concept(concept)

        self._save_progress(tracker)
        return tracker

    def assess_skill(
        self,
        student_id: str,
        project_id: str,
        skill: str,
        score: float
    ) -> ProgressTracker:
        """Record a skill assessment.

        Args:
            student_id: Student identifier
            project_id: Project identifier
            skill: Skill identifier
            score: Assessment score (0.0 to 100.0)

        Returns:
            Updated ProgressTracker
        """
        tracker = self.track_progress(student_id, project_id)
        tracker.update_skill_assessment(skill, score)
        self._save_progress(tracker)
        return tracker

    def get_progress_summary(self, student_id: str, project_id: str) -> Dict:
        """Get a summary of student progress.

        Args:
            student_id: Student identifier
            project_id: Project identifier

        Returns:
            Dictionary with progress summary
        """
        tracker = self.track_progress(student_id, project_id)

        return {
            "student_id": student_id,
            "project_id": project_id,
            "started_at": tracker.started_at.isoformat(),
            "last_activity": tracker.last_activity.isoformat(),
            "completed_steps_count": len(tracker.completed_steps),
            "learned_concepts_count": len(tracker.learned_concepts),
            "average_skill_score": tracker.get_average_skill_score(),
            "current_step": tracker.current_step
        }

    def suggest_next_concepts(
        self,
        student_id: str,
        project_id: str,
        limit: int = 5
    ) -> List[EducationalContent]:
        """Suggest next concepts to learn based on progress.

        Args:
            student_id: Student identifier
            project_id: Project identifier
            limit: Maximum number of suggestions

        Returns:
            List of suggested EducationalContent
        """
        tracker = self.track_progress(student_id, project_id)

        # Find concepts that are related to learned concepts but not yet learned
        suggestions = []
        for learned_concept in tracker.learned_concepts:
            related = self.get_related_concepts(learned_concept)
            for content in related:
                if (content.concept_id not in tracker.learned_concepts and
                    content not in suggestions):
                    suggestions.append(content)

                if len(suggestions) >= limit:
                    return suggestions

        return suggestions

    def provide_help(
        self,
        concept_id: str,
        student_level: DifficultyLevel,
        include_examples: bool = True,
        include_analogies: bool = True
    ) -> Optional[Dict]:
        """Provide help for a concept.

        Args:
            concept_id: Concept identifier
            student_level: Student's difficulty level
            include_examples: Whether to include examples
            include_analogies: Whether to include analogies

        Returns:
            Dictionary with help content, None if concept not found
        """
        content = self.get_concept_explanation(concept_id, student_level)
        if not content:
            return None

        help_data = {
            "concept_id": content.concept_id,
            "title": content.title,
            "explanation": content.explanation,
            "difficulty_level": content.difficulty_level.value
        }

        if include_examples and content.examples:
            help_data["examples"] = content.examples

        if include_analogies and content.analogies:
            help_data["analogies"] = content.analogies

        # Add related concepts
        related = self.get_related_concepts(concept_id)
        help_data["related_concepts"] = [
            {"id": r.concept_id, "title": r.title}
            for r in related
        ]

        return help_data
