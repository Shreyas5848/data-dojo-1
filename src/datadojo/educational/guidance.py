"""Interactive guidance system for DataDojo.

Provides context-aware, adaptive guidance for learners based on their
progress, difficulty level, and current challenges.
"""

from typing import Dict, List, Optional, Any
from ..models.progress_tracker import ProgressTracker
from ..models.processing_step import ProcessingStep
from ..models.educational_content import DifficultyLevel
from .concepts import get_concept_database


class GuidanceSystem:
    """Provides interactive, adaptive guidance for learners."""

    def __init__(self):
        """Initialize the guidance system."""
        self.concept_db = get_concept_database()

    def get_step_guidance(
        self,
        step: ProcessingStep,
        progress: Optional[ProgressTracker] = None,
        data_issues: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get guidance for a specific processing step.

        Args:
            step: ProcessingStep to provide guidance for
            progress: Optional student progress tracker
            data_issues: Optional list of detected data issues

        Returns:
            Dictionary with guidance information
        """
        guidance = {
            "step_name": step.name,
            "operation_type": step.operation_type.value,
            "hints": [],
            "warnings": [],
            "tips": [],
            "related_concepts": []
        }

        # Add operation-specific hints
        guidance["hints"].extend(
            self._get_operation_hints(step.operation_type.value)
        )

        # Add data issue-specific warnings
        if data_issues:
            guidance["warnings"].extend(
                self._get_issue_warnings(data_issues)
            )

        # Add difficulty-appropriate tips
        if progress:
            avg_score = progress.get_average_skill_score()
            if avg_score < 50:
                guidance["tips"].append(
                    "Take your time and review the concepts before proceeding"
                )
            elif avg_score > 80:
                guidance["tips"].append(
                    "You're doing great! Consider trying the advanced version"
                )

        # Add related concepts
        for concept_id in step.learned_concepts:
            concept = self.concept_db.get_concept(concept_id)
            if concept:
                guidance["related_concepts"].append({
                    "id": concept.concept_id,
                    "title": concept.title,
                    "summary": concept.get_summary()
                })

        return guidance

    def _get_operation_hints(self, operation_type: str) -> List[str]:
        """Get hints for a specific operation type."""
        hints_map = {
            "data_cleaning": [
                "Start by exploring the data with df.info() and df.describe()",
                "Look for missing values, duplicates, and inconsistencies",
                "Document your cleaning decisions for reproducibility"
            ],
            "feature_engineering": [
                "Think about what features would be meaningful for your problem",
                "Consider domain knowledge when creating features",
                "Test new features' correlation with your target variable"
            ],
            "transformation": [
                "Check data distributions before choosing transformations",
                "Remember to apply the same transformations to test data",
                "Consider the scale and range of your features"
            ],
            "validation": [
                "Define clear validation rules based on domain knowledge",
                "Test edge cases and boundary conditions",
                "Log validation failures for investigation"
            ]
        }
        return hints_map.get(operation_type, [])

    def _get_issue_warnings(self, issues: List[str]) -> List[str]:
        """Get warnings based on detected data issues."""
        warnings = []
        warning_map = {
            "missing_values": "High percentage of missing values detected - consider imputation strategies",
            "duplicates": "Duplicate rows found - review before removing to avoid data loss",
            "outliers": "Outliers detected - investigate whether they're errors or valid extreme values",
            "skewed_distribution": "Highly skewed distribution - consider log transformation",
            "high_cardinality": "High cardinality categorical variable - consider grouping or encoding strategies"
        }

        for issue in issues:
            if issue in warning_map:
                warnings.append(warning_map[issue])

        return warnings

    def provide_hint(
        self,
        context: str,
        difficulty: DifficultyLevel = DifficultyLevel.BEGINNER
    ) -> str:
        """Provide a hint based on context and difficulty.

        Args:
            context: Context or problem description
            difficulty: Student difficulty level

        Returns:
            Appropriate hint message
        """
        # Simple keyword-based hints
        context_lower = context.lower()

        if "missing" in context_lower or "nan" in context_lower:
            if difficulty == DifficultyLevel.BEGINNER:
                return "Use df.fillna() to fill missing values with a specific value like mean or median"
            else:
                return "Consider advanced imputation methods like KNN or iterative imputation"

        elif "duplicate" in context_lower:
            return "Use df.duplicated() to find duplicates and df.drop_duplicates() to remove them"

        elif "type" in context_lower or "convert" in context_lower:
            return "Use df.astype() to convert column types, or pd.to_numeric() for safe conversion"

        elif "scale" in context_lower or "normalize" in context_lower:
            return "Use StandardScaler for standardization or MinMaxScaler for normalization from sklearn.preprocessing"

        else:
            return "Break down the problem into smaller steps and tackle each one individually"

    def suggest_next_steps(
        self,
        progress: ProgressTracker,
        all_steps: List[ProcessingStep]
    ) -> List[Dict[str, str]]:
        """Suggest next steps based on progress.

        Args:
            progress: Student progress tracker
            all_steps: All available processing steps

        Returns:
            List of suggested next steps
        """
        suggestions = []
        completed_ids = set(progress.completed_steps)

        for step in all_steps:
            if step.id not in completed_ids and step.is_ready(list(completed_ids)):
                suggestions.append({
                    "step_id": step.id,
                    "name": step.name,
                    "description": step.description,
                    "reason": "Prerequisites completed, ready to start"
                })

        return suggestions[:3]  # Return top 3 suggestions

    def detect_struggling_areas(
        self,
        progress: ProgressTracker
    ) -> List[Dict[str, Any]]:
        """Detect areas where the student might be struggling.

        Args:
            progress: Student progress tracker

        Returns:
            List of struggling areas with recommendations
        """
        struggling = []

        # Check skill scores
        for skill, score in progress.skill_assessments.items():
            if score < 60:
                struggling.append({
                    "area": skill,
                    "score": score,
                    "recommendation": f"Consider reviewing {skill} concepts and practicing more"
                })

        # Check for concepts not yet learned
        expected_concepts = ["missing_values", "outliers", "data_types"]
        not_learned = [c for c in expected_concepts if c not in progress.learned_concepts]

        if not_learned:
            struggling.append({
                "area": "core_concepts",
                "missing_concepts": not_learned,
                "recommendation": "Review these fundamental concepts before advancing"
            })

        return struggling
