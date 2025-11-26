"""EducationalInterface implementation for DataDojo framework.

Interface for accessing educational content and guidance.
"""

from typing import Dict, Any, List

from datadojo.dojo_api import EducationalInterface
from ..services.educational_service import EducationalService
from ..models.educational_content import DifficultyLevel


class Educational(EducationalInterface):
    """Interface for accessing educational content and guidance.

    Provides concept explanations, step guidance, and progress tracking
    for learners.
    """

    def __init__(self, educational_service: EducationalService):
        """Initialize Educational.

        Args:
            educational_service: Service for educational features
        """
        self._educational_service = educational_service

    def get_concept_explanation(self, concept_id: str) -> Dict[str, Any]:
        """Get detailed explanation for a data preprocessing concept.

        Args:
            concept_id: Unique concept identifier

        Returns:
            Educational content with explanations and examples

        Raises:
            KeyError: If concept_id is not found
        """
        if not concept_id:
            raise KeyError("Concept ID cannot be empty")

        # Get concept from educational service
        content = self._educational_service.get_concept_explanation(concept_id)

        if content is None:
            raise KeyError(f"Concept '{concept_id}' not found")

        # Convert to dictionary format
        return {
            "concept_id": content.concept_id,
            "title": content.title,
            "explanation": content.explanation,
            "difficulty_level": content.difficulty_level.value,
            "analogies": content.analogies,
            "examples": content.examples,
            "related_concepts": content.related_concepts
        }

    def get_step_guidance(self, step_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get context-aware guidance for current step.

        Args:
            step_context: Current state including data issues and operation type

        Returns:
            Guidance content with hints and suggestions
        """
        operation_type = step_context.get("operation_type", "")
        data_issues = step_context.get("data_issues", [])
        difficulty_level = step_context.get("difficulty_level", "beginner")

        # Convert difficulty level string to enum
        difficulty_map = {
            "beginner": DifficultyLevel.BEGINNER,
            "intermediate": DifficultyLevel.INTERMEDIATE,
            "advanced": DifficultyLevel.ADVANCED
        }
        level = difficulty_map.get(difficulty_level, DifficultyLevel.BEGINNER)

        # Build guidance based on context
        guidance = {
            "operation_type": operation_type,
            "hints": [],
            "suggestions": [],
            "warnings": []
        }

        # Add operation-specific guidance
        if operation_type == "data_cleaning":
            guidance["hints"].append("Start by identifying missing values and duplicates")
            guidance["suggestions"].append("Use descriptive statistics to understand data distribution")

            if "missing_values" in data_issues:
                guidance["warnings"].append("Dataset contains missing values that need handling")
                guidance["suggestions"].append("Consider imputation strategies: mean, median, or mode")

            if "duplicates" in data_issues:
                guidance["warnings"].append("Duplicate rows detected in the dataset")
                guidance["suggestions"].append("Review duplicates before removal to avoid data loss")

        elif operation_type == "feature_engineering":
            guidance["hints"].append("Create features that capture domain knowledge")
            guidance["suggestions"].append("Consider interaction terms and polynomial features")

            if level == DifficultyLevel.BEGINNER:
                guidance["hints"].append("Start with simple transformations like ratios and differences")
            else:
                guidance["hints"].append("Explore domain-specific features and advanced transformations")

        elif operation_type == "transformation":
            guidance["hints"].append("Normalize or standardize numerical features")
            guidance["suggestions"].append("Encode categorical variables appropriately")

            if "skewed_distribution" in data_issues:
                guidance["warnings"].append("Some features have skewed distributions")
                guidance["suggestions"].append("Consider log transformation or Box-Cox transformation")

        elif operation_type == "validation":
            guidance["hints"].append("Check data quality and schema compliance")
            guidance["suggestions"].append("Validate data types and value ranges")

        # Add related concepts
        concept_map = {
            "data_cleaning": ["missing_data", "outliers", "duplicates"],
            "feature_engineering": ["feature_creation", "feature_selection", "domain_knowledge"],
            "transformation": ["normalization", "encoding", "scaling"],
            "validation": ["data_quality", "schema_validation", "constraints"]
        }

        concepts = concept_map.get(operation_type, [])
        guidance["related_concepts"] = []

        for concept_id in concepts:
            try:
                content = self._educational_service.get_concept_explanation(concept_id, level)
                if content:
                    guidance["related_concepts"].append({
                        "concept_id": content.concept_id,
                        "title": content.title,
                        "summary": content.get_summary()
                    })
            except:
                # Concept not found, skip
                pass

        return guidance

    def search_concepts(self, keyword: str) -> List[Dict[str, Any]]:
        """Search for educational concepts.

        Args:
            keyword: The keyword to search for.

        Returns:
            A list of concept dictionaries.
        """
        concepts = self._educational_service.search_concepts(keyword)
        return [concept.to_dict() for concept in concepts]

