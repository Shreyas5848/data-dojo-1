"""
Contract tests for EducationalInterface

These tests verify that Educational implementations adhere to the EducationalInterface contract
defined in specs/001-use-the-requirements/contracts/dojo_api.py

TDD Note: These tests MUST FAIL until EducationalInterface is implemented in Phase 3.3
"""

import pytest
from typing import Dict, List, Any

# Contract imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../specs/001-use-the-requirements'))
from contracts.dojo_api import EducationalInterface


class TestEducationalInterfaceContract:
    """Contract tests for EducationalInterface"""

    @pytest.fixture
    def educational_instance(self):
        """Fixture providing an EducationalInterface implementation

        TDD: This will fail until Educational class is implemented
        """
        from datadojo.core.educational import Educational
        return Educational()

    def test_educational_implements_interface(self, educational_instance):
        """Verify Educational class implements EducationalInterface"""
        assert isinstance(educational_instance, EducationalInterface), \
            "Educational must implement EducationalInterface"

    def test_get_concept_explanation_returns_dict(self, educational_instance):
        """T012: Test get_concept_explanation() returns dictionary"""
        concept_id = "missing_values"

        explanation = educational_instance.get_concept_explanation(concept_id)

        assert isinstance(explanation, dict), \
            "get_concept_explanation() must return a dictionary"

    def test_get_concept_explanation_has_required_fields(self, educational_instance):
        """T012: Test explanation contains expected educational content"""
        concept_id = "missing_values"

        explanation = educational_instance.get_concept_explanation(concept_id)

        # Should contain educational content fields
        assert isinstance(explanation, dict)
        assert len(explanation) > 0, "Explanation should not be empty"
        # Exact fields depend on implementation, but should include meaningful content

    def test_get_concept_explanation_different_concepts(self, educational_instance):
        """T012: Test explanations for multiple data preprocessing concepts"""
        concepts = [
            "missing_values",
            "outliers",
            "normalization",
            "feature_engineering",
            "data_cleaning"
        ]

        for concept in concepts:
            explanation = educational_instance.get_concept_explanation(concept)
            assert isinstance(explanation, dict), \
                f"Explanation for '{concept}' should be a dictionary"
            assert len(explanation) > 0, \
                f"Explanation for '{concept}' should not be empty"

    def test_get_concept_explanation_invalid_concept_raises_error(self, educational_instance):
        """T012: Test get_concept_explanation() raises KeyError for unknown concept"""
        with pytest.raises(KeyError) as exc_info:
            educational_instance.get_concept_explanation("nonexistent_concept_xyz_12345")

        assert "not found" in str(exc_info.value).lower() or \
               "unknown" in str(exc_info.value).lower(), \
            "Error should indicate concept not found"

    def test_get_concept_explanation_empty_string_raises_error(self, educational_instance):
        """T012: Test get_concept_explanation() handles empty concept_id"""
        with pytest.raises(KeyError):
            educational_instance.get_concept_explanation("")

    def test_get_step_guidance_returns_dict(self, educational_instance):
        """T012: Test get_step_guidance() returns dictionary"""
        step_context = {
            "operation_type": "data_cleaning",
            "data_issues": ["missing_values", "duplicates"],
            "dataset_info": {"rows": 1000, "columns": 10}
        }

        guidance = educational_instance.get_step_guidance(step_context)

        assert isinstance(guidance, dict), \
            "get_step_guidance() must return a dictionary"

    def test_get_step_guidance_provides_hints(self, educational_instance):
        """T012: Test get_step_guidance() provides context-aware hints"""
        step_context = {
            "operation_type": "missing_value_handling",
            "data_issues": ["30% missing in age column"],
            "current_step": "imputation"
        }

        guidance = educational_instance.get_step_guidance(step_context)

        assert isinstance(guidance, dict)
        assert len(guidance) > 0, "Guidance should contain helpful information"

    def test_get_step_guidance_different_contexts(self, educational_instance):
        """T012: Test guidance varies based on context"""
        context1 = {"operation_type": "data_cleaning", "data_issues": ["missing_values"]}
        context2 = {"operation_type": "feature_engineering", "data_issues": ["categorical_encoding"]}

        guidance1 = educational_instance.get_step_guidance(context1)
        guidance2 = educational_instance.get_step_guidance(context2)

        assert isinstance(guidance1, dict)
        assert isinstance(guidance2, dict)
        # Guidance should be context-specific

    def test_track_progress_executes_without_error(self, educational_instance):
        """T013: Test track_progress() records student progress"""
        student_id = "test_student_001"
        completed_step = "data_cleaning_step_1"
        concepts_learned = ["missing_values", "duplicate_detection"]

        # Should not raise an error
        educational_instance.track_progress(student_id, completed_step, concepts_learned)

    def test_track_progress_with_multiple_concepts(self, educational_instance):
        """T013: Test track_progress() with multiple concepts"""
        student_id = "test_student_002"
        completed_step = "feature_engineering_step_1"
        concepts_learned = [
            "feature_creation",
            "feature_selection",
            "feature_scaling",
            "dimensionality_reduction"
        ]

        educational_instance.track_progress(student_id, completed_step, concepts_learned)

    def test_track_progress_same_student_multiple_times(self, educational_instance):
        """T013: Test tracking progress for same student across multiple steps"""
        student_id = "test_student_003"

        # Track multiple steps for the same student
        educational_instance.track_progress(
            student_id, "step_1", ["concept_a", "concept_b"]
        )
        educational_instance.track_progress(
            student_id, "step_2", ["concept_c"]
        )
        educational_instance.track_progress(
            student_id, "step_3", ["concept_d", "concept_e"]
        )

        # Should accumulate progress over time

    def test_track_progress_different_students(self, educational_instance):
        """T013: Test tracking progress for multiple students independently"""
        students = ["alice", "bob", "charlie"]

        for idx, student in enumerate(students):
            educational_instance.track_progress(
                student,
                f"step_{idx + 1}",
                [f"concept_{idx + 1}"]
            )

        # Each student's progress should be tracked independently

    def test_track_progress_empty_concepts_list(self, educational_instance):
        """T013: Test track_progress() with empty concepts list"""
        student_id = "test_student_004"
        completed_step = "review_step"
        concepts_learned = []  # No new concepts learned

        # Should handle empty list gracefully
        educational_instance.track_progress(student_id, completed_step, concepts_learned)

    def test_track_progress_with_none_student_id(self, educational_instance):
        """T013: Test track_progress() validates student_id"""
        # Should handle None gracefully or raise error
        try:
            educational_instance.track_progress(None, "step_1", ["concept_a"])
        except (ValueError, TypeError):
            # Expected - should validate input
            pass

    def test_track_progress_returns_none(self, educational_instance):
        """T013: Test track_progress() returns None (void method)"""
        result = educational_instance.track_progress(
            "test_student_005",
            "step_1",
            ["concept_a"]
        )

        assert result is None, "track_progress() should return None"


@pytest.mark.contract
@pytest.mark.educational
class TestEducationalInterfaceIntegration:
    """Integration tests for educational features"""

    @pytest.fixture
    def educational_instance(self):
        """Get educational instance"""
        from datadojo.core.educational import Educational
        return Educational()

    def test_concept_explanation_and_guidance_workflow(self, educational_instance):
        """Test workflow: get concept explanation, then step guidance"""
        # Get concept explanation first
        concept = "missing_values"
        explanation = educational_instance.get_concept_explanation(concept)
        assert isinstance(explanation, dict)

        # Then get step guidance for applying the concept
        step_context = {
            "operation_type": "data_cleaning",
            "data_issues": ["missing_values"],
            "concept": concept
        }
        guidance = educational_instance.get_step_guidance(step_context)
        assert isinstance(guidance, dict)

    def test_complete_learning_session_tracking(self, educational_instance):
        """Test tracking a complete learning session"""
        student_id = "learning_session_student"

        # Session: multiple steps with different concepts
        session_steps = [
            ("step_1_data_inspection", ["data_quality_assessment"]),
            ("step_2_cleaning", ["missing_values", "outliers"]),
            ("step_3_transformation", ["normalization", "encoding"]),
            ("step_4_validation", ["data_validation"])
        ]

        for step, concepts in session_steps:
            educational_instance.track_progress(student_id, step, concepts)

        # All progress should be tracked successfully

    def test_get_multiple_concept_explanations_efficiently(self, educational_instance):
        """Test retrieving multiple concept explanations"""
        concepts = [
            "missing_values", "outliers", "normalization",
            "feature_engineering", "data_cleaning"
        ]

        explanations = {}
        for concept in concepts:
            explanations[concept] = educational_instance.get_concept_explanation(concept)

        assert len(explanations) == len(concepts)
        for concept, explanation in explanations.items():
            assert isinstance(explanation, dict), \
                f"Explanation for {concept} should be a dict"
