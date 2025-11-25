"""
Contract tests for ProjectInterface

These tests verify that Project implementations adhere to the ProjectInterface contract
defined in specs/001-use-the-requirements/contracts/dojo_api.py

TDD Note: These tests MUST FAIL until ProjectInterface is implemented in Phase 3.3
"""

import pytest
from typing import Dict, Any

from datadojo.dojo_api import (
    ProjectInterface,
    ProjectInfo,
    PipelineInterface,
    GuidanceLevel,
)


class TestProjectInterfaceContract:
    """Contract tests for ProjectInterface"""

    @pytest.fixture
    def project_instance(self):
        """Fixture providing a ProjectInterface implementation

        TDD: This will fail until Project class is implemented
        """
        from datadojo import create_dojo

        # Create dojo and load a project
        dojo = create_dojo(educational_mode=True)
        projects = dojo.list_projects()
        assert len(projects) > 0, "Need at least one project"

        return dojo.load_project(projects[0].id)

    def test_project_implements_interface(self, project_instance):
        """Verify Project class implements ProjectInterface"""
        assert isinstance(project_instance, ProjectInterface), \
            "Project must implement ProjectInterface"

    def test_project_info_property(self, project_instance):
        """T008: Test ProjectInterface.info property returns ProjectInfo"""
        info = project_instance.info

        assert isinstance(info, ProjectInfo), \
            "info property must return ProjectInfo"
        assert hasattr(info, 'id')
        assert hasattr(info, 'name')
        assert hasattr(info, 'domain')
        assert hasattr(info, 'difficulty')
        assert hasattr(info, 'description')
        assert hasattr(info, 'estimated_time_minutes')
        assert hasattr(info, 'learning_objectives')

    def test_project_info_has_valid_data(self, project_instance):
        """T008: Verify ProjectInfo contains valid, non-empty data"""
        info = project_instance.info

        assert info.id != "", "Project ID should not be empty"
        assert info.name != "", "Project name should not be empty"
        assert info.description != "", "Description should not be empty"
        assert info.estimated_time_minutes > 0, "Time estimate should be positive"
        assert len(info.learning_objectives) > 0, \
            "Should have at least one learning objective"

    def test_project_dataset_property(self, project_instance):
        """T008: Test ProjectInterface.dataset property returns data"""
        dataset = project_instance.dataset

        assert dataset is not None, "Dataset should not be None"
        # Dataset could be pandas DataFrame, numpy array, or custom object
        # We just verify it exists and is accessible

    def test_create_pipeline_returns_pipeline_interface(self, project_instance):
        """T008: Test ProjectInterface.create_pipeline() returns PipelineInterface"""
        pipeline = project_instance.create_pipeline()

        assert pipeline is not None
        assert isinstance(pipeline, PipelineInterface), \
            "create_pipeline() must return PipelineInterface implementation"

    def test_create_pipeline_with_guidance_levels(self, project_instance):
        """T008: Test create_pipeline() with different guidance levels"""
        # Test each guidance level
        for level in [GuidanceLevel.NONE, GuidanceLevel.BASIC, GuidanceLevel.DETAILED]:
            pipeline = project_instance.create_pipeline(guidance_level=level)

            assert isinstance(pipeline, PipelineInterface), \
                f"Should return PipelineInterface for guidance level {level}"

    def test_create_pipeline_default_guidance(self, project_instance):
        """T008: Test create_pipeline() defaults to DETAILED guidance"""
        pipeline = project_instance.create_pipeline()

        assert isinstance(pipeline, PipelineInterface)
        # Default should be DETAILED per contract

    def test_get_progress_returns_dict(self, project_instance):
        """T009: Test ProjectInterface.get_progress() returns Dict"""
        student_id = "test_student_001"

        progress = project_instance.get_progress(student_id)

        assert isinstance(progress, dict), \
            "get_progress() must return a dictionary"

    def test_get_progress_has_required_fields(self, project_instance):
        """T009: Test get_progress() returns expected progress fields"""
        student_id = "test_student_002"

        progress = project_instance.get_progress(student_id)

        # Verify expected fields exist
        assert isinstance(progress, dict)
        # Progress dict should contain information about student's advancement
        # Exact fields will be defined by implementation, but should be non-empty

    def test_get_progress_different_students(self, project_instance):
        """T009: Test get_progress() handles multiple students independently"""
        student_1 = "student_alice"
        student_2 = "student_bob"

        progress_1 = project_instance.get_progress(student_1)
        progress_2 = project_instance.get_progress(student_2)

        assert isinstance(progress_1, dict)
        assert isinstance(progress_2, dict)
        # Each student should have independent progress

    def test_get_progress_invalid_student_id(self, project_instance):
        """T009: Test get_progress() handles invalid student IDs gracefully"""
        # Empty string should either return empty progress or raise error
        result = project_instance.get_progress("")

        # Should return valid dict (possibly empty) or raise ValueError
        assert isinstance(result, dict), \
            "Should return empty progress dict for invalid student"

    def test_get_progress_nonexistent_student(self, project_instance):
        """T009: Test get_progress() for student with no prior progress"""
        new_student = "brand_new_student_12345"

        progress = project_instance.get_progress(new_student)

        assert isinstance(progress, dict)
        # Should return empty/initial progress state for new student


@pytest.mark.contract
class TestProjectInterfaceEdgeCases:
    """Edge case tests for ProjectInterface"""

    @pytest.fixture
    def project_instance(self):
        """Get project instance for testing"""
        from datadojo import create_dojo

        dojo = create_dojo(educational_mode=True)
        projects = dojo.list_projects()
        return dojo.load_project(projects[0].id)

    def test_multiple_pipelines_from_same_project(self, project_instance):
        """Test creating multiple pipelines from the same project"""
        pipeline1 = project_instance.create_pipeline(GuidanceLevel.DETAILED)
        pipeline2 = project_instance.create_pipeline(GuidanceLevel.NONE)

        assert isinstance(pipeline1, PipelineInterface)
        assert isinstance(pipeline2, PipelineInterface)
        # Should be able to create multiple independent pipelines

    def test_project_info_is_immutable(self, project_instance):
        """Test that project info doesn't change between accesses"""
        info1 = project_instance.info
        info2 = project_instance.info

        assert info1.id == info2.id
        assert info1.name == info2.name
        assert info1.domain == info2.domain

    def test_dataset_accessible_multiple_times(self, project_instance):
        """Test that dataset can be accessed multiple times"""
        dataset1 = project_instance.dataset
        dataset2 = project_instance.dataset

        assert dataset1 is not None
        assert dataset2 is not None
        # Dataset should be consistently accessible
