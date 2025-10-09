"""
Contract tests for DojoInterface

These tests verify that any implementation of DojoInterface adheres to the contract
defined in specs/001-use-the-requirements/contracts/dojo_api.py

TDD Note: These tests MUST FAIL until DojoInterface is implemented in Phase 3.3
"""

import pytest
from typing import Optional

# Contract imports - these define what we're testing against
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../specs/001-use-the-requirements'))
from contracts.dojo_api import (
    DojoInterface,
    ProjectInterface,
    ProjectInfo,
    Domain,
    DifficultyLevel,
)


class TestDojoInterfaceContract:
    """Contract tests for DojoInterface - the main entry point"""

    @pytest.fixture
    def dojo_instance(self):
        """Fixture that will provide DojoInterface implementation

        TDD: This will fail until we implement Dojo class in Phase 3.3
        """
        # This import will fail - that's expected in TDD!
        from datadojo.core.dojo import Dojo
        return Dojo(educational_mode=True)

    def test_dojo_implements_interface(self, dojo_instance):
        """Verify that Dojo class properly implements DojoInterface"""
        assert isinstance(dojo_instance, DojoInterface), \
            "Dojo must implement DojoInterface"

    def test_dojo_initialization_with_educational_mode(self):
        """T006: Test DojoInterface.__init__ with educational_mode parameter"""
        from datadojo.core.dojo import Dojo

        # Test with educational mode enabled
        dojo = Dojo(educational_mode=True)
        assert dojo is not None
        assert isinstance(dojo, DojoInterface)

        # Test with educational mode disabled
        dojo_prod = Dojo(educational_mode=False)
        assert dojo_prod is not None
        assert isinstance(dojo_prod, DojoInterface)

    def test_list_projects_returns_list_of_project_info(self, dojo_instance):
        """T006: Test DojoInterface.list_projects() returns List[ProjectInfo]"""
        # Call without filters
        projects = dojo_instance.list_projects()

        assert isinstance(projects, list), \
            "list_projects() must return a list"
        assert len(projects) > 0, \
            "Should have at least one project available"

        # Verify each item is ProjectInfo
        for project in projects:
            assert isinstance(project, ProjectInfo), \
                f"Each item must be ProjectInfo, got {type(project)}"
            assert hasattr(project, 'id')
            assert hasattr(project, 'name')
            assert hasattr(project, 'domain')
            assert hasattr(project, 'difficulty')

    def test_list_projects_filter_by_domain(self, dojo_instance):
        """T006: Test DojoInterface.list_projects() with domain filter"""
        # Filter by e-commerce domain
        ecommerce_projects = dojo_instance.list_projects(domain=Domain.ECOMMERCE)

        assert isinstance(ecommerce_projects, list)
        for project in ecommerce_projects:
            assert project.domain == Domain.ECOMMERCE, \
                "All projects should be from e-commerce domain"

    def test_list_projects_filter_by_difficulty(self, dojo_instance):
        """T006: Test DojoInterface.list_projects() with difficulty filter"""
        # Filter by beginner difficulty
        beginner_projects = dojo_instance.list_projects(
            difficulty=DifficultyLevel.BEGINNER
        )

        assert isinstance(beginner_projects, list)
        for project in beginner_projects:
            assert project.difficulty == DifficultyLevel.BEGINNER, \
                "All projects should be beginner level"

    def test_list_projects_filter_by_both(self, dojo_instance):
        """T006: Test DojoInterface.list_projects() with both filters"""
        # Filter by both domain and difficulty
        filtered_projects = dojo_instance.list_projects(
            domain=Domain.HEALTHCARE,
            difficulty=DifficultyLevel.INTERMEDIATE
        )

        assert isinstance(filtered_projects, list)
        for project in filtered_projects:
            assert project.domain == Domain.HEALTHCARE
            assert project.difficulty == DifficultyLevel.INTERMEDIATE

    def test_load_project_returns_project_interface(self, dojo_instance):
        """T007: Test DojoInterface.load_project() returns ProjectInterface"""
        # First get a valid project ID
        projects = dojo_instance.list_projects()
        assert len(projects) > 0, "Need at least one project to test"

        project_id = projects[0].id

        # Load the project
        project = dojo_instance.load_project(project_id)

        assert project is not None
        assert isinstance(project, ProjectInterface), \
            "load_project() must return ProjectInterface implementation"

    def test_load_project_with_difficulty_override(self, dojo_instance):
        """T007: Test DojoInterface.load_project() with difficulty override"""
        projects = dojo_instance.list_projects()
        project_id = projects[0].id

        # Load with difficulty override
        project = dojo_instance.load_project(
            project_id,
            difficulty=DifficultyLevel.ADVANCED
        )

        assert project is not None
        assert isinstance(project, ProjectInterface)

    def test_load_project_invalid_id_raises_error(self, dojo_instance):
        """T007: Test DojoInterface.load_project() raises ValueError for invalid ID"""
        with pytest.raises(ValueError) as exc_info:
            dojo_instance.load_project("nonexistent_project_id_12345")

        assert "not found" in str(exc_info.value).lower(), \
            "Error message should indicate project not found"

    def test_load_project_none_id_raises_error(self, dojo_instance):
        """T007: Test DojoInterface.load_project() handles None project_id"""
        with pytest.raises((ValueError, TypeError)):
            dojo_instance.load_project(None)

    def test_load_project_empty_string_raises_error(self, dojo_instance):
        """T007: Test DojoInterface.load_project() handles empty string"""
        with pytest.raises(ValueError):
            dojo_instance.load_project("")


@pytest.mark.contract
class TestDojoFactoryFunction:
    """Test the factory function for creating Dojo instances"""

    def test_create_dojo_function_exists(self):
        """Verify create_dojo factory function is available"""
        from datadojo import create_dojo
        assert callable(create_dojo)

    def test_create_dojo_returns_dojo_interface(self):
        """Test that create_dojo() returns DojoInterface implementation"""
        from datadojo import create_dojo

        dojo = create_dojo(educational_mode=True)
        assert isinstance(dojo, DojoInterface)

    def test_create_dojo_default_educational_mode(self):
        """Test create_dojo() defaults to educational mode"""
        from datadojo import create_dojo

        dojo = create_dojo()  # Should default to educational_mode=True
        assert isinstance(dojo, DojoInterface)
