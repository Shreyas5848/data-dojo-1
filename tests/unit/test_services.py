"""Unit tests for DataDojo services."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datadojo.services.project_service import ProjectService
from datadojo.services.pipeline_service import PipelineService
from datadojo.services.educational_service import EducationalService
from datadojo.services.domain_service import DomainService
from datadojo.models.learning_project import LearningProject, Domain, Difficulty
from datadojo.models.pipeline import Pipeline, GuidanceLevel
from datadojo.models.processing_step import ProcessingStep, OperationType
from datadojo.models.progress_tracker import ProgressTracker
from datadojo.models.domain_module import DomainModule


class TestProjectService:
    """Tests for ProjectService."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def project_service(self, temp_storage):
        """Create ProjectService with temporary storage."""
        return ProjectService(storage_path=temp_storage)

    def test_create_project(self, project_service):
        """Test creating a project."""
        project = project_service.create_project(
            project_id="test-001",
            name="Test Project",
            domain=Domain.GENERAL,
            difficulty=Difficulty.BEGINNER,
            description="Test project",
            dataset_path="/data.csv"
        )

        assert project.id == "test-001"
        assert project.name == "Test Project"
        assert project.domain == Domain.GENERAL

    def test_get_project(self, project_service):
        """Test retrieving a project."""
        # Create project
        project_service.create_project(
            project_id="test-002",
            name="Test",
            domain=Domain.ECOMMERCE,
            difficulty=Difficulty.INTERMEDIATE,
            description="Test",
            dataset_path="/data.csv"
        )

        # Retrieve project
        retrieved = project_service.get_project("test-002")
        assert retrieved is not None
        assert retrieved.id == "test-002"

    def test_get_nonexistent_project(self, project_service):
        """Test getting project that doesn't exist."""
        project = project_service.get_project("nonexistent")
        assert project is None

    def test_list_all_projects(self, project_service):
        """Test listing all projects."""
        # Create multiple projects
        project_service.create_project(
            "p1", "Project 1", Domain.GENERAL,
            Difficulty.BEGINNER, "Test", "/data.csv"
        )
        project_service.create_project(
            "p2", "Project 2", Domain.FINANCE,
            Difficulty.ADVANCED, "Test", "/data.csv"
        )

        projects = project_service.list_projects()
        assert len(projects) == 2

    def test_filter_projects_by_domain(self, project_service):
        """Test filtering projects by domain."""
        project_service.create_project(
            "p1", "E-commerce", Domain.ECOMMERCE,
            Difficulty.BEGINNER, "Test", "/data.csv"
        )
        project_service.create_project(
            "p2", "Finance", Domain.FINANCE,
            Difficulty.BEGINNER, "Test", "/data.csv"
        )

        ecommerce_projects = project_service.list_projects(domain=Domain.ECOMMERCE)
        assert len(ecommerce_projects) == 1
        assert ecommerce_projects[0].domain == Domain.ECOMMERCE

    def test_filter_projects_by_difficulty(self, project_service):
        """Test filtering projects by difficulty."""
        project_service.create_project(
            "p1", "Easy", Domain.GENERAL,
            Difficulty.BEGINNER, "Test", "/data.csv"
        )
        project_service.create_project(
            "p2", "Hard", Domain.GENERAL,
            Difficulty.ADVANCED, "Test", "/data.csv"
        )

        beginner_projects = project_service.list_projects(difficulty=Difficulty.BEGINNER)
        assert len(beginner_projects) == 1
        assert beginner_projects[0].difficulty == Difficulty.BEGINNER

    def test_update_project(self, project_service):
        """Test updating a project."""
        # Create project
        project = project_service.create_project(
            "p1", "Original", Domain.GENERAL,
            Difficulty.BEGINNER, "Original description", "/data.csv"
        )

        # Update project
        project.name = "Updated Name"
        project.description = "Updated description"
        project_service.update_project(project)

        # Retrieve and verify
        updated = project_service.get_project("p1")
        assert updated.name == "Updated Name"
        assert updated.description == "Updated description"

    def test_delete_project(self, project_service):
        """Test deleting a project."""
        # Create project
        project_service.create_project(
            "p1", "Test", Domain.GENERAL,
            Difficulty.BEGINNER, "Test", "/data.csv"
        )

        # Delete project
        result = project_service.delete_project("p1")
        assert result is True

        # Verify deletion
        deleted = project_service.get_project("p1")
        assert deleted is None

    def test_delete_nonexistent_project(self, project_service):
        """Test deleting project that doesn't exist."""
        result = project_service.delete_project("nonexistent")
        assert result is False


class TestPipelineService:
    """Tests for PipelineService."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def pipeline_service(self, temp_storage):
        """Create PipelineService with temporary storage."""
        return PipelineService(storage_path=temp_storage)

    def test_create_pipeline(self, pipeline_service):
        """Test creating a pipeline."""
        pipeline = pipeline_service.create_pipeline(
            pipeline_id="pipe-001",
            project_id="proj-001",
            name="Test Pipeline"
        )

        assert pipeline.id == "pipe-001"
        assert pipeline.project_id == "proj-001"
        assert len(pipeline.steps) == 0

    def test_add_step_to_pipeline(self, pipeline_service):
        """Test adding a step to pipeline."""
        pipeline = pipeline_service.create_pipeline(
            "pipe-001", "proj-001", "Test"
        )

        step = ProcessingStep(
            id="step-1",
            name="Load Data",
            operation_type=OperationType.DATA_LOADING,
            description="Load dataset"
        )

        pipeline = pipeline_service.add_step(pipeline, step)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].id == "step-1"

    def test_execute_empty_pipeline(self, pipeline_service):
        """Test executing pipeline with no steps."""
        pipeline = pipeline_service.create_pipeline(
            "pipe-001", "proj-001", "Empty"
        )

        with pytest.raises(ValueError, match="Pipeline has no steps"):
            pipeline_service.execute_pipeline(pipeline, data=None)

    def test_pipeline_prerequisite_checking(self, pipeline_service):
        """Test that prerequisites are validated."""
        pipeline = pipeline_service.create_pipeline(
            "pipe-001", "proj-001", "Test"
        )

        step1 = ProcessingStep(
            id="step-1",
            name="Step 1",
            operation_type=OperationType.DATA_LOADING,
            description="Load"
        )

        step2 = ProcessingStep(
            id="step-2",
            name="Step 2",
            operation_type=OperationType.DATA_CLEANING,
            description="Clean",
            prerequisites=["step-1"]
        )

        pipeline = pipeline_service.add_step(pipeline, step2)
        pipeline = pipeline_service.add_step(pipeline, step1)

        # Should have both steps
        assert len(pipeline.steps) == 2

    def test_get_pipeline(self, pipeline_service):
        """Test retrieving a pipeline."""
        pipeline_service.create_pipeline("pipe-001", "proj-001", "Test")

        retrieved = pipeline_service.get_pipeline("pipe-001")
        assert retrieved is not None
        assert retrieved.id == "pipe-001"

    def test_list_pipelines_for_project(self, pipeline_service):
        """Test listing pipelines for a specific project."""
        pipeline_service.create_pipeline("pipe-1", "proj-1", "Pipeline 1")
        pipeline_service.create_pipeline("pipe-2", "proj-1", "Pipeline 2")
        pipeline_service.create_pipeline("pipe-3", "proj-2", "Pipeline 3")

        proj1_pipelines = pipeline_service.list_pipelines("proj-1")
        assert len(proj1_pipelines) == 2


class TestEducationalService:
    """Tests for EducationalService."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def educational_service(self, temp_storage):
        """Create EducationalService with temporary storage."""
        return EducationalService(storage_path=temp_storage)

    def test_get_concept(self, educational_service):
        """Test retrieving a concept."""
        concept = educational_service.get_concept("missing_values")
        assert concept is not None
        assert concept.concept_id == "missing_values"
        assert "Missing" in concept.title

    def test_get_nonexistent_concept(self, educational_service):
        """Test getting concept that doesn't exist."""
        concept = educational_service.get_concept("nonexistent_concept")
        assert concept is None

    def test_search_concepts(self, educational_service):
        """Test searching for concepts."""
        results = educational_service.search_concepts("missing")
        assert len(results) > 0
        assert any("missing" in c.title.lower() for c in results)

    def test_list_concepts_by_difficulty(self, educational_service):
        """Test listing concepts filtered by difficulty."""
        from datadojo.models.educational_content import DifficultyLevel

        beginner_concepts = educational_service.list_concepts(
            difficulty=DifficultyLevel.BEGINNER
        )
        assert len(beginner_concepts) > 0
        assert all(c.difficulty_level == DifficultyLevel.BEGINNER
                  for c in beginner_concepts)

    def test_get_progress(self, educational_service):
        """Test retrieving progress for a student."""
        progress = educational_service.get_progress("student-1", "project-1")
        assert progress.student_id == "student-1"
        assert progress.project_id == "project-1"

    def test_update_progress(self, educational_service):
        """Test updating student progress."""
        progress = educational_service.get_progress("student-1", "project-1")
        progress.complete_step("step-1")
        progress.learn_concept("missing_values")

        educational_service.update_progress(progress)

        # Retrieve and verify
        retrieved = educational_service.get_progress("student-1", "project-1")
        assert "step-1" in retrieved.completed_steps
        assert "missing_values" in retrieved.learned_concepts

    def test_provide_hint(self, educational_service):
        """Test providing context-aware hints."""
        from datadojo.models.educational_content import DifficultyLevel

        hint = educational_service.provide_hint(
            "I have missing values in my dataset",
            difficulty=DifficultyLevel.BEGINNER
        )
        assert hint is not None
        assert isinstance(hint, str)
        assert len(hint) > 0

    def test_get_guidance_for_step(self, educational_service):
        """Test getting guidance for a processing step."""
        step = ProcessingStep(
            id="step-1",
            name="Clean Data",
            operation_type=OperationType.DATA_CLEANING,
            description="Clean the dataset",
            learned_concepts=["missing_values"]
        )

        guidance = educational_service.get_guidance_for_step(step)
        assert "hints" in guidance
        assert "related_concepts" in guidance
        assert len(guidance["related_concepts"]) > 0


class TestDomainService:
    """Tests for DomainService."""

    @pytest.fixture
    def domain_service(self):
        """Create DomainService instance."""
        return DomainService()

    def test_list_available_domains(self, domain_service):
        """Test listing available domains."""
        domains = domain_service.list_domains()
        assert len(domains) > 0
        assert any(d.domain_id == "ecommerce" for d in domains)

    def test_get_domain(self, domain_service):
        """Test retrieving a specific domain."""
        domain = domain_service.get_domain("ecommerce")
        assert domain is not None
        assert domain.domain_id == "ecommerce"

    def test_get_nonexistent_domain(self, domain_service):
        """Test getting domain that doesn't exist."""
        domain = domain_service.get_domain("nonexistent_domain")
        assert domain is None

    def test_register_custom_domain(self, domain_service):
        """Test registering a custom domain."""
        custom_domain = DomainModule(
            domain_id="custom",
            name="Custom Domain",
            description="A custom test domain"
        )

        domain_service.register_domain(custom_domain)

        retrieved = domain_service.get_domain("custom")
        assert retrieved is not None
        assert retrieved.domain_id == "custom"

    def test_get_domain_projects(self, domain_service):
        """Test getting projects for a domain."""
        projects = domain_service.get_domain_projects("ecommerce")
        assert len(projects) > 0
        assert all(hasattr(p, 'domain') for p in projects)

    def test_suggest_domain_for_keywords(self, domain_service):
        """Test domain suggestion based on keywords."""
        suggestions = domain_service.suggest_domain(
            ["sales", "customer", "product"]
        )
        assert len(suggestions) > 0
        # E-commerce should be suggested for these keywords
        assert any(s.domain_id == "ecommerce" for s in suggestions)

    def test_get_domain_operations(self, domain_service):
        """Test retrieving domain-specific operations."""
        domain = domain_service.get_domain("finance")
        assert domain is not None
        assert len(domain.operations) > 0


class TestServiceIntegration:
    """Integration tests across multiple services."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def services(self, temp_storage):
        """Create all services with shared storage."""
        return {
            'project': ProjectService(storage_path=temp_storage),
            'pipeline': PipelineService(storage_path=temp_storage),
            'educational': EducationalService(storage_path=temp_storage),
            'domain': DomainService()
        }

    def test_full_project_workflow(self, services):
        """Test complete workflow from project creation to pipeline execution."""
        # Create project
        project = services['project'].create_project(
            "proj-001",
            "Test Project",
            Domain.GENERAL,
            Difficulty.BEGINNER,
            "Test workflow",
            "/data.csv"
        )

        # Create pipeline for project
        pipeline = services['pipeline'].create_pipeline(
            "pipe-001",
            project.id,
            "Test Pipeline"
        )

        # Add step to pipeline
        step = ProcessingStep(
            id="step-1",
            name="Data Validation",
            operation_type=OperationType.VALIDATION,
            description="Validate data",
            learned_concepts=["data_quality"]
        )

        pipeline = services['pipeline'].add_step(pipeline, step)

        # Track progress
        progress = services['educational'].get_progress("student-1", project.id)
        progress.complete_step(step.id)
        progress.learn_concept("data_quality")
        services['educational'].update_progress(progress)

        # Verify everything is connected
        assert project.id == pipeline.project_id
        assert len(pipeline.steps) == 1
        assert "step-1" in progress.completed_steps
        assert "data_quality" in progress.learned_concepts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
