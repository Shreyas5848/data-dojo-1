"""Unit tests for DataDojo data models."""

import pytest
from datetime import datetime
from datadojo.models.learning_project import (
    LearningProject, Domain, Difficulty, ValidationRule
)
from datadojo.models.pipeline import Pipeline, ExecutionState, GuidanceLevel
from datadojo.models.processing_step import (
    ProcessingStep, OperationType, CompletionStatus
)
from datadojo.models.progress_tracker import ProgressTracker
from datadojo.models.domain_module import DomainModule, OperationDefinition
from datadojo.models.educational_content import EducationalContent, DifficultyLevel
from datadojo.models.dataset import Dataset, DataFormat


class TestLearningProject:
    """Tests for LearningProject model."""

    def test_create_learning_project(self):
        """Test creating a basic learning project."""
        project = LearningProject(
            id="test-001",
            name="Test Project",
            domain=Domain.GENERAL,
            difficulty=Difficulty.BEGINNER,
            description="A test project",
            dataset_path="/path/to/data.csv"
        )

        assert project.id == "test-001"
        assert project.name == "Test Project"
        assert project.domain == Domain.GENERAL
        assert project.difficulty == Difficulty.BEGINNER
        assert project.description == "A test project"
        assert project.dataset_path == "/path/to/data.csv"
        assert project.expected_outcomes == []
        assert project.validation_rules == []

    def test_learning_project_with_outcomes(self):
        """Test project with expected outcomes."""
        project = LearningProject(
            id="test-002",
            name="Test Project",
            domain=Domain.ECOMMERCE,
            difficulty=Difficulty.INTERMEDIATE,
            description="Test",
            dataset_path="/data.csv",
            expected_outcomes=["Outcome 1", "Outcome 2"]
        )

        assert len(project.expected_outcomes) == 2
        assert "Outcome 1" in project.expected_outcomes

    def test_learning_project_with_validation_rules(self):
        """Test project with validation rules."""
        rule = ValidationRule(
            rule_id="rule-1",
            description="Check age range",
            validation_function="check_age_range",
            parameters={"min": 0, "max": 120}
        )

        project = LearningProject(
            id="test-003",
            name="Test",
            domain=Domain.HEALTHCARE,
            difficulty=Difficulty.ADVANCED,
            description="Test",
            dataset_path="/data.csv",
            validation_rules=[rule]
        )

        assert len(project.validation_rules) == 1
        assert project.validation_rules[0].rule_id == "rule-1"

    def test_learning_project_serialization(self):
        """Test project to_dict and from_dict."""
        project = LearningProject(
            id="test-004",
            name="Test",
            domain=Domain.FINANCE,
            difficulty=Difficulty.BEGINNER,
            description="Test project",
            dataset_path="/data.csv"
        )

        # Serialize
        data = project.to_dict()
        assert data["id"] == "test-004"
        assert data["domain"] == "finance"
        assert data["difficulty"] == "beginner"

        # Deserialize
        restored = LearningProject.from_dict(data)
        assert restored.id == project.id
        assert restored.domain == project.domain
        assert restored.difficulty == project.difficulty


class TestPipeline:
    """Tests for Pipeline model."""

    def test_create_pipeline(self):
        """Test creating a pipeline."""
        pipeline = Pipeline(
            id="pipeline-001",
            project_id="project-001",
            name="Test Pipeline",
            steps=[]
        )

        assert pipeline.id == "pipeline-001"
        assert pipeline.project_id == "project-001"
        assert pipeline.name == "Test Pipeline"
        assert pipeline.state == ExecutionState.NOT_STARTED
        assert pipeline.guidance_level == GuidanceLevel.DETAILED
        assert len(pipeline.steps) == 0

    def test_pipeline_state_transitions(self):
        """Test pipeline state management."""
        pipeline = Pipeline(
            id="p-001",
            project_id="proj-001",
            name="Test",
            steps=[]
        )

        assert pipeline.state == ExecutionState.NOT_STARTED

        # Start execution
        pipeline.start_execution()
        assert pipeline.state == ExecutionState.RUNNING
        assert pipeline.started_at is not None

        # Complete execution
        pipeline.complete_execution()
        assert pipeline.state == ExecutionState.COMPLETED
        assert pipeline.completed_at is not None

    def test_pipeline_error_handling(self):
        """Test pipeline error state."""
        pipeline = Pipeline(
            id="p-002",
            project_id="proj-002",
            name="Test",
            steps=[]
        )

        pipeline.start_execution()
        pipeline.mark_error("Test error message")

        assert pipeline.state == ExecutionState.FAILED
        assert pipeline.error_message == "Test error message"

    def test_pipeline_with_steps(self):
        """Test pipeline with processing steps."""
        step1 = ProcessingStep(
            id="step-1",
            name="Load Data",
            operation_type=OperationType.DATA_LOADING,
            description="Load dataset"
        )

        step2 = ProcessingStep(
            id="step-2",
            name="Clean Data",
            operation_type=OperationType.DATA_CLEANING,
            description="Clean the data",
            prerequisites=["step-1"]
        )

        pipeline = Pipeline(
            id="p-003",
            project_id="proj-003",
            name="Test",
            steps=[step1, step2]
        )

        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].id == "step-1"
        assert pipeline.steps[1].prerequisites == ["step-1"]

    def test_pipeline_serialization(self):
        """Test pipeline serialization."""
        pipeline = Pipeline(
            id="p-004",
            project_id="proj-004",
            name="Test Pipeline",
            steps=[]
        )

        data = pipeline.to_dict()
        assert data["id"] == "p-004"
        assert data["state"] == "not_started"

        restored = Pipeline.from_dict(data)
        assert restored.id == pipeline.id
        assert restored.state == pipeline.state


class TestProcessingStep:
    """Tests for ProcessingStep model."""

    def test_create_processing_step(self):
        """Test creating a processing step."""
        step = ProcessingStep(
            id="step-001",
            name="Data Cleaning",
            operation_type=OperationType.DATA_CLEANING,
            description="Clean the dataset"
        )

        assert step.id == "step-001"
        assert step.name == "Data Cleaning"
        assert step.operation_type == OperationType.DATA_CLEANING
        assert step.status == CompletionStatus.NOT_STARTED

    def test_step_prerequisites(self):
        """Test step with prerequisites."""
        step = ProcessingStep(
            id="step-002",
            name="Feature Engineering",
            operation_type=OperationType.FEATURE_ENGINEERING,
            description="Create features",
            prerequisites=["step-001"]
        )

        assert len(step.prerequisites) == 1
        assert not step.is_ready([])
        assert step.is_ready(["step-001"])

    def test_step_status_transitions(self):
        """Test step status changes."""
        step = ProcessingStep(
            id="step-003",
            name="Test",
            operation_type=OperationType.TRANSFORMATION,
            description="Test"
        )

        assert step.status == CompletionStatus.NOT_STARTED

        step.mark_in_progress()
        assert step.status == CompletionStatus.IN_PROGRESS

        step.mark_completed()
        assert step.status == CompletionStatus.COMPLETED
        assert step.completed_at is not None

    def test_step_with_learned_concepts(self):
        """Test step with learned concepts."""
        step = ProcessingStep(
            id="step-004",
            name="Handle Missing Values",
            operation_type=OperationType.DATA_CLEANING,
            description="Impute missing values",
            learned_concepts=["missing_values", "imputation"]
        )

        assert len(step.learned_concepts) == 2
        assert "missing_values" in step.learned_concepts

    def test_step_serialization(self):
        """Test step serialization."""
        step = ProcessingStep(
            id="step-005",
            name="Test",
            operation_type=OperationType.VALIDATION,
            description="Validate data"
        )

        data = step.to_dict()
        assert data["id"] == "step-005"
        assert data["operation_type"] == "validation"

        restored = ProcessingStep.from_dict(data)
        assert restored.id == step.id
        assert restored.operation_type == step.operation_type


class TestProgressTracker:
    """Tests for ProgressTracker model."""

    def test_create_progress_tracker(self):
        """Test creating a progress tracker."""
        tracker = ProgressTracker(
            student_id="student-001",
            project_id="project-001"
        )

        assert tracker.student_id == "student-001"
        assert tracker.project_id == "project-001"
        assert len(tracker.completed_steps) == 0
        assert len(tracker.learned_concepts) == 0

    def test_track_step_completion(self):
        """Test tracking step completion."""
        tracker = ProgressTracker(
            student_id="s-001",
            project_id="p-001"
        )

        tracker.complete_step("step-1")
        assert "step-1" in tracker.completed_steps
        assert "step-1" in tracker.step_completion_dates

    def test_track_concept_learning(self):
        """Test tracking learned concepts."""
        tracker = ProgressTracker(
            student_id="s-002",
            project_id="p-002"
        )

        tracker.learn_concept("missing_values")
        tracker.learn_concept("outliers")

        assert len(tracker.learned_concepts) == 2
        assert "missing_values" in tracker.learned_concepts

    def test_skill_assessment(self):
        """Test skill assessment tracking."""
        tracker = ProgressTracker(
            student_id="s-003",
            project_id="p-003"
        )

        tracker.update_skill_score("data_cleaning", 75.0)
        tracker.update_skill_score("feature_engineering", 60.0)

        assert tracker.skill_assessments["data_cleaning"] == 75.0
        avg_score = tracker.get_average_skill_score()
        assert avg_score == 67.5

    def test_completion_percentage(self):
        """Test completion percentage calculation."""
        tracker = ProgressTracker(
            student_id="s-004",
            project_id="p-004"
        )

        tracker.complete_step("step-1")
        tracker.complete_step("step-2")

        percentage = tracker.get_completion_percentage(total_steps=4)
        assert percentage == 50.0

    def test_progress_serialization(self):
        """Test progress tracker serialization."""
        tracker = ProgressTracker(
            student_id="s-005",
            project_id="p-005"
        )
        tracker.complete_step("step-1")

        data = tracker.to_dict()
        assert data["student_id"] == "s-005"
        assert "step-1" in data["completed_steps"]

        restored = ProgressTracker.from_dict(data)
        assert restored.student_id == tracker.student_id
        assert "step-1" in restored.completed_steps


class TestDomainModule:
    """Tests for DomainModule model."""

    def test_create_domain_module(self):
        """Test creating a domain module."""
        module = DomainModule(
            domain_id="test-domain",
            name="Test Domain",
            description="A test domain"
        )

        assert module.domain_id == "test-domain"
        assert module.name == "Test Domain"
        assert len(module.operations) == 0

    def test_domain_with_operations(self):
        """Test domain with operations."""
        op = OperationDefinition(
            operation_id="op-1",
            name="Test Operation",
            description="Test",
            operation_type="transformation"
        )

        module = DomainModule(
            domain_id="domain-1",
            name="Test",
            description="Test",
            operations=[op]
        )

        assert len(module.operations) == 1
        assert module.operations[0].operation_id == "op-1"

    def test_domain_serialization(self):
        """Test domain module serialization."""
        module = DomainModule(
            domain_id="domain-2",
            name="Test Domain",
            description="Test"
        )

        data = module.to_dict()
        assert data["domain_id"] == "domain-2"

        restored = DomainModule.from_dict(data)
        assert restored.domain_id == module.domain_id


class TestEducationalContent:
    """Tests for EducationalContent model."""

    def test_create_educational_content(self):
        """Test creating educational content."""
        content = EducationalContent(
            concept_id="concept-001",
            title="Test Concept",
            explanation="This is a test concept"
        )

        assert content.concept_id == "concept-001"
        assert content.title == "Test Concept"
        assert content.difficulty_level == DifficultyLevel.BEGINNER

    def test_content_with_analogies(self):
        """Test content with analogies."""
        content = EducationalContent(
            concept_id="concept-002",
            title="Test",
            explanation="Test",
            analogies=["Like a car engine", "Similar to a recipe"]
        )

        assert len(content.analogies) == 2
        assert "Like a car engine" in content.analogies

    def test_content_with_examples(self):
        """Test content with code examples."""
        content = EducationalContent(
            concept_id="concept-003",
            title="Test",
            explanation="Test",
            examples=["df.head()", "df.info()"]
        )

        assert len(content.examples) == 2

    def test_content_get_summary(self):
        """Test getting content summary."""
        content = EducationalContent(
            concept_id="concept-004",
            title="Test Concept",
            explanation="This is a long explanation" * 20
        )

        summary = content.get_summary(max_length=50)
        assert len(summary) <= 53  # 50 + "..."

    def test_content_serialization(self):
        """Test educational content serialization."""
        content = EducationalContent(
            concept_id="concept-005",
            title="Test",
            explanation="Test content"
        )

        data = content.to_dict()
        assert data["concept_id"] == "concept-005"

        restored = EducationalContent.from_dict(data)
        assert restored.concept_id == content.concept_id


class TestDataset:
    """Tests for Dataset model."""

    def test_create_dataset(self):
        """Test creating a dataset."""
        dataset = Dataset(
            id="dataset-001",
            name="Test Dataset",
            file_path="/path/to/data.csv",
            format=DataFormat.CSV
        )

        assert dataset.id == "dataset-001"
        assert dataset.name == "Test Dataset"
        assert dataset.format == DataFormat.CSV

    def test_dataset_with_metadata(self):
        """Test dataset with metadata."""
        dataset = Dataset(
            id="dataset-002",
            name="Test",
            file_path="/data.csv",
            format=DataFormat.CSV,
            metadata={
                "rows": 1000,
                "columns": 10,
                "source": "test"
            }
        )

        assert dataset.metadata["rows"] == 1000
        assert dataset.metadata["columns"] == 10

    def test_dataset_quality_issues(self):
        """Test tracking quality issues."""
        dataset = Dataset(
            id="dataset-003",
            name="Test",
            file_path="/data.csv",
            format=DataFormat.CSV
        )

        dataset.add_quality_issue("missing_values", "High percentage of missing values")
        dataset.add_quality_issue("outliers", "Outliers detected in price column")

        assert len(dataset.quality_issues) == 2
        assert "missing_values" in dataset.quality_issues

    def test_dataset_serialization(self):
        """Test dataset serialization."""
        dataset = Dataset(
            id="dataset-004",
            name="Test",
            file_path="/data.csv",
            format=DataFormat.PARQUET
        )

        data = dataset.to_dict()
        assert data["id"] == "dataset-004"
        assert data["format"] == "parquet"

        restored = Dataset.from_dict(data)
        assert restored.id == dataset.id
        assert restored.format == dataset.format


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
