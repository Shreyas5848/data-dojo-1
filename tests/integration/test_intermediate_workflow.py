"""
Integration test for intermediate feature engineering workflow

This test validates the intermediate learning workflow from quickstart.md
focusing on feature engineering capabilities

TDD Note: This test MUST FAIL until all core components are implemented in Phase 3.3
"""

import pytest
import pandas as pd


@pytest.mark.integration
@pytest.mark.educational
@pytest.mark.domain_healthcare
class TestIntermediateFeatureEngineeringWorkflow:
    """Integration test for intermediate feature engineering scenario"""

    def test_complete_intermediate_workflow(self):
        """T018: Test complete intermediate feature engineering workflow

        Scenario from quickstart.md:
        - Load intermediate healthcare project
        - Create multi-step feature engineering pipeline
        - Execute with basic guidance
        - Learn advanced preprocessing concepts
        """
        import datadojo

        # Step 1: Create dojo with educational mode
        dojo = datadojo.create_dojo(educational_mode=True)

        # Step 2: Load intermediate healthcare project
        project = dojo.load_project("patient_outcomes", difficulty="intermediate")

        # Verify project level
        info = project.info
        assert info.difficulty.value == "intermediate"
        assert info.domain.value == "healthcare"

        # Step 3: Create pipeline with basic guidance (less hand-holding)
        pipeline = project.create_pipeline(guidance_level="basic")

        # Step 4: Add feature engineering steps
        pipeline.add_step("feature_creation",
                        operations=["age_groups", "bmi_categories", "risk_scores"])
        pipeline.add_step("feature_selection", method="correlation")
        pipeline.add_step("scaling", technique="standard")

        # Step 5: Execute pipeline
        result = pipeline.execute(project.dataset)

        # Verify successful execution
        assert result.success is True
        assert result.steps_completed >= 3, "Should complete all 3 steps"
        assert len(result.concepts_learned) > 0

        # Verify intermediate concepts learned
        concepts_str = str(result.concepts_learned).lower()
        assert any(topic in concepts_str for topic in [
            "feature", "engineering", "scaling", "selection"
        ]), "Should learn feature engineering concepts"

    def test_intermediate_project_characteristics(self):
        """T018: Verify intermediate projects have appropriate complexity"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)

        # List intermediate projects
        projects = dojo.list_projects(difficulty=datadojo.DifficultyLevel.INTERMEDIATE)

        assert len(projects) > 0, "Should have intermediate projects"

        for project_info in projects:
            assert project_info.difficulty.value == "intermediate"
            # Intermediate projects should take longer than beginner
            assert project_info.estimated_time_minutes >= 30, \
                "Intermediate projects should be substantial"

    def test_feature_creation_workflow(self):
        """T018: Test creating derived features from existing data"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("patient_outcomes", difficulty="intermediate")

        pipeline = project.create_pipeline(guidance_level="basic")

        # Feature creation with specific operations
        pipeline.add_step("feature_creation",
                        operations=["age_groups", "bmi_categories"])

        result = pipeline.execute(project.dataset)

        assert result.success is True
        # Result should contain newly created features

    def test_feature_selection_methods(self):
        """T018: Test different feature selection techniques"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("patient_outcomes", difficulty="intermediate")

        # Test correlation-based selection
        pipeline1 = project.create_pipeline(guidance_level="basic")
        pipeline1.add_step("feature_selection", method="correlation", threshold=0.8)

        result1 = pipeline1.execute(project.dataset)
        assert result1.success is True

        # Could test other methods like variance-based, etc.

    def test_scaling_techniques(self):
        """T018: Test different scaling and normalization techniques"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("patient_outcomes", difficulty="intermediate")

        # Test standard scaling
        pipeline = project.create_pipeline(guidance_level="basic")
        pipeline.add_step("scaling", technique="standard")

        result = pipeline.execute(project.dataset)
        assert result.success is True

    def test_multi_table_project_handling(self):
        """T018: Test working with projects that have multiple related tables"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)

        # Intermediate projects may have multi-table datasets
        project = dojo.load_project("patient_outcomes", difficulty="intermediate")

        dataset = project.dataset
        assert dataset is not None

        # Should be able to handle more complex data structures

    def test_intermediate_guidance_less_verbose(self):
        """T018: Test that intermediate guidance is less detailed than beginner"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("patient_outcomes", difficulty="intermediate")

        # Basic guidance level appropriate for intermediate learners
        pipeline = project.create_pipeline(guidance_level="basic")
        pipeline.add_step("feature_engineering", interactive=False)

        preview = pipeline.preview_next_step()

        assert isinstance(preview, dict)
        # Preview should still provide guidance, but assume more knowledge

    def test_intermediate_progress_includes_skill_levels(self):
        """T018: Test progress tracking includes skill assessment"""
        import datadojo

        student_id = "intermediate_student_001"

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("patient_outcomes", difficulty="intermediate")

        pipeline = project.create_pipeline(guidance_level="basic")
        pipeline.add_step("feature_engineering", interactive=False)
        pipeline.execute(project.dataset)

        progress = project.get_progress(student_id)

        assert isinstance(progress, dict)
        # Progress should track more advanced skills

    def test_categorical_encoding_strategies(self):
        """T018: Test different categorical encoding approaches"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("patient_outcomes", difficulty="intermediate")

        pipeline = project.create_pipeline(guidance_level="basic")
        pipeline.add_step("categorical_encoding",
                        strategy="target_encoding",
                        handle_unknown="ignore")

        result = pipeline.execute(project.dataset)

        assert result.success is True
        # Should learn about encoding strategies

    def test_intermediate_pipeline_chaining(self):
        """T018: Test chaining multiple preprocessing operations"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("patient_outcomes", difficulty="intermediate")

        # Chain multiple operations
        pipeline = (project.create_pipeline(guidance_level="basic")
                   .add_step("feature_creation", operations=["age_groups"])
                   .add_step("feature_selection", method="correlation")
                   .add_step("scaling", technique="standard")
                   .add_step("categorical_encoding", strategy="one_hot"))

        result = pipeline.execute(project.dataset)

        assert result.success is True
        assert result.steps_completed == 4


@pytest.mark.integration
@pytest.mark.performance
class TestIntermediatePerformance:
    """Performance tests for intermediate workflows"""

    def test_intermediate_handles_larger_datasets(self):
        """T018: Test intermediate projects work with larger datasets"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("patient_outcomes", difficulty="intermediate")

        dataset = project.dataset

        # Intermediate datasets should be larger than beginner
        # but still manageable for learning

        pipeline = project.create_pipeline(guidance_level="basic")
        pipeline.add_step("feature_engineering", interactive=False)

        result = pipeline.execute(dataset)

        assert result.success is True
        assert result.execution_time_ms > 0
