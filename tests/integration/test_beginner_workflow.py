"""
Integration test for beginner e-commerce project workflow

This test validates the complete learning workflow from the beginner's perspective
following the quickstart scenario from specs/001-use-the-requirements/quickstart.md

TDD Note: This test MUST FAIL until all core components are implemented in Phase 3.3
"""

import pytest
import pandas as pd


@pytest.mark.integration
@pytest.mark.educational
@pytest.mark.domain_ecommerce
class TestBeginnerEcommerceWorkflow:
    """Integration test for beginner e-commerce learning scenario"""

    def test_complete_beginner_workflow(self):
        """T017: Test complete beginner e-commerce data cleaning workflow

        Scenario from quickstart.md:
        - Load beginner e-commerce project
        - Create pipeline with detailed guidance
        - Add data cleaning steps with interactive guidance
        - Execute and learn concepts
        """
        import datadojo

        # Step 1: Create educational environment
        dojo = datadojo.create_dojo(educational_mode=True)
        assert dojo is not None, "Should create Dojo instance"

        # Step 2: Load beginner e-commerce project
        project = dojo.load_project("ecommerce_customer_data", difficulty="beginner")
        assert project is not None, "Should load project"

        # Verify project info
        info = project.info
        assert info.difficulty.value == "beginner"
        assert info.domain.value == "ecommerce"
        assert len(info.learning_objectives) > 0

        # Step 3: Verify dataset is available and has quality issues
        dataset = project.dataset
        assert dataset is not None, "Project should have a dataset"

        # Step 4: Create pipeline with detailed guidance
        pipeline = project.create_pipeline(guidance_level="detailed")
        assert pipeline is not None, "Should create pipeline"

        # Step 5: Add interactive data cleaning steps
        pipeline.add_step("data_cleaning", interactive=True)

        # Step 6: Execute pipeline and learn concepts
        result = pipeline.execute(dataset)

        # Verify execution success
        assert result.success is True, "Pipeline execution should succeed"
        assert result.processed_data is not None, "Should return processed data"
        assert result.steps_completed > 0, "Should complete at least one step"
        assert len(result.concepts_learned) > 0, "Should learn at least one concept"

        # Verify concepts learned include beginner topics
        concepts_str = str(result.concepts_learned).lower()
        assert any(topic in concepts_str for topic in [
            "missing", "duplicate", "data quality", "cleaning"
        ]), "Should learn beginner data cleaning concepts"

    def test_beginner_project_characteristics(self):
        """T017: Verify beginner project has appropriate characteristics"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)

        # List beginner e-commerce projects
        projects = dojo.list_projects(
            domain=datadojo.Domain.ECOMMERCE,
            difficulty=datadojo.DifficultyLevel.BEGINNER
        )

        assert len(projects) > 0, "Should have beginner e-commerce projects"

        # Verify project characteristics
        for project_info in projects:
            assert project_info.difficulty.value == "beginner"
            assert project_info.domain.value == "ecommerce"
            assert project_info.estimated_time_minutes > 0
            assert project_info.estimated_time_minutes < 120, \
                "Beginner projects should be short (< 2 hours)"

    def test_beginner_workflow_with_step_by_step_guidance(self):
        """T017: Test that detailed guidance is provided at each step"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("ecommerce_customer_data", difficulty="beginner")
        pipeline = project.create_pipeline(guidance_level="detailed")

        # Add step and preview guidance
        pipeline.add_step("data_cleaning", interactive=True)

        # Preview next step should provide guidance
        preview = pipeline.preview_next_step()
        assert isinstance(preview, dict), "Preview should return guidance information"
        assert len(preview) > 0, "Should provide guidance content"

    def test_beginner_missing_values_handling(self):
        """T017: Test handling missing values - core beginner skill"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("ecommerce_customer_data", difficulty="beginner")

        # Create dataset with missing values
        dataset = project.dataset

        # Create pipeline with missing value handling
        pipeline = project.create_pipeline(guidance_level="detailed")
        pipeline.add_step("missing_values", strategy="interactive", interactive=True)

        result = pipeline.execute(dataset)

        assert result.success is True
        # Should learn about missing value strategies
        assert any("missing" in concept.lower() for concept in result.concepts_learned)

    def test_beginner_duplicate_detection(self):
        """T017: Test duplicate detection - core beginner skill"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("ecommerce_customer_data", difficulty="beginner")

        pipeline = project.create_pipeline(guidance_level="detailed")
        pipeline.add_step("duplicate_detection", interactive=True)

        result = pipeline.execute(project.dataset)

        assert result.success is True
        # Should learn about duplicate handling

    def test_beginner_progress_tracking(self):
        """T017: Test that student progress is tracked during beginner workflow"""
        import datadojo

        student_id = "beginner_student_001"

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("ecommerce_customer_data", difficulty="beginner")

        # Execute some steps
        pipeline = project.create_pipeline(guidance_level="detailed")
        pipeline.add_step("data_cleaning", interactive=False)
        pipeline.execute(project.dataset)

        # Check progress
        progress = project.get_progress(student_id)
        assert isinstance(progress, dict), "Should return progress information"

    def test_beginner_workflow_performance(self):
        """T017: Test that beginner guidance responds quickly (<500ms per guidance)"""
        import datadojo
        import time

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("ecommerce_customer_data", difficulty="beginner")
        pipeline = project.create_pipeline(guidance_level="detailed")

        # Add step and measure preview performance
        pipeline.add_step("data_cleaning", interactive=False)

        start_time = time.time()
        preview = pipeline.preview_next_step()
        elapsed_ms = (time.time() - start_time) * 1000

        assert elapsed_ms < 500, \
            f"Guidance should respond in <500ms, took {elapsed_ms:.2f}ms"

    def test_beginner_can_switch_to_production_mode(self):
        """T017: Test that beginner can switch to production mode after learning"""
        import datadojo

        # Start in educational mode
        dojo_edu = datadojo.create_dojo(educational_mode=True)
        project_edu = dojo_edu.load_project("ecommerce_customer_data", difficulty="beginner")
        pipeline_edu = project_edu.create_pipeline(guidance_level="detailed")

        # Switch to production mode (no guidance)
        dojo_prod = datadojo.create_dojo(educational_mode=False)
        project_prod = dojo_prod.load_project("ecommerce_customer_data", difficulty="beginner")
        pipeline_prod = project_prod.create_pipeline(guidance_level="none")

        # Both should work
        assert pipeline_edu is not None
        assert pipeline_prod is not None
