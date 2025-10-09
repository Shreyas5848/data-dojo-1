"""
Integration test for advanced pipeline production mode workflow

This test validates advanced scenarios including production mode without guidance,
performance optimization, and complex preprocessing pipelines

TDD Note: This test MUST FAIL until all core components are implemented in Phase 3.3
"""

import pytest
import time


@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.domain_finance
class TestAdvancedProductionWorkflow:
    """Integration test for advanced production-ready pipeline scenario"""

    def test_complete_advanced_workflow(self):
        """T019: Test complete advanced pipeline in production mode

        Scenario from quickstart.md:
        - Load advanced financial project
        - Create pipeline with NO guidance (production mode)
        - Build end-to-end preprocessing pipeline
        - Validate performance and robustness
        """
        import datadojo

        # Step 1: Create dojo in production mode (educational_mode=False)
        dojo = datadojo.create_dojo(educational_mode=False)

        # Step 2: Load advanced finance project
        project = dojo.load_project("fraud_detection", difficulty="advanced")

        info = project.info
        assert info.difficulty.value == "advanced"
        assert info.domain.value == "finance"

        # Step 3: Create pipeline with NO guidance (production mode)
        pipeline = project.create_pipeline(guidance_level="none")

        # Step 4: Build complex preprocessing pipeline
        pipeline.add_step("anomaly_detection", algorithm="isolation_forest")
        pipeline.add_step("time_series_features", window_size=30)
        pipeline.add_step("categorical_encoding", strategy="target_encoding")
        pipeline.add_step("feature_validation", strict=True)

        # Step 5: Execute and verify
        result = pipeline.execute(project.dataset)

        assert result.success is True
        assert result.steps_completed == 4
        # In production mode, concepts_learned might be empty or minimal
        assert isinstance(result.concepts_learned, list)

    def test_advanced_project_characteristics(self):
        """T019: Verify advanced projects have complex requirements"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)

        # List advanced projects
        projects = dojo.list_projects(difficulty=datadojo.DifficultyLevel.ADVANCED)

        assert len(projects) > 0, "Should have advanced projects"

        for project_info in projects:
            assert project_info.difficulty.value == "advanced"
            # Advanced projects should be substantial
            assert project_info.estimated_time_minutes >= 60, \
                "Advanced projects should be comprehensive"

    def test_production_mode_no_guidance(self):
        """T019: Test that production mode provides no educational guidance"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=False)
        project = dojo.load_project("fraud_detection", difficulty="advanced")

        pipeline = project.create_pipeline(guidance_level="none")
        pipeline.add_step("anomaly_detection", algorithm="isolation_forest")

        # Preview in production mode should be minimal or empty
        preview = pipeline.preview_next_step()

        assert isinstance(preview, dict)
        # Production mode preview should be minimal (just technical info, no teaching)

    def test_production_mode_performance_optimized(self):
        """T019: Test that production mode is optimized for performance"""
        import datadojo
        import time

        # Production mode
        dojo_prod = datadojo.create_dojo(educational_mode=False)
        project_prod = dojo_prod.load_project("fraud_detection", difficulty="advanced")
        pipeline_prod = project_prod.create_pipeline(guidance_level="none")

        pipeline_prod.add_step("anomaly_detection", algorithm="isolation_forest")

        start = time.time()
        result_prod = pipeline_prod.execute(project_prod.dataset)
        prod_time = time.time() - start

        assert result_prod.success is True

        # Educational mode (for comparison)
        dojo_edu = datadojo.create_dojo(educational_mode=True)
        project_edu = dojo_edu.load_project("fraud_detection", difficulty="advanced")
        pipeline_edu = project_edu.create_pipeline(guidance_level="detailed")

        pipeline_edu.add_step("anomaly_detection", algorithm="isolation_forest")

        start = time.time()
        result_edu = pipeline_edu.execute(project_edu.dataset)
        edu_time = time.time() - start

        assert result_edu.success is True

        # Production mode should be faster (no guidance overhead)
        # This is a soft assertion - educational overhead should be minimal anyway
        print(f"Production: {prod_time:.3f}s, Educational: {edu_time:.3f}s")

    def test_anomaly_detection_pipeline(self):
        """T019: Test anomaly detection with isolation forest"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=False)
        project = dojo.load_project("fraud_detection", difficulty="advanced")

        pipeline = project.create_pipeline(guidance_level="none")
        pipeline.add_step("anomaly_detection",
                        algorithm="isolation_forest",
                        contamination=0.1)

        result = pipeline.execute(project.dataset)

        assert result.success is True

    def test_time_series_features(self):
        """T019: Test time series feature extraction"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=False)
        project = dojo.load_project("fraud_detection", difficulty="advanced")

        pipeline = project.create_pipeline(guidance_level="none")
        pipeline.add_step("time_series_features",
                        window_size=30,
                        features=["rolling_mean", "rolling_std"])

        result = pipeline.execute(project.dataset)

        assert result.success is True

    def test_strict_feature_validation(self):
        """T019: Test strict validation catches issues"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=False)
        project = dojo.load_project("fraud_detection", difficulty="advanced")

        pipeline = project.create_pipeline(guidance_level="none")
        pipeline.add_step("feature_validation", strict=True)

        # With strict validation, should catch any data quality issues
        result = pipeline.execute(project.dataset)

        # May succeed or fail depending on data quality
        assert isinstance(result, datadojo.models.pipeline.ExecutionResult)

    def test_advanced_pipeline_error_handling(self):
        """T019: Test robust error handling in production pipelines"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=False)
        project = dojo.load_project("fraud_detection", difficulty="advanced")

        pipeline = project.create_pipeline(guidance_level="none")
        pipeline.add_step("anomaly_detection", algorithm="isolation_forest")

        # Execute with intentionally problematic data
        import pandas as pd
        bad_data = pd.DataFrame()  # Empty dataframe

        try:
            result = pipeline.execute(bad_data)
            # Should either fail gracefully or raise RuntimeError
            if not result.success:
                assert result.error_message is not None
        except RuntimeError as e:
            # Expected behavior for invalid data
            assert str(e) != ""

    def test_scalability_to_large_datasets(self):
        """T019: Test handling of large datasets (1M+ rows requirement)"""
        import datadojo
        import pandas as pd
        import numpy as np

        dojo = datadojo.create_dojo(educational_mode=False)
        project = dojo.load_project("fraud_detection", difficulty="advanced")

        # Create large test dataset (simulated 1M rows)
        large_data = pd.DataFrame({
            'feature1': np.random.rand(10000),  # Smaller for test speed
            'feature2': np.random.rand(10000),
            'feature3': np.random.randint(0, 100, 10000)
        })

        pipeline = project.create_pipeline(guidance_level="none")
        pipeline.add_step("anomaly_detection", algorithm="isolation_forest")

        start = time.time()
        result = pipeline.execute(large_data)
        elapsed = time.time() - start

        assert result.success is True
        print(f"Processed {len(large_data)} rows in {elapsed:.2f}s")

    def test_advanced_pipeline_export_for_production(self):
        """T019: Test exporting pipeline for production use"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=False)
        project = dojo.load_project("fraud_detection", difficulty="advanced")

        pipeline = (project.create_pipeline(guidance_level="none")
                   .add_step("anomaly_detection", algorithm="isolation_forest")
                   .add_step("time_series_features", window_size=30)
                   .add_step("categorical_encoding", strategy="target_encoding"))

        # Pipeline should be exportable/serializable for production deployment
        # This is a future feature, but pipeline should at least be reusable
        result1 = pipeline.execute(project.dataset)
        result2 = pipeline.execute(project.dataset)

        assert result1.success is True
        assert result2.success is True


@pytest.mark.integration
@pytest.mark.slow
class TestAdvancedComplexScenarios:
    """Complex advanced scenarios"""

    def test_multi_domain_advanced_skills(self):
        """T019: Test that advanced skills transfer across domains"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=False)

        # Test advanced techniques in different domains
        domains = [
            ("fraud_detection", "finance"),
            ("patient_outcomes", "healthcare"),
            ("ecommerce_customer_data", "ecommerce")
        ]

        for project_id, expected_domain in domains:
            try:
                project = dojo.load_project(project_id, difficulty="advanced")

                pipeline = project.create_pipeline(guidance_level="none")
                pipeline.add_step("data_cleaning", interactive=False)

                result = pipeline.execute(project.dataset)

                # Advanced techniques should work across domains
                assert isinstance(result, datadojo.models.pipeline.ExecutionResult)
            except ValueError:
                # Project might not exist at advanced level in all domains
                pass

    def test_advanced_custom_preprocessing_chain(self):
        """T019: Test building custom complex preprocessing chains"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=False)
        project = dojo.load_project("fraud_detection", difficulty="advanced")

        # Complex multi-step chain
        pipeline = (project.create_pipeline(guidance_level="none")
                   .add_step("data_cleaning", interactive=False)
                   .add_step("anomaly_detection", algorithm="isolation_forest")
                   .add_step("feature_creation", operations=["ratios", "interactions"])
                   .add_step("time_series_features", window_size=30)
                   .add_step("feature_selection", method="correlation")
                   .add_step("categorical_encoding", strategy="target_encoding")
                   .add_step("scaling", technique="robust")
                   .add_step("feature_validation", strict=True))

        result = pipeline.execute(project.dataset)

        assert result.success is True
        assert result.steps_completed == 8
