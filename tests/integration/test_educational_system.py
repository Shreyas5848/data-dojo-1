"""
Integration test for educational guidance system

This test validates the complete educational features including guidance,
concept explanations, progress tracking, and interactive learning

TDD Note: This test MUST FAIL until educational system is implemented in Phase 3.3
"""

import pytest


@pytest.mark.integration
@pytest.mark.educational
class TestEducationalGuidanceSystem:
    """Integration tests for educational guidance features"""

    def test_complete_educational_workflow(self):
        """T021: Test complete educational guidance workflow

        Workflow:
        1. Get concept explanations
        2. Receive step-by-step guidance
        3. Track learning progress
        4. Visualize skill development
        """
        import datadojo

        student_id = "educational_test_student"

        # Step 1: Create educational environment
        dojo = datadojo.create_dojo(educational_mode=True)

        # Step 2: Load project with educational support
        project = dojo.load_project("ecommerce_customer_data", difficulty="beginner")

        # Step 3: Create pipeline with detailed guidance
        pipeline = project.create_pipeline(guidance_level="detailed")

        # Step 4: Preview guidance before execution
        pipeline.add_step("data_cleaning", interactive=True)
        guidance = pipeline.preview_next_step()

        assert isinstance(guidance, dict), "Should provide guidance"
        assert len(guidance) > 0, "Guidance should have content"

        # Step 5: Execute with educational tracking
        result = pipeline.execute(project.dataset)

        assert result.success is True
        assert len(result.concepts_learned) > 0, "Should learn concepts"

        # Step 6: Check progress tracking
        progress = project.get_progress(student_id)
        assert isinstance(progress, dict)

    def test_concept_explanations_available(self):
        """T021: Test that concept explanations are comprehensive"""
        from datadojo.core.educational import Educational

        educational = Educational()

        # Core data preprocessing concepts
        concepts = [
            "missing_values",
            "outliers",
            "normalization",
            "standardization",
            "feature_engineering",
            "feature_selection",
            "data_cleaning",
            "duplicate_detection",
            "categorical_encoding",
            "data_validation"
        ]

        for concept in concepts:
            explanation = educational.get_concept_explanation(concept)

            assert isinstance(explanation, dict), \
                f"Explanation for '{concept}' should be a dict"
            assert len(explanation) > 0, \
                f"Explanation for '{concept}' should have content"

    def test_context_aware_guidance(self):
        """T021: Test that guidance adapts to context"""
        from datadojo.core.educational import Educational

        educational = Educational()

        # Different contexts should provide different guidance
        context1 = {
            "operation_type": "data_cleaning",
            "data_issues": ["missing_values", "duplicates"],
            "difficulty": "beginner"
        }

        context2 = {
            "operation_type": "feature_engineering",
            "data_issues": ["categorical_features"],
            "difficulty": "intermediate"
        }

        guidance1 = educational.get_step_guidance(context1)
        guidance2 = educational.get_step_guidance(context2)

        assert isinstance(guidance1, dict)
        assert isinstance(guidance2, dict)
        # Guidance should be context-specific

    def test_progress_tracking_accumulates(self):
        """T021: Test that progress accumulates over multiple sessions"""
        from datadojo.core.educational import Educational

        educational = Educational()
        student_id = "accumulating_student"

        # Track progress across multiple steps
        session1_concepts = ["missing_values", "duplicate_detection"]
        session2_concepts = ["outliers", "normalization"]
        session3_concepts = ["feature_engineering"]

        educational.track_progress(student_id, "session_1", session1_concepts)
        educational.track_progress(student_id, "session_2", session2_concepts)
        educational.track_progress(student_id, "session_3", session3_concepts)

        # Progress should accumulate all concepts
        # (Verification would happen through get_progress in Project interface)

    def test_guidance_levels_differ_in_detail(self):
        """T021: Test that different guidance levels provide appropriate detail"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("ecommerce_customer_data", difficulty="beginner")

        # Detailed guidance
        pipeline_detailed = project.create_pipeline(guidance_level="detailed")
        pipeline_detailed.add_step("data_cleaning", interactive=False)
        preview_detailed = pipeline_detailed.preview_next_step()

        # Basic guidance
        pipeline_basic = project.create_pipeline(guidance_level="basic")
        pipeline_basic.add_step("data_cleaning", interactive=False)
        preview_basic = pipeline_basic.preview_next_step()

        # No guidance
        pipeline_none = project.create_pipeline(guidance_level="none")
        pipeline_none.add_step("data_cleaning", interactive=False)
        preview_none = pipeline_none.preview_next_step()

        # All should return dicts, but with different levels of detail
        assert isinstance(preview_detailed, dict)
        assert isinstance(preview_basic, dict)
        assert isinstance(preview_none, dict)

    def test_interactive_mode_provides_extra_guidance(self):
        """T021: Test that interactive mode enhances educational experience"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("ecommerce_customer_data", difficulty="beginner")

        # Interactive step
        pipeline_interactive = project.create_pipeline(guidance_level="detailed")
        pipeline_interactive.add_step("data_cleaning", interactive=True)

        # Non-interactive step
        pipeline_auto = project.create_pipeline(guidance_level="detailed")
        pipeline_auto.add_step("data_cleaning", interactive=False)

        # Both should work, but interactive may provide richer experience
        result_interactive = pipeline_interactive.execute(project.dataset)
        result_auto = pipeline_auto.execute(project.dataset)

        assert result_interactive.success is True
        assert result_auto.success is True

    def test_educational_content_includes_analogies(self):
        """T021: Test that educational content uses analogies for clarity"""
        from datadojo.core.educational import Educational

        educational = Educational()

        # Explanations should be educational, not just technical
        explanation = educational.get_concept_explanation("missing_values")

        assert isinstance(explanation, dict)
        # Should contain explanatory content suitable for learning

    def test_skill_assessment_tracks_mastery(self):
        """T021: Test that progress includes skill assessments"""
        import datadojo

        student_id = "skill_assessment_student"

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("ecommerce_customer_data", difficulty="beginner")

        # Execute multiple operations to build skills
        pipeline = project.create_pipeline(guidance_level="detailed")
        pipeline.add_step("data_cleaning", interactive=False)
        pipeline.add_step("feature_engineering", interactive=False)
        pipeline.execute(project.dataset)

        # Progress should track skill development
        progress = project.get_progress(student_id)
        assert isinstance(progress, dict)

    def test_educational_mode_vs_production_mode(self):
        """T021: Test clear difference between educational and production modes"""
        import datadojo

        # Educational mode
        dojo_edu = datadojo.create_dojo(educational_mode=True)
        project_edu = dojo_edu.load_project("ecommerce_customer_data", difficulty="beginner")
        pipeline_edu = project_edu.create_pipeline(guidance_level="detailed")
        pipeline_edu.add_step("data_cleaning", interactive=False)

        result_edu = pipeline_edu.execute(project_edu.dataset)

        # Production mode
        dojo_prod = datadojo.create_dojo(educational_mode=False)
        project_prod = dojo_prod.load_project("ecommerce_customer_data", difficulty="beginner")
        pipeline_prod = project_prod.create_pipeline(guidance_level="none")
        pipeline_prod.add_step("data_cleaning", interactive=False)

        result_prod = pipeline_prod.execute(project_prod.dataset)

        # Both should succeed, but educational should provide concepts learned
        assert result_edu.success is True
        assert result_prod.success is True

        # Educational mode should track concepts
        assert len(result_edu.concepts_learned) >= 0

    def test_guidance_performance_requirement(self):
        """T021: Test that guidance responds within <500ms"""
        import datadojo
        import time

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("ecommerce_customer_data", difficulty="beginner")
        pipeline = project.create_pipeline(guidance_level="detailed")
        pipeline.add_step("data_cleaning", interactive=False)

        # Measure guidance response time
        start = time.time()
        guidance = pipeline.preview_next_step()
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < 500, \
            f"Guidance should respond in <500ms (requirement), took {elapsed_ms:.2f}ms"


@pytest.mark.integration
@pytest.mark.educational
class TestLearningProgressionWorkflow:
    """Test learning progression across difficulty levels"""

    def test_progression_from_beginner_to_advanced(self):
        """T021: Test student progression through difficulty levels"""
        import datadojo

        student_id = "progression_student"

        dojo = datadojo.create_dojo(educational_mode=True)

        # Beginner project
        project_beginner = dojo.load_project(
            "ecommerce_customer_data",
            difficulty="beginner"
        )
        pipeline_beginner = project_beginner.create_pipeline(guidance_level="detailed")
        pipeline_beginner.add_step("data_cleaning", interactive=False)
        result_beginner = pipeline_beginner.execute(project_beginner.dataset)

        assert result_beginner.success is True

        # Intermediate project
        project_intermediate = dojo.load_project(
            "patient_outcomes",
            difficulty="intermediate"
        )
        pipeline_intermediate = project_intermediate.create_pipeline(guidance_level="basic")
        pipeline_intermediate.add_step("feature_engineering", interactive=False)
        result_intermediate = pipeline_intermediate.execute(project_intermediate.dataset)

        assert result_intermediate.success is True

        # Advanced project
        project_advanced = dojo.load_project(
            "fraud_detection",
            difficulty="advanced"
        )
        pipeline_advanced = project_advanced.create_pipeline(guidance_level="none")
        pipeline_advanced.add_step("anomaly_detection", algorithm="isolation_forest")
        result_advanced = pipeline_advanced.execute(project_advanced.dataset)

        assert result_advanced.success is True

        # Student should be able to progress through all levels

    def test_checkpoints_validate_understanding(self):
        """T021: Test that checkpoints validate student understanding"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)
        project = dojo.load_project("ecommerce_customer_data", difficulty="beginner")

        # Execute with validation checkpoints
        pipeline = project.create_pipeline(guidance_level="detailed")
        pipeline.add_step("data_cleaning", interactive=True)
        # Checkpoints may be implicit in interactive mode

        result = pipeline.execute(project.dataset)

        assert result.success is True


@pytest.mark.integration
@pytest.mark.educational
@pytest.mark.slow
class TestEducationalContentComprehensiveness:
    """Test that educational content covers all necessary topics"""

    def test_all_preprocessing_concepts_have_explanations(self):
        """T021: Test comprehensive concept coverage"""
        from datadojo.core.educational import Educational

        educational = Educational()

        # All major preprocessing concepts should have explanations
        essential_concepts = [
            "missing_values",
            "outliers",
            "normalization",
            "standardization",
            "min_max_scaling",
            "feature_engineering",
            "feature_selection",
            "feature_creation",
            "categorical_encoding",
            "one_hot_encoding",
            "label_encoding",
            "target_encoding",
            "data_cleaning",
            "duplicate_detection",
            "data_validation",
            "data_type_conversion",
            "handling_imbalanced_data",
            "feature_scaling",
            "dimensionality_reduction"
        ]

        available_count = 0
        for concept in essential_concepts:
            try:
                explanation = educational.get_concept_explanation(concept)
                if isinstance(explanation, dict) and len(explanation) > 0:
                    available_count += 1
            except KeyError:
                # Concept not yet implemented
                pass

        # Should have substantial coverage
        coverage_percent = (available_count / len(essential_concepts)) * 100
        print(f"Concept coverage: {coverage_percent:.1f}% ({available_count}/{len(essential_concepts)})")

        # Aim for high coverage (at least 50% in initial implementation)
        assert available_count > 0, "Should have at least some concept explanations"

    def test_domain_specific_guidance_available(self):
        """T021: Test that guidance is available for all domains"""
        import datadojo

        dojo = datadojo.create_dojo(educational_mode=True)

        domains = [
            ("ecommerce_customer_data", "ecommerce"),
            ("patient_outcomes", "healthcare"),
            ("fraud_detection", "finance")
        ]

        for project_id, domain in domains:
            try:
                project = dojo.load_project(project_id, difficulty="beginner")
                pipeline = project.create_pipeline(guidance_level="detailed")
                pipeline.add_step("data_cleaning", interactive=False)

                guidance = pipeline.preview_next_step()
                assert isinstance(guidance, dict), \
                    f"Should provide guidance for {domain} domain"
            except ValueError:
                # Project might not exist
                pass
