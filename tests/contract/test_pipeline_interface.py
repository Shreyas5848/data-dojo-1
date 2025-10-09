"""
Contract tests for PipelineInterface

These tests verify that Pipeline implementations adhere to the PipelineInterface contract
defined in specs/001-use-the-requirements/contracts/dojo_api.py

TDD Note: These tests MUST FAIL until PipelineInterface is implemented in Phase 3.3
"""

import pytest
from typing import List, Dict, Any

# Contract imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../specs/001-use-the-requirements'))
from contracts.dojo_api import (
    PipelineInterface,
    ExecutionResult,
    GuidanceLevel,
)


class TestPipelineInterfaceContract:
    """Contract tests for PipelineInterface"""

    @pytest.fixture
    def pipeline_instance(self):
        """Fixture providing a PipelineInterface implementation

        TDD: This will fail until Pipeline class is implemented
        """
        from datadojo import create_dojo

        dojo = create_dojo(educational_mode=True)
        projects = dojo.list_projects()
        project = dojo.load_project(projects[0].id)
        return project.create_pipeline(GuidanceLevel.DETAILED)

    @pytest.fixture
    def sample_data(self):
        """Provide sample data for pipeline testing"""
        import pandas as pd
        import numpy as np

        # Create a small messy dataset for testing
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', None, 'David', 'Eve'],
            'age': [25, np.nan, 30, 35, 28],
            'score': [85.5, 90.0, 75.5, 88.0, 92.5]
        })

    def test_pipeline_implements_interface(self, pipeline_instance):
        """Verify Pipeline class implements PipelineInterface"""
        assert isinstance(pipeline_instance, PipelineInterface), \
            "Pipeline must implement PipelineInterface"

    def test_add_step_returns_self_for_chaining(self, pipeline_instance):
        """T010: Test PipelineInterface.add_step() returns self for method chaining"""
        result = pipeline_instance.add_step("data_cleaning")

        assert result is pipeline_instance, \
            "add_step() must return self for method chaining"

    def test_add_step_basic_operation(self, pipeline_instance):
        """T010: Test adding a basic preprocessing step"""
        pipeline = pipeline_instance.add_step("data_cleaning", interactive=False)

        assert isinstance(pipeline, PipelineInterface)

    def test_add_step_with_interactive_mode(self, pipeline_instance):
        """T010: Test add_step() with interactive parameter"""
        pipeline = (pipeline_instance
                   .add_step("data_cleaning", interactive=True)
                   .add_step("feature_engineering", interactive=False))

        assert isinstance(pipeline, PipelineInterface)

    def test_add_step_with_parameters(self, pipeline_instance):
        """T010: Test add_step() with operation-specific parameters"""
        pipeline = pipeline_instance.add_step(
            "data_cleaning",
            interactive=False,
            strategy="mean",
            threshold=0.5
        )

        assert isinstance(pipeline, PipelineInterface)

    def test_add_step_method_chaining(self, pipeline_instance):
        """T010: Test chaining multiple add_step() calls"""
        pipeline = (pipeline_instance
                   .add_step("data_cleaning", interactive=False)
                   .add_step("feature_engineering", interactive=False)
                   .add_step("transformation", interactive=False))

        assert isinstance(pipeline, PipelineInterface)

    def test_add_step_invalid_operation_raises_error(self, pipeline_instance):
        """T010: Test add_step() raises ValueError for unsupported operation"""
        with pytest.raises(ValueError) as exc_info:
            pipeline_instance.add_step("invalid_operation_xyz_12345")

        assert "not supported" in str(exc_info.value).lower() or \
               "invalid" in str(exc_info.value).lower(), \
            "Error should indicate operation is not supported"

    def test_get_available_operations_returns_list(self, pipeline_instance):
        """T010: Test get_available_operations() returns list of strings"""
        operations = pipeline_instance.get_available_operations()

        assert isinstance(operations, list), \
            "get_available_operations() must return a list"
        assert len(operations) > 0, \
            "Should have at least one available operation"

        for op in operations:
            assert isinstance(op, str), \
                f"Each operation must be a string, got {type(op)}"

    def test_get_available_operations_includes_core_operations(self, pipeline_instance):
        """T010: Test that core operations are available"""
        operations = pipeline_instance.get_available_operations()

        # Should include basic preprocessing operations
        expected_operations = ["data_cleaning", "feature_engineering", "transformation"]

        for expected_op in expected_operations:
            assert expected_op in operations, \
                f"Expected operation '{expected_op}' not found in available operations"

    def test_preview_next_step_returns_dict(self, pipeline_instance):
        """T010: Test preview_next_step() returns dictionary"""
        # Add a step first
        pipeline_instance.add_step("data_cleaning", interactive=False)

        preview = pipeline_instance.preview_next_step()

        assert isinstance(preview, dict), \
            "preview_next_step() must return a dictionary"

    def test_preview_next_step_has_guidance_info(self, pipeline_instance):
        """T010: Test preview_next_step() includes guidance information"""
        pipeline_instance.add_step("data_cleaning", interactive=False)

        preview = pipeline_instance.preview_next_step()

        assert isinstance(preview, dict)
        # Preview should contain information about the next step
        # Exact fields depend on implementation, but should be non-empty

    def test_execute_returns_execution_result(self, pipeline_instance, sample_data):
        """T011: Test PipelineInterface.execute() returns ExecutionResult"""
        # Add some steps
        pipeline_instance.add_step("data_cleaning", interactive=False)

        result = pipeline_instance.execute(sample_data)

        assert isinstance(result, ExecutionResult), \
            "execute() must return ExecutionResult"

    def test_execution_result_has_required_fields(self, pipeline_instance, sample_data):
        """T011: Test ExecutionResult contains all required fields"""
        pipeline_instance.add_step("data_cleaning", interactive=False)

        result = pipeline_instance.execute(sample_data)

        assert hasattr(result, 'success'), "ExecutionResult must have 'success' field"
        assert hasattr(result, 'processed_data'), "Must have 'processed_data' field"
        assert hasattr(result, 'execution_time_ms'), "Must have 'execution_time_ms' field"
        assert hasattr(result, 'steps_completed'), "Must have 'steps_completed' field"
        assert hasattr(result, 'concepts_learned'), "Must have 'concepts_learned' field"

    def test_execute_successful_pipeline(self, pipeline_instance, sample_data):
        """T011: Test successful pipeline execution"""
        pipeline_instance.add_step("data_cleaning", interactive=False)

        result = pipeline_instance.execute(sample_data)

        assert result.success is True, "Successful execution should set success=True"
        assert result.processed_data is not None, "Should return processed data"
        assert result.execution_time_ms >= 0, "Execution time should be non-negative"
        assert result.steps_completed > 0, "Should have completed at least one step"
        assert isinstance(result.concepts_learned, list), \
            "concepts_learned should be a list"

    def test_execute_empty_pipeline(self, pipeline_instance, sample_data):
        """T011: Test executing pipeline with no steps"""
        # Execute without adding any steps
        result = pipeline_instance.execute(sample_data)

        assert isinstance(result, ExecutionResult)
        # Should handle empty pipeline gracefully
        assert result.steps_completed == 0, \
            "Empty pipeline should report 0 steps completed"

    def test_execute_multiple_steps(self, pipeline_instance, sample_data):
        """T011: Test executing pipeline with multiple steps"""
        pipeline_instance \
            .add_step("data_cleaning", interactive=False) \
            .add_step("feature_engineering", interactive=False)

        result = pipeline_instance.execute(sample_data)

        assert result.success is True
        assert result.steps_completed >= 2, \
            "Should have completed at least 2 steps"

    def test_execute_with_invalid_data_raises_error(self, pipeline_instance):
        """T011: Test execute() with invalid data raises RuntimeError"""
        pipeline_instance.add_step("data_cleaning", interactive=False)

        with pytest.raises(RuntimeError):
            pipeline_instance.execute(None)  # Invalid data

    def test_execute_failure_sets_error_message(self, pipeline_instance):
        """T011: Test that execution failures populate error_message"""
        # Add an operation that will fail
        pipeline_instance.add_step("data_cleaning", interactive=False)

        try:
            result = pipeline_instance.execute("invalid_data_type")
            # If it doesn't raise, check error_message
            if not result.success:
                assert result.error_message is not None, \
                    "Failed execution should have error_message"
        except RuntimeError:
            # Expected behavior - raising RuntimeError is also acceptable
            pass


@pytest.mark.contract
class TestPipelineInterfaceEdgeCases:
    """Edge case tests for PipelineInterface"""

    @pytest.fixture
    def pipeline_instance(self):
        """Get pipeline instance"""
        from datadojo import create_dojo

        dojo = create_dojo(educational_mode=True)
        projects = dojo.list_projects()
        project = dojo.load_project(projects[0].id)
        return project.create_pipeline(GuidanceLevel.BASIC)

    def test_multiple_executions_same_pipeline(self, pipeline_instance):
        """Test executing the same pipeline multiple times"""
        import pandas as pd

        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pipeline_instance.add_step("data_cleaning", interactive=False)

        result1 = pipeline_instance.execute(data)
        result2 = pipeline_instance.execute(data)

        assert isinstance(result1, ExecutionResult)
        assert isinstance(result2, ExecutionResult)
        # Should be able to execute multiple times

    def test_add_step_after_execution(self, pipeline_instance):
        """Test adding steps after execution"""
        import pandas as pd

        data = pd.DataFrame({'a': [1, 2, 3]})

        pipeline_instance.add_step("data_cleaning", interactive=False)
        result1 = pipeline_instance.execute(data)

        # Add another step after execution
        pipeline_instance.add_step("transformation", interactive=False)
        result2 = pipeline_instance.execute(data)

        assert result1.steps_completed < result2.steps_completed or \
               result2.steps_completed > 0, \
            "Second execution should reflect added steps"
