"""
Contract tests for CLIInterface

These tests verify that CLI implementations adhere to the CLIInterface contract
defined in specs/001-use-the-requirements/contracts/cli_interface.py

TDD Note: These tests MUST FAIL until CLIInterface is implemented in Phase 3.3
"""

import pytest
from typing import List, Optional

from datadojo.cli.interface import CLIInterface, CLIResult


class TestCLIInterfaceContract:
    """Contract tests for CLIInterface"""

    @pytest.fixture
    def cli_instance(self):
        """Fixture providing a CLIInterface implementation

        TDD: This will fail until CLI class is implemented
        """
        from datadojo.cli.interface import CLI
        return CLI()

    def test_cli_implements_interface(self, cli_instance):
        """Verify CLI class implements CLIInterface"""
        assert isinstance(cli_instance, CLIInterface), \
            "CLI must implement CLIInterface"

    def test_list_projects_cmd_returns_cli_result(self, cli_instance):
        """T014: Test list_projects_cmd() returns CLIResult"""
        result = cli_instance.list_projects_cmd()

        assert isinstance(result, CLIResult), \
            "list_projects_cmd() must return CLIResult"

    def test_list_projects_cmd_successful_execution(self, cli_instance):
        """T014: Test list_projects_cmd() successful execution"""
        result = cli_instance.list_projects_cmd()

        assert result.success is True, "Should succeed with valid execution"
        assert result.exit_code == 0, "Success should have exit code 0"
        assert result.output != "", "Should produce non-empty output"
        assert result.error_message is None, "Success should have no error message"

    def test_list_projects_cmd_with_domain_filter(self, cli_instance):
        """T014: Test list_projects_cmd() with domain filter"""
        result = cli_instance.list_projects_cmd(domain="ecommerce")

        assert isinstance(result, CLIResult)
        assert result.success is True
        assert "ecommerce" in result.output.lower(), \
            "Output should contain filtered domain"

    def test_list_projects_cmd_with_difficulty_filter(self, cli_instance):
        """T014: Test list_projects_cmd() with difficulty filter"""
        result = cli_instance.list_projects_cmd(difficulty="beginner")

        assert isinstance(result, CLIResult)
        assert result.success is True
        assert "beginner" in result.output.lower()

    def test_list_projects_cmd_with_both_filters(self, cli_instance):
        """T014: Test list_projects_cmd() with domain and difficulty filters"""
        result = cli_instance.list_projects_cmd(
            domain="healthcare",
            difficulty="intermediate"
        )

        assert isinstance(result, CLIResult)
        assert result.success is True

    def test_list_projects_cmd_output_formats(self, cli_instance):
        """T014: Test list_projects_cmd() with different output formats"""
        formats = ["table", "json", "csv"]

        for fmt in formats:
            result = cli_instance.list_projects_cmd(format_output=fmt)

            assert isinstance(result, CLIResult)
            assert result.success is True, f"Should succeed with format {fmt}"
            assert result.output != "", f"Should produce output for format {fmt}"

    def test_list_projects_cmd_invalid_domain(self, cli_instance):
        """T014: Test list_projects_cmd() with invalid domain"""
        result = cli_instance.list_projects_cmd(domain="invalid_domain_xyz")

        # Should either fail gracefully or return empty results
        assert isinstance(result, CLIResult)
        if not result.success:
            assert result.exit_code != 0
            assert result.error_message is not None

    def test_start_project_cmd_returns_cli_result(self, cli_instance):
        """T015: Test start_project_cmd() returns CLIResult"""
        result = cli_instance.start_project_cmd(
            project_id="test_project",
            student_id="test_student"
        )

        assert isinstance(result, CLIResult), \
            "start_project_cmd() must return CLIResult"

    def test_start_project_cmd_with_required_parameters(self, cli_instance):
        """T015: Test start_project_cmd() with required parameters"""
        result = cli_instance.start_project_cmd(
            project_id="ecommerce_customer_data",
            student_id="student_001"
        )

        assert isinstance(result, CLIResult)
        # May succeed or fail depending on project availability

    def test_start_project_cmd_with_guidance_level(self, cli_instance):
        """T015: Test start_project_cmd() with different guidance levels"""
        guidance_levels = ["none", "basic", "detailed"]

        for level in guidance_levels:
            result = cli_instance.start_project_cmd(
                project_id="test_project",
                student_id="test_student",
                guidance_level=level
            )

            assert isinstance(result, CLIResult), \
                f"Should return CLIResult for guidance level {level}"

    def test_start_project_cmd_with_interactive_mode(self, cli_instance):
        """T015: Test start_project_cmd() with interactive flag"""
        result = cli_instance.start_project_cmd(
            project_id="test_project",
            student_id="test_student",
            interactive=True
        )

        assert isinstance(result, CLIResult)

        result_non_interactive = cli_instance.start_project_cmd(
            project_id="test_project",
            student_id="test_student",
            interactive=False
        )

        assert isinstance(result_non_interactive, CLIResult)

    def test_start_project_cmd_invalid_project_id(self, cli_instance):
        """T015: Test start_project_cmd() with invalid project ID"""
        result = cli_instance.start_project_cmd(
            project_id="nonexistent_project_12345",
            student_id="test_student"
        )

        assert isinstance(result, CLIResult)
        assert result.success is False, "Should fail for invalid project"
        assert result.exit_code != 0
        assert result.error_message is not None

    def test_start_project_cmd_empty_student_id(self, cli_instance):
        """T015: Test start_project_cmd() with empty student ID"""
        result = cli_instance.start_project_cmd(
            project_id="test_project",
            student_id=""
        )

        # Should fail validation
        assert isinstance(result, CLIResult)
        assert result.success is False
        assert result.error_message is not None

    def test_pipeline_cmd_returns_cli_result(self, cli_instance):
        """T016: Test pipeline_cmd() returns CLIResult"""
        result = cli_instance.pipeline_cmd(
            operations=["data_cleaning"],
            input_file="test_data.csv"
        )

        assert isinstance(result, CLIResult), \
            "pipeline_cmd() must return CLIResult"

    def test_pipeline_cmd_with_operations_and_input(self, cli_instance):
        """T016: Test pipeline_cmd() with operations and input file"""
        result = cli_instance.pipeline_cmd(
            operations=["data_cleaning", "feature_engineering"],
            input_file="test_input.csv"
        )

        assert isinstance(result, CLIResult)
        # May fail if file doesn't exist, but should return proper result

    def test_pipeline_cmd_with_output_file(self, cli_instance):
        """T016: Test pipeline_cmd() with output file specified"""
        result = cli_instance.pipeline_cmd(
            operations=["data_cleaning"],
            input_file="input.csv",
            output_file="output.csv"
        )

        assert isinstance(result, CLIResult)

    def test_pipeline_cmd_with_config_file(self, cli_instance):
        """T016: Test pipeline_cmd() with configuration file"""
        result = cli_instance.pipeline_cmd(
            operations=["data_cleaning"],
            input_file="input.csv",
            config_file="pipeline_config.yaml"
        )

        assert isinstance(result, CLIResult)

    def test_pipeline_cmd_multiple_operations(self, cli_instance):
        """T016: Test pipeline_cmd() with multiple operations"""
        operations = [
            "data_cleaning",
            "feature_engineering",
            "transformation",
            "validation"
        ]

        result = cli_instance.pipeline_cmd(
            operations=operations,
            input_file="test_data.csv"
        )

        assert isinstance(result, CLIResult)

    def test_pipeline_cmd_missing_input_file(self, cli_instance):
        """T016: Test pipeline_cmd() fails when input file is missing"""
        result = cli_instance.pipeline_cmd(
            operations=["data_cleaning"],
            input_file="nonexistent_file_xyz_12345.csv"
        )

        assert isinstance(result, CLIResult)
        assert result.success is False
        assert result.exit_code != 0
        assert result.error_message is not None

    def test_pipeline_cmd_empty_operations_list(self, cli_instance):
        """T016: Test pipeline_cmd() with empty operations list"""
        result = cli_instance.pipeline_cmd(
            operations=[],
            input_file="test_data.csv"
        )

        assert isinstance(result, CLIResult)
        # Should fail validation or return warning
        if not result.success:
            assert result.error_message is not None


@pytest.mark.contract
class TestCLIResultContract:
    """Tests for CLIResult data structure"""

    def test_cli_result_structure(self):
        """Test CLIResult has required fields"""
        result = CLIResult(
            success=True,
            output="Test output",
            exit_code=0,
            error_message=None
        )

        assert result.success is True
        assert result.output == "Test output"
        assert result.exit_code == 0
        assert result.error_message is None

    def test_cli_result_failure_case(self):
        """Test CLIResult for failure scenario"""
        result = CLIResult(
            success=False,
            output="",
            exit_code=1,
            error_message="Operation failed"
        )

        assert result.success is False
        assert result.exit_code == 1
        assert result.error_message == "Operation failed"

    def test_cli_result_with_partial_output(self):
        """Test CLIResult with partial output before error"""
        result = CLIResult(
            success=False,
            output="Partial output before failure",
            exit_code=2,
            error_message="Failed during processing"
        )

        assert result.success is False
        assert result.output != ""
        assert result.error_message is not None


@pytest.mark.contract
class TestAdditionalCLICommands:
    """Tests for additional CLI commands from the contract"""

    @pytest.fixture
    def cli_instance(self):
        """Get CLI instance"""
        from datadojo.cli.interface import CLI
        return CLI()

    def test_show_progress_cmd_exists(self, cli_instance):
        """Test show_progress_cmd() is implemented"""
        result = cli_instance.show_progress_cmd(student_id="test_student")

        assert isinstance(result, CLIResult)

    def test_explain_concept_cmd_exists(self, cli_instance):
        """Test explain_concept_cmd() is implemented"""
        result = cli_instance.explain_concept_cmd(
            concept_id="missing_values",
            detail_level="basic"
        )

        assert isinstance(result, CLIResult)

    def test_validate_data_cmd_exists(self, cli_instance):
        """Test validate_data_cmd() is implemented"""
        result = cli_instance.validate_data_cmd(
            data_file="test_data.csv",
            report_format="summary"
        )

        assert isinstance(result, CLIResult)

    def test_create_project_cmd_exists(self, cli_instance):
        """Test create_project_cmd() is implemented"""
        result = cli_instance.create_project_cmd(
            name="Test Project",
            domain="ecommerce",
            difficulty="beginner",
            dataset_path="test_dataset.csv"
        )

        assert isinstance(result, CLIResult)
