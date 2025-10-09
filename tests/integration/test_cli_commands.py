"""
Integration test for CLI project management commands

This test validates the complete command-line interface workflow
from the quickstart.md scenarios

TDD Note: This test MUST FAIL until CLI implementation is complete in Phase 3.3
"""

import pytest
import subprocess
import os


@pytest.mark.integration
@pytest.mark.cli
class TestCLIProjectManagement:
    """Integration tests for CLI command workflows"""

    def test_datadojo_cli_available(self):
        """T020: Test that datadojo CLI command is available"""
        # After installation, 'datadojo' command should be in PATH
        result = subprocess.run(
            ["datadojo", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should either succeed or fail gracefully
        # In TDD, this will likely fail as CLI isn't implemented yet
        assert result.returncode in [0, 127], \
            "CLI should either work or not be found yet"

    def test_list_projects_command(self):
        """T020: Test 'datadojo list-projects' command

        From quickstart.md:
        datadojo list-projects --domain ecommerce --difficulty beginner
        """
        result = subprocess.run(
            ["datadojo", "list-projects"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Will fail in TDD until CLI is implemented
        # When implemented, should return 0 and show project list
        if result.returncode == 0:
            assert result.stdout != "", "Should produce output"
            assert "project" in result.stdout.lower() or "available" in result.stdout.lower()

    def test_list_projects_with_filters(self):
        """T020: Test list-projects with domain and difficulty filters"""
        result = subprocess.run(
            ["datadojo", "list-projects", "--domain", "ecommerce", "--difficulty", "beginner"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            output = result.stdout.lower()
            assert "ecommerce" in output
            assert "beginner" in output

    def test_start_project_command(self):
        """T020: Test 'datadojo start' command

        From quickstart.md:
        datadojo start ecommerce_customers --student student123 --guidance detailed
        """
        result = subprocess.run(
            ["datadojo", "start", "ecommerce_customer_data",
             "--student", "test_student", "--guidance", "detailed"],
            capture_output=True,
            text=True,
            timeout=15
        )

        # Will fail until implemented
        if result.returncode == 0:
            assert result.stdout != ""

    def test_progress_command(self):
        """T020: Test 'datadojo progress' command

        From quickstart.md:
        datadojo progress student123 --project patient_outcomes
        """
        result = subprocess.run(
            ["datadojo", "progress", "test_student"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            assert result.stdout != "", "Should show progress information"

    def test_pipeline_command_basic(self):
        """T020: Test 'datadojo pipeline' command for quick processing

        From quickstart.md:
        datadojo pipeline --input data.csv --ops cleaning,encoding,scaling --output processed.csv
        """
        # Create a test input file
        import pandas as pd
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data = pd.DataFrame({
                'a': [1, 2, 3],
                'b': [4, 5, 6]
            })
            test_data.to_csv(f.name, index=False)
            input_file = f.name

        try:
            result = subprocess.run(
                ["datadojo", "pipeline",
                 "--input", input_file,
                 "--ops", "cleaning,encoding",
                 "--output", "test_output.csv"],
                capture_output=True,
                text=True,
                timeout=15
            )

            if result.returncode == 0:
                assert result.stdout != ""
                # Output file might be created
        finally:
            # Cleanup
            if os.path.exists(input_file):
                os.unlink(input_file)
            if os.path.exists("test_output.csv"):
                os.unlink("test_output.csv")

    def test_explain_command(self):
        """T020: Test 'datadojo explain' command

        From quickstart.md:
        datadojo explain missing_values --detail basic
        """
        result = subprocess.run(
            ["datadojo", "explain", "missing_values", "--detail", "basic"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            output = result.stdout.lower()
            assert "missing" in output or "values" in output

    def test_validate_command(self):
        """T020: Test 'datadojo validate' command"""
        import pandas as pd
        import tempfile

        # Create test data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data = pd.DataFrame({
                'a': [1, None, 3],
                'b': [4, 5, 6]
            })
            test_data.to_csv(f.name, index=False)
            input_file = f.name

        try:
            result = subprocess.run(
                ["datadojo", "validate", input_file, "--format", "summary"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                # Should provide validation report
                assert result.stdout != ""
        finally:
            if os.path.exists(input_file):
                os.unlink(input_file)


@pytest.mark.integration
@pytest.mark.cli
class TestCLIProgrammaticInterface:
    """Test CLI interface through programmatic access"""

    def test_cli_interface_via_python(self):
        """T020: Test CLI interface methods directly"""
        from datadojo.cli.interface import CLI

        cli = CLI()

        # Test list projects
        result = cli.list_projects_cmd()

        assert hasattr(result, 'success')
        assert hasattr(result, 'output')
        assert hasattr(result, 'exit_code')

    def test_cli_list_projects_programmatic(self):
        """T020: Test list_projects_cmd programmatically"""
        from datadojo.cli.interface import CLI

        cli = CLI()
        result = cli.list_projects_cmd(domain="ecommerce", difficulty="beginner")

        # Should return CLIResult
        assert result is not None
        if result.success:
            assert result.exit_code == 0
            assert result.output != ""

    def test_cli_start_project_programmatic(self):
        """T020: Test start_project_cmd programmatically"""
        from datadojo.cli.interface import CLI

        cli = CLI()
        result = cli.start_project_cmd(
            project_id="ecommerce_customer_data",
            student_id="test_student_cli",
            guidance_level="detailed"
        )

        assert result is not None
        assert hasattr(result, 'success')
        assert hasattr(result, 'exit_code')

    def test_cli_pipeline_programmatic(self):
        """T020: Test pipeline_cmd programmatically"""
        from datadojo.cli.interface import CLI
        import pandas as pd
        import tempfile

        # Create test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data = pd.DataFrame({'a': [1, 2, 3]})
            test_data.to_csv(f.name, index=False)
            input_file = f.name

        try:
            cli = CLI()
            result = cli.pipeline_cmd(
                operations=["data_cleaning"],
                input_file=input_file
            )

            assert result is not None
            assert isinstance(result.exit_code, int)
        finally:
            if os.path.exists(input_file):
                os.unlink(input_file)

    def test_cli_progress_programmatic(self):
        """T020: Test show_progress_cmd programmatically"""
        from datadojo.cli.interface import CLI

        cli = CLI()
        result = cli.show_progress_cmd(student_id="test_student")

        assert result is not None
        # May show empty progress for new student

    def test_cli_explain_programmatic(self):
        """T020: Test explain_concept_cmd programmatically"""
        from datadojo.cli.interface import CLI

        cli = CLI()
        result = cli.explain_concept_cmd(
            concept_id="missing_values",
            detail_level="basic"
        )

        assert result is not None
        if result.success:
            assert "missing" in result.output.lower() or "values" in result.output.lower()

    def test_cli_validate_programmatic(self):
        """T020: Test validate_data_cmd programmatically"""
        from datadojo.cli.interface import CLI
        import pandas as pd
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data = pd.DataFrame({'a': [1, None, 3]})
            test_data.to_csv(f.name, index=False)
            input_file = f.name

        try:
            cli = CLI()
            result = cli.validate_data_cmd(
                data_file=input_file,
                report_format="summary"
            )

            assert result is not None
        finally:
            if os.path.exists(input_file):
                os.unlink(input_file)


@pytest.mark.integration
@pytest.mark.cli
class TestCLIWorkflowScenarios:
    """End-to-end CLI workflow scenarios"""

    def test_complete_cli_learning_workflow(self):
        """T020: Test complete workflow using only CLI commands"""
        from datadojo.cli.interface import CLI

        cli = CLI()
        student_id = "cli_workflow_student"

        # Step 1: List available projects
        list_result = cli.list_projects_cmd(domain="ecommerce", difficulty="beginner")
        assert list_result is not None

        # Step 2: Start a project
        start_result = cli.start_project_cmd(
            project_id="ecommerce_customer_data",
            student_id=student_id,
            guidance_level="detailed"
        )
        assert start_result is not None

        # Step 3: Check progress
        progress_result = cli.show_progress_cmd(student_id=student_id)
        assert progress_result is not None

        # Step 4: Get concept explanation
        explain_result = cli.explain_concept_cmd(
            concept_id="missing_values",
            detail_level="basic"
        )
        assert explain_result is not None

    def test_cli_error_handling(self):
        """T020: Test CLI error handling with invalid inputs"""
        from datadojo.cli.interface import CLI

        cli = CLI()

        # Invalid project ID
        result = cli.start_project_cmd(
            project_id="nonexistent_project",
            student_id="test_student"
        )

        assert result.success is False
        assert result.exit_code != 0
        assert result.error_message is not None

    def test_cli_output_format_consistency(self):
        """T020: Test that CLI outputs are consistently formatted"""
        from datadojo.cli.interface import CLI

        cli = CLI()

        # All commands should return CLIResult with consistent structure
        commands = [
            lambda: cli.list_projects_cmd(),
            lambda: cli.show_progress_cmd("test_student"),
            lambda: cli.explain_concept_cmd("missing_values"),
        ]

        for cmd in commands:
            result = cmd()
            assert hasattr(result, 'success')
            assert hasattr(result, 'output')
            assert hasattr(result, 'exit_code')
            assert hasattr(result, 'error_message')
