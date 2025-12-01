"""Main CLI entry point for DataDojo."""

import argparse
import sys

from ..core.dojo import Dojo
from .list_projects import list_projects
from .start_project import start_project
from .show_progress import show_progress
from .pipeline_cmd import pipeline_cmd
from .explain_concept import explain_concept
from .validate_data import validate_data
from .complete_step import complete_step
from .complete_step import complete_step
from .doctor import doctor
from .interactive_session import start_interactive_session
from .practice import practice
from .web_launch import launch_web_dashboard, check_web_status


def main():
    """Main CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        prog="datadojo",
        description="DataDojo - AI-Powered Data Preparation Learning Framework"
    )

    # Add global options
    parser.add_argument(
        "--educational",
        action="store_true",
        help="Enable educational mode with guidance"
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list-projects command
    list_parser = subparsers.add_parser(
        "list-projects",
        help="Show available learning projects"
    )
    list_parser.add_argument(
        "--domain",
        choices=["ecommerce", "healthcare", "finance"],
        help="Filter by domain"
    )
    list_parser.add_argument(
        "--difficulty",
        choices=["beginner", "intermediate", "advanced"],
        help="Filter by difficulty level"
    )
    list_parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)"
    )

    # start command
    start_parser = subparsers.add_parser(
        "start",
        help="Begin a learning project"
    )
    start_parser.add_argument(
        "project_id",
        help="Project identifier"
    )
    start_parser.add_argument(
        "--student",
        required=True,
        help="Student identifier for progress tracking"
    )
    start_parser.add_argument(
        "--guidance",
        choices=["none", "basic", "detailed"],
        default="detailed",
        help="Educational assistance level (default: detailed)"
    )
    start_parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Enable step-by-step interactive mode"
    )

    # progress command
    progress_parser = subparsers.add_parser(
        "progress",
        help="Show learning progress"
    )
    progress_parser.add_argument(
        "student_id",
        help="Student identifier"
    )
    progress_parser.add_argument(
        "--project",
        help="Show progress for specific project only"
    )
    progress_parser.add_argument(
        "--format",
        choices=["summary", "detailed", "json"],
        default="summary",
        help="Output detail level (default: summary)"
    )

    # pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Execute preprocessing pipeline"
    )
    pipeline_parser.add_argument(
        "--ops",
        required=True,
        help="Comma-separated preprocessing operations"
    )
    pipeline_parser.add_argument(
        "--input",
        required=True,
        help="Input dataset path"
    )
    pipeline_parser.add_argument(
        "--output",
        help="Output file path (optional)"
    )
    pipeline_parser.add_argument(
        "--config",
        help="Pipeline configuration file (optional)"
    )

    # explain command
    explain_parser = subparsers.add_parser(
        "explain",
        help="Get explanation of data preprocessing concept"
    )
    explain_parser.add_argument(
        "concept",
        help="Concept to explain (e.g., missing_values, outliers)"
    )
    explain_parser.add_argument(
        "--detail",
        choices=["basic", "detailed", "expert"],
        default="basic",
        help="Explanation depth (default: basic)"
    )
    explain_parser.add_argument(
        "--examples",
        action="store_true",
        help="Include code examples"
    )

    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate dataset quality"
    )
    validate_parser.add_argument(
        "data_file",
        help="Path to dataset for validation"
    )
    validate_parser.add_argument(
        "--rules",
        help="Custom validation rules file"
    )
    validate_parser.add_argument(
        "--format",
        choices=["summary", "detailed", "json"],
        default="summary",
        help="Report format (default: summary)"
    )

    # complete-step command
    complete_step_parser = subparsers.add_parser(
        "complete-step",
        help="Mark a step as complete"
    )
    complete_step_parser.add_argument(
        "step_id",
        help="The ID of the step to mark as complete"
    )
    complete_step_parser.add_argument(
        "--student",
        required=True,
        help="Student identifier for progress tracking"
    )
    complete_step_parser.add_argument(
        "--project",
        required=True,
        help="Project identifier"
    )

    # doctor command
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check the DataDojo environment and report any issues"
    )

    # learn command
    learn_parser = subparsers.add_parser(
        "learn",
        help="Start an interactive learning session"
    )

    # practice command
    practice_parser = subparsers.add_parser(
        "practice",
        help="Start an interactive practice session for a concept"
    )
    practice_parser.add_argument(
        "concept_id",
        help="The ID of the concept to practice"
    )
    practice_parser.add_argument(
        "--student",
        required=True,
        help="Student identifier for progress tracking"
    )
    practice_parser.add_argument(
        "--project",
        required=True,
        help="Project identifier"
    )

    # web command
    web_parser = subparsers.add_parser(
        "web",
        help="Launch interactive web dashboard"
    )
    web_parser.add_argument(
        "--port",
        type=int,
        help="Port to run web server on (auto-detected if not specified)"
    )
    web_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser"
    )
    web_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    web_parser.add_argument(
        "--status",
        action="store_true",
        help="Check if web dashboard is running"
    )

    # Parse arguments
    args = parser.parse_args()

    # Show help if no command specified
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    if args.command == "doctor":
        result = doctor()
        if result.success:
            print(result.output)
            sys.exit(result.exit_code)
        else:
            if result.error_message:
                print(f"Error: {result.error_message}", file=sys.stderr)
            sys.exit(result.exit_code)

    elif args.command == "learn":
        # Initialize Dojo even for learn command, so it can be passed to the session
        try:
            dojo = Dojo(educational_mode=True) # Interactive session is always educational
        except Exception as e:
            print(f"Error: Failed to initialize DataDojo: {e}", file=sys.stderr)
            sys.exit(1)
        start_interactive_session(dojo)
        sys.exit(0)
    else:
        # Initialize Dojo
        try:
            dojo = Dojo(educational_mode=args.educational)
        except Exception as e:
            print(f"Error: Failed to initialize DataDojo: {e}", file=sys.stderr)
            sys.exit(1)

        # Execute command
        result = None

        try:
            if args.command == "list-projects":
                result = list_projects(
                    dojo=dojo,
                    domain=args.domain,
                    difficulty=args.difficulty,
                    format_output=args.format
                )

            elif args.command == "start":
                result = start_project(
                    dojo=dojo,
                    project_id=args.project_id,
                    student_id=args.student,
                    guidance_level=args.guidance,
                    interactive=args.interactive
                )

            elif args.command == "progress":
                result = show_progress(
                    dojo=dojo,
                    student_id=args.student_id,
                    project_id=args.project,
                    format_output=args.format
                )

            elif args.command == "pipeline":
                operations = [op.strip() for op in args.ops.split(",")]
                result = pipeline_cmd(
                    dojo=dojo,
                    operations=operations,
                    input_file=getattr(args, 'input'),
                    output_file=args.output,
                    config_file=args.config,
                    educational_mode=args.educational
                )

            elif args.command == "explain":
                result = explain_concept(
                    dojo=dojo,
                    concept_id=args.concept,
                    detail_level=args.detail,
                    include_examples=args.examples
                )

            elif args.command == "validate":
                result = validate_data(
                    dojo=dojo,
                    data_file=args.data_file,
                    validation_rules=args.rules,
                    report_format=args.format
                )

            elif args.command == "complete-step":
                result = complete_step(
                    dojo=dojo,
                    student_id=args.student,
                    project_id=args.project,
                    step_id=args.step_id
                )
            
            elif args.command == "practice":
                result = practice(
                    dojo=dojo,
                    student_id=args.student,
                    project_id=args.project,
                    concept_id=args.concept_id
                )

            elif args.command == "web":
                if args.status:
                    result = check_web_status(args.port or 8501)
                else:
                    result = launch_web_dashboard(
                        port=args.port,
                        auto_open=not args.no_browser,
                        debug=args.debug
                    )

            else:
                print(f"Error: Unknown command: {args.command}", file=sys.stderr)
                sys.exit(1)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.", file=sys.stderr)
            sys.exit(130)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Handle result
    if result:
        if result.success:
            print(result.output)
            sys.exit(result.exit_code)
        else:
            if result.error_message:
                print(f"Error: {result.error_message}", file=sys.stderr)
            sys.exit(result.exit_code)
    else:
        print("Error: Command did not return a result", file=sys.stderr)
        sys.exit(1)



if __name__ == "__main__":
    main()
