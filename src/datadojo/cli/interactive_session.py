from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from typing import Dict, Any, List, Optional
import shlex
import argparse
import io
import sys
import inspect

from ..core.dojo import Dojo
from .list_projects import list_projects
from .start_project import start_project
from .show_progress import show_progress
from .explain_concept import explain_concept
from .complete_step import complete_step
from .practice import practice


class DojoSession:
    def __init__(self, dojo_instance: Dojo):
        self.dojo = dojo_instance
        self.current_student_id = None
        self.current_project_id = None
        self.prompt_text = "(dojo) > "
        self.commands = {
            "help": self._help_command,
            "exit": self._exit_command,
            "quit": self._exit_command,
            "list-projects": self._list_projects_command,
            "start": self._start_command,
            "explain": self._explain_command,
            "progress": self._progress_command,
            "complete-step": self._complete_step_command,
            "practice": self._practice_command,
        }
        self.completer = WordCompleter(list(self.commands.keys()) + ["--domain", "--difficulty", "--student", "--project", "--detail", "--examples"], ignore_case=True)

    def _help_command(self, args: List[str]) -> str:
        """Displays help information.\nUsage: help [command]"""
        if not args:
            return "Available commands: " + ", ".join(self.commands.keys()) + "\nType 'help <command>' for more details."
        
        cmd_name = args[0]
        if cmd_name in self.commands:
            return self.commands[cmd_name].__doc__ or f"No specific help available for '{cmd_name}'."
        return f"Unknown command: '{cmd_name}'."

    def _exit_command(self, args: List[str]) -> str:
        """Exits the interactive session."""
        raise EOFError # Use EOFError to break the loop

    def _execute_cli_command(self, func, args_list, parser_creator) -> str:
        """Helper to parse and execute CLI commands within the interactive session."""
        parser = parser_creator()
        
        # Capture stderr to prevent argparse from printing directly
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            parsed_args = parser.parse_args(args_list)
            
            # Prepare arguments for the function call
            # Inspect the function signature to get expected parameter names
            sig = inspect.signature(func)
            call_args = {}

            for param_name, param in sig.parameters.items():
                if param_name == 'dojo':
                    continue # 'dojo' is handled separately

                # Check for direct match from parsed_args
                if hasattr(parsed_args, param_name):
                    value = getattr(parsed_args, param_name)
                    if value is not None:
                        call_args[param_name] = value
                # Handle common mismatches, e.g., 'student' in parsed_args but 'student_id' in func
                elif param_name == 'student_id' and hasattr(parsed_args, 'student'):
                    value = getattr(parsed_args, 'student')
                    if value is not None:
                        call_args['student_id'] = value
                elif param_name == 'concept_id' and hasattr(parsed_args, 'concept'):
                    value = getattr(parsed_args, 'concept')
                    if value is not None:
                        call_args['concept_id'] = value
                
                # If a required parameter is still missing and it's not handled by argparse default
                # this will typically be caught by the called function itself.

            result = func(dojo=self.dojo, **call_args)
            
            sys.stderr.seek(0)
            argparse_output = sys.stderr.read()

            if result.success:
                return f"{argparse_output.strip()}\n{result.output}" if argparse_output else result.output
            else:
                return f"Error: {argparse_output.strip()}\n{result.error_message}" if argparse_output else f"Error: {result.error_message}"

        except SystemExit as e:
            # argparse calls sys.exit(status) on errors or --help
            sys.stderr.seek(0)
            argparse_output = sys.stderr.read()
            if e.code == 0: # This usually means --help was requested
                return argparse_output
            else:
                return f"Invalid arguments: {argparse_output.strip()}"
        except Exception as e:
            return f"Command execution error: {e}"
        finally:
            sys.stderr = old_stderr # Ensure stderr is always restored


    def _list_projects_command(self, args: List[str]) -> str:
        """Lists available learning projects.\nUsage: list-projects [--domain <domain>] [--difficulty <level>]"""
        def parser_creator():
            parser = argparse.ArgumentParser(prog="list-projects", add_help=False, exit_on_error=False)
            parser.add_argument("--domain", choices=["ecommerce", "healthcare", "finance"])
            parser.add_argument("--difficulty", choices=["beginner", "intermediate", "advanced"])
            return parser
        return self._execute_cli_command(list_projects, args, parser_creator)
        
    def _start_command(self, args: List[str]) -> str:
        """Begins a learning project.\nUsage: start <project_id> --student <student_id> [--guidance <level>]"""
        def parser_creator():
            parser = argparse.ArgumentParser(prog="start", add_help=False, exit_on_error=False)
            parser.add_argument("project_id")
            parser.add_argument("--student", required=True)
            parser.add_argument("--guidance", choices=["none", "basic", "detailed"], default="detailed")
            return parser
        
        result_output = self._execute_cli_command(start_project, args, parser_creator)
        
        # If start was successful, update session state
        if not result_output.startswith("Error:") and "Invalid arguments:" not in result_output:
            try:
                parser = parser_creator()
                parsed_args = parser.parse_args(args)
                self.current_student_id = parsed_args.student
                self.current_project_id = parsed_args.project_id
                self.prompt_text = f"({self.current_project_id}) > "
            except (SystemExit, Exception):
                pass # Already handled by _execute_cli_command, just preventing duplicate error

        return result_output

    def _explain_command(self, args: List[str]) -> str:
        """Get explanation of data preprocessing concept.\nUsage: explain <concept_id> [--detail <level>] [--examples]"""
        def parser_creator():
            parser = argparse.ArgumentParser(prog="explain", add_help=False, exit_on_error=False)
            parser.add_argument("concept")
            parser.add_argument("--detail", choices=["basic", "detailed", "expert"], default="basic")
            parser.add_argument("--examples", action="store_true")
            return parser
        return self._execute_cli_command(explain_concept, args, parser_creator)

    def _progress_command(self, args: List[str]) -> str:
        """Show learning progress.\nUsage: progress [--student_id <student_id>] [--project <project_id>] [--format <format>]"""
        def parser_creator():
            parser = argparse.ArgumentParser(prog="progress", add_help=False, exit_on_error=False)
            parser.add_argument("--student_id", help="Student identifier")
            parser.add_argument("--project", help="Show progress for specific project only")
            parser.add_argument("--format", choices=["summary", "detailed", "json"], default="summary")
            return parser
        
        # Custom parsing for progress to allow defaulting to current_student/project
        parser = parser_creator()
        try:
            parsed_args = parser.parse_args(args)
            student_id = parsed_args.student_id or self.current_student_id
            project_id = parsed_args.project or self.current_project_id

            if not student_id:
                return "Error: Student ID is required. Either specify with --student_id or start a project."
            if not project_id and self.current_project_id:
                 # If no project specified, and one is active, show progress for current project
                project_id = self.current_project_id
            elif not project_id:
                return "Error: Project ID is required if no project is active. Specify with --project."

            # Recreate parser to handle arguments for show_progress correctly
            actual_args = []
            if student_id:
                actual_args.extend(["--student_id", student_id])
            if project_id:
                actual_args.extend(["--project", project_id])
            if parsed_args.format:
                actual_args.extend(["--format", parsed_args.format])

            return self._execute_cli_command(show_progress, actual_args, parser_creator)

        except SystemExit as e:
            # Handle help output
            sys.stderr.seek(0)
            argparse_output = sys.stderr.read()
            if e.code == 0:
                return argparse_output
            else:
                return f"Invalid arguments for progress: {argparse_output.strip()}"
        except Exception as e:
            return f"Command execution error: {e}"

    def _complete_step_command(self, args: List[str]) -> str:
        """Marks a step as complete.\nUsage: complete-step <step_id> [--student <student_id>] [--project <project_id>]"""
        def parser_creator():
            parser = argparse.ArgumentParser(prog="complete-step", add_help=False, exit_on_error=False)
            parser.add_argument("step_id", help="The ID of the step to mark as complete")
            parser.add_argument("--student", help="Student identifier for progress tracking")
            parser.add_argument("--project", help="Project identifier")
            return parser
        
        # Custom parsing for complete-step to allow defaulting to current_student/project
        parser = parser_creator()
        try:
            parsed_args = parser.parse_args(args)
            student_id = parsed_args.student or self.current_student_id
            project_id = parsed_args.project or self.current_project_id
            step_id = parsed_args.step_id
            
            if not student_id:
                return "Error: Student ID is required. Either specify with --student or start a project."
            if not project_id:
                return "Error: Project ID is required. Either specify with --project or start a project."

            # Recreate parser to handle arguments for complete_step correctly
            actual_args = [step_id]
            if student_id:
                actual_args.extend(["--student", student_id])
            if project_id:
                actual_args.extend(["--project", project_id])

            return self._execute_cli_command(complete_step, actual_args, parser_creator)

        except SystemExit as e:
            sys.stderr.seek(0)
            argparse_output = sys.stderr.read()
            if e.code == 0:
                return argparse_output
            else:
                return f"Invalid arguments for complete-step: {argparse_output.strip()}"
        except Exception as e:
            return f"Command execution error: {e}"

    def _practice_command(self, args: List[str]) -> str:
        """Starts an interactive practice session for a concept.\nUsage: practice <concept_id>"""
        if not args:
            return "Usage: practice <concept_id>"

        concept_id = args[0]
        student_id = self.current_student_id
        project_id = self.current_project_id

        if not student_id:
            return "Error: Student ID is required. Please start a project first."
        if not project_id:
            return "Error: Project ID is required. Please start a project first."

        result = practice(
            dojo=self.dojo,
            student_id=student_id,
            project_id=project_id,
            concept_id=concept_id
        )
        return result.output if result.success else f"Error: {result.error_message}"

    def run(self):
        print("Welcome to the DataDojo Interactive Session!")
        print("Type 'help' for a list of commands, or 'exit' to quit.")

        while True:
            try:
                user_input = prompt(self.prompt_text, completer=self.completer)
                if not user_input.strip():
                    continue

                parts = shlex.split(user_input)
                command_name = parts[0]
                command_args = parts[1:]

                if command_name in self.commands:
                    try:
                        output = self.commands[command_name](command_args)
                        if output:
                            print(output)
                    except EOFError: # Raised by _exit_command
                        break
                    except Exception as e:
                        print(f"Command error: {e}")
                else:
                    print(f"Unknown command: '{command_name}'. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\nSession interrupted. Type 'exit' to quit.")
            except EOFError:
                break
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

        print("Exiting DataDojo session. Goodbye!")


def start_interactive_session(dojo_instance: Dojo):
    """Initializes and runs the interactive Dojo session."""
    session = DojoSession(dojo_instance)
    session.run()

