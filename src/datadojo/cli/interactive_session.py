from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion
from typing import Dict, Any, List
import shlex
import argparse
import io
import sys
import inspect

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.dojo import Dojo
from ..utils.theme import default_theme
from .list_projects import list_projects
from .start_project import start_project
from .show_progress import show_progress
from .explain_concept import explain_concept
from .complete_step import complete_step
from .practice import practice


class DojoCompleter(Completer):
    def __init__(self, session):
        self.session = session

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        words = text.split()

        if len(words) == 0 or (len(words) == 1 and not text.endswith(" ")):
            for cmd in self.session.commands:
                if cmd.startswith(words[0] if words else ""):
                    yield Completion(cmd, start_position=-len(words[0]) if words else 0)
        else:
            command = words[0]
            if command in self.session.command_args:
                current_arg = ""
                if not text.endswith(" "):
                    current_arg = words[-1]
                
                for arg in self.session.command_args[command]:
                    if arg.startswith(current_arg):
                        yield Completion(arg, start_position=-len(current_arg))

class DojoSession:
    def __init__(self, dojo_instance: Dojo):
        self.dojo = dojo_instance
        self.console = Console(theme=default_theme)
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
            "back": self._back_command,
            "set-student": self._set_student_command,
        }
        self.command_args = {
            "list-projects": ["--domain", "--difficulty"],
            "start": ["--student", "--guidance"],
            "explain": ["--detail", "--examples"],
            "progress": ["--project", "--format"],
            "complete-step": [], "practice": [], "help": [],
            "exit": [], "quit": [], "back": [], "set-student": [],
        }
        self.completer = DojoCompleter(self)

    def _help_command(self, args: List[str]):
        if not args:
            table = Table(title="[title]Available Commands[/title]", border_style="border")
            table.add_column("Command", style="command", no_wrap=True)
            table.add_column("Description", style="info")
            for cmd, func in self.commands.items():
                description = (func.__doc__ or "No description.").split('\n')[0]
                table.add_row(cmd, description)
            self.console.print(table)
        else:
            cmd_name = args[0]
            if cmd_name in self.commands:
                docstring = self.commands[cmd_name].__doc__ or f"No help for '{cmd_name}'."
                self.console.print(Panel(docstring.strip(), title=f"[title]Help: {cmd_name}[/title]", border_style="border"))
            else:
                self.console.print(f"Unknown command: '{cmd_name}'.", style="danger")

    def _exit_command(self, args: List[str]):
        raise EOFError

    def _back_command(self, args: List[str]):
        self.current_project_id = None
        self.prompt_text = "(dojo) > "
        self.console.print("Returned to the main dojo prompt.", style="info")

    def _set_student_command(self, args: List[str]):
        if not args:
            self.console.print("Usage: set-student <student_name>", style="warning")
            return
        self.current_student_id = args[0]
        self.console.print(f"Student ID set to: [info]{self.current_student_id}[/info]")

    def _list_projects_command(self, args: List[str]):
        parser = argparse.ArgumentParser(prog="list-projects", add_help=False)
        parser.add_argument("--domain", choices=["ecommerce", "healthcare", "finance"])
        parser.add_argument("--difficulty", choices=["beginner", "intermediate", "advanced"])
        try:
            parsed_args = parser.parse_args(args)
            result = list_projects(dojo=self.dojo, domain=parsed_args.domain, difficulty=parsed_args.difficulty)
            if result.success:
                projects = result.output
                if not projects:
                    self.console.print(result.error_message or "No projects found.", style="info")
                    return
                table = Table(title="[title]Available Projects[/title]", border_style="border")
                table.add_column("ID", style="code", no_wrap=True)
                table.add_column("Name", style="bold")
                table.add_column("Domain", style="info")
                table.add_column("Difficulty", style="info")
                for p in projects:
                    table.add_row(p.id, p.name, p.domain.value, p.difficulty.value)
                self.console.print(table)
            else:
                self.console.print(f"Error: {result.error_message}", style="danger")
        except SystemExit:
            self.console.print(parser.format_help(), style="warning")
    
    def _start_command(self, args: List[str]):
        parser = argparse.ArgumentParser(prog="start", add_help=False)
        parser.add_argument("project_id")
        parser.add_argument("--student", required=True)
        try:
            parsed_args = parser.parse_args(args)
            result = start_project(dojo=self.dojo, project_id=parsed_args.project_id, student_id=parsed_args.student)
            if result.success:
                self.current_project_id = parsed_args.project_id
                self.current_student_id = parsed_args.student
                self.prompt_text = f"({self.current_project_id}) > "
                self.console.print(result.output)
            else:
                self.console.print(f"Error: {result.error_message}", style="danger")
        except SystemExit:
            self.console.print(parser.format_help(), style="warning")

    def _explain_command(self, args: List[str]):
        parser = argparse.ArgumentParser(prog="explain", add_help=False)
        parser.add_argument("concept")
        try:
            parsed_args = parser.parse_args(args)
            result = explain_concept(dojo=self.dojo, concept_id=parsed_args.concept)
            if result.success:
                self.console.print(result.output)
            else:
                self.console.print(f"Error: {result.error_message}", style="danger")
        except SystemExit:
            self.console.print(parser.format_help(), style="warning")

    def _progress_command(self, args: List[str]):
        parser = argparse.ArgumentParser(prog="progress", add_help=False)
        parser.add_argument("--project", help="Project ID")
        try:
            parsed_args = parser.parse_args(args)
            project_id = parsed_args.project or self.current_project_id
            if not self.current_student_id:
                self.console.print("Please set a student ID first using 'set-student <name>'.", style="warning")
                return
            if not project_id:
                self.console.print("Please start a project first or specify one with --project.", style="warning")
                return
            
            result = show_progress(dojo=self.dojo, student_id=self.current_student_id, project_id=project_id)
            if result.success:
                self.console.print(result.output)
            else:
                self.console.print(f"Error: {result.error_message}", style="danger")
        except SystemExit:
            self.console.print(parser.format_help(), style="warning")
            
    def _complete_step_command(self, args: List[str]):
        parser = argparse.ArgumentParser(prog="complete-step", add_help=False)
        parser.add_argument("step_id")
        try:
            parsed_args = parser.parse_args(args)
            if not self.current_student_id:
                self.console.print("Please set a student ID first using 'set-student <name>'.", style="warning")
                return
            if not self.current_project_id:
                self.console.print("Please start a project first.", style="warning")
                return
            
            result = complete_step(dojo=self.dojo, student_id=self.current_student_id, project_id=self.current_project_id, step_id=parsed_args.step_id)
            if result.success:
                self.console.print(result.output)
            else:
                self.console.print(f"Error: {result.error_message}", style="danger")
        except SystemExit:
            self.console.print(parser.format_help(), style="warning")
            
    def _practice_command(self, args: List[str]):
        parser = argparse.ArgumentParser(prog="practice", add_help=False)
        parser.add_argument("concept_id")
        try:
            parsed_args = parser.parse_args(args)
            if not self.current_student_id:
                self.console.print("Please set a student ID first using 'set-student <name>'.", style="warning")
                return
            if not self.current_project_id:
                self.console.print("Please start a project first.", style="warning")
                return
                
            result = practice(dojo=self.dojo, student_id=self.current_student_id, project_id=self.current_project_id, concept_id=parsed_args.concept_id)
            if result.success:
                self.console.print(result.output)
            else:
                self.console.print(f"Error: {result.error_message}", style="danger")
        except SystemExit:
            self.console.print(parser.format_help(), style="warning")

    def run(self):
        self.console.print(Panel("Welcome to the [title]DataDojo Interactive Session[/title]!\nType 'help' for commands or 'exit' to quit.", border_style="border"))
        while True:
            try:
                user_input = prompt(self.prompt_text, completer=self.completer)
                if not user_input.strip():
                    continue
                parts = shlex.split(user_input)
                command_name = parts[0]
                command_args = parts[1:]
                if command_name in self.commands:
                    self.commands[command_name](command_args)
                else:
                    self.console.print(f"Unknown command: '{command_name}'. Type 'help'.", style="warning")
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
        self.console.print("Exiting DataDojo session. Goodbye!", style="info")

def start_interactive_session(dojo_instance: Dojo):
    session = DojoSession(dojo_instance)
    session.run()