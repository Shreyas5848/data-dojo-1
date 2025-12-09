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
from rich.text import Text

from ..core.dojo import Dojo
from ..utils.claude_theme import claude_theme
from ..utils.theme import default_theme
from ..utils.synthetic_data_generator import SyntheticDataGenerator
from ..utils.intelligent_profiler import IntelligentProfiler, quick_profile
from .list_projects import list_projects
from .list_datasets import list_datasets, discover_datasets
from .start_project import start_project
from .show_progress import show_progress
from .explain_concept import explain_concept
from .complete_step import complete_step
from .practice import practice
from .web_launch import launch_web_dashboard
import pandas as pd
from pathlib import Path

ASCII_ART = """
[bold #FF9900]
██████╗  █████╗ ████████╗ █████╗     ██████╗  ██████╗      ██╗ ██████╗ 
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ██║  ██║██╔═══██╗     ██║██╔═══██╗
██║  ██║███████║   ██║   ███████║    ██║  ██║██║   ██║     ██║██║   ██║
██║  ██║██╔══██║   ██║   ██╔══██║    ██║  ██║██║   ██║██   ██║██║   ██║
██████╔╝██║  ██║   ██║   ██║  ██║    ██████╔╝╚██████╔╝╚█████╔╝╚██████╔╝
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚═════╝  ╚═════╝  ╚════╝  ╚═════╝ 
[/bold #FF9900]
"""

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
        self.themes = {"default": default_theme, "claude": claude_theme}
        self.console = Console(theme=self.themes["claude"]) # Default to claude theme
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
            "theme": self._theme_command,
            "generate-data": self._generate_data_command,
            "profile-data": self._profile_data_command,
            "list-datasets": self._list_datasets_command,
            "profile-all": self._profile_all_command,
            "web": self._web_command,
            "notebook": self._notebook_command,
        }
        self.command_args = {
            "list-projects": ["--domain", "--difficulty"],
            "start": ["--student", "--guidance"],
            "explain": ["--detail", "--examples"],
            "progress": ["--project", "--format"],
            "complete-step": [], "practice": [], "help": [],
            "exit": [], "quit": [], "back": [], "set-student": [], "theme": ["default", "claude"],
            "generate-data": ["--domain", "--size", "--output"],
            "profile-data": ["--file", "--output", "--format"],
            "list-datasets": ["--domain", "--format", "--min-size"],
            "profile-all": ["--domain", "--output-dir", "--format"],
            "web": ["--port", "--no-browser"],
            "notebook": ["--template", "--output", "--open", "--list-templates"],
        }
        self.completer = DojoCompleter(self)

    def _help_command(self, args: List[str]):
        """Show available commands and their descriptions.
Usage: help [command_name]

Without arguments, lists all commands. 
With a command name, shows detailed help for that command."""
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
        """Exit the interactive session."""
        raise EOFError

    def _back_command(self, args: List[str]):
        """Return to the main dojo prompt from a project context."""
        self.current_project_id = None
        self.prompt_text = "(dojo) > "
        self.console.print("Returned to the main dojo prompt.", style="info")

    def _set_student_command(self, args: List[str]):
        """Set the current student ID for progress tracking.
Usage: set-student <student_name>"""
        if not args:
            self.console.print("Usage: set-student <student_name>", style="warning")
            return
        self.current_student_id = args[0]
        self.console.print(f"Student ID set to: [info]{self.current_student_id}[/info]")

    def _theme_command(self, args: List[str]):
        """Switches the color theme.
Usage: theme [default|claude]"""
        if not args or args[0] not in self.themes:
            self.console.print(f"Usage: theme [{'/'.join(self.themes.keys())}]", style="warning")
            return
        self.console.theme = self.themes[args[0]]
        self.console.print(f"Theme changed to [info]{args[0]}[/info].")

    def _web_command(self, args: List[str]):
        """Launch the interactive web dashboard.
Usage: web [--port PORT] [--no-browser]

The web dashboard provides:
  • Interactive data exploration & profiling
  • Auto-generated learning notebooks  
  • Progress tracking with XP & achievements
  • 9 guided projects across 3 domains"""
        parser = argparse.ArgumentParser(prog="web", add_help=False)
        parser.add_argument("--port", type=int, help="Port to run on")
        parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
        try:
            parsed_args = parser.parse_args(args)
            self.console.print("\n🌐 [bold cyan]Launching web dashboard...[/bold cyan]\n")
            result = launch_web_dashboard(
                port=parsed_args.port,
                auto_open=not parsed_args.no_browser
            )
            if result and result.success:
                self.console.print(result.output)
            elif result and result.error_message:
                self.console.print(f"Error: {result.error_message}", style="danger")
        except SystemExit:
            self.console.print(parser.format_help(), style="warning")

    def _notebook_command(self, args: List[str]):
        """Generate a Jupyter notebook from a dataset file.
Usage: notebook <dataset.csv> [--template TYPE] [--output DIR] [--open]
       notebook --list-templates

Templates: exploratory_data_analysis, data_cleaning, classification_analysis,
           regression_analysis, time_series_analysis, clustering_analysis,
           dimensionality_reduction, feature_engineering

Example: notebook datasets/healthcare/patient_demographics.csv --template data_cleaning"""
        from .generate_notebook import generate_notebook, list_templates
        
        parser = argparse.ArgumentParser(prog="notebook", add_help=False)
        parser.add_argument("dataset", nargs="?", help="Path to CSV dataset")
        parser.add_argument("--template", "-t", default="exploratory_data_analysis",
                          choices=['exploratory_data_analysis', 'data_cleaning', 'classification_analysis',
                                   'regression_analysis', 'time_series_analysis', 'clustering_analysis',
                                   'dimensionality_reduction', 'feature_engineering'])
        parser.add_argument("--output", "-o", help="Output directory")
        parser.add_argument("--open", action="store_true", help="Open notebook after creation")
        parser.add_argument("--list-templates", action="store_true", help="List available templates")
        
        try:
            parsed_args = parser.parse_args(args)
            
            if parsed_args.list_templates:
                result = list_templates()
                self.console.print(result.output)
                return
            
            if not parsed_args.dataset:
                self.console.print("Usage: notebook <dataset.csv> [--template TYPE]", style="warning")
                self.console.print("       notebook --list-templates", style="warning")
                return
            
            self.console.print(f"\n📓 [bold cyan]Generating notebook for {parsed_args.dataset}...[/bold cyan]\n")
            
            result = generate_notebook(
                dataset_path=parsed_args.dataset,
                template_type=parsed_args.template,
                output_dir=parsed_args.output,
                open_notebook=parsed_args.open
            )
            
            if result.success:
                self.console.print(result.output)
            else:
                self.console.print(f"Error: {result.error_message}", style="danger")
                
        except SystemExit:
            self.console.print(parser.format_help(), style="warning")

    def _list_projects_command(self, args: List[str]):
        """List available learning projects across different domains.
Usage: list-projects [--domain <ecommerce|healthcare|finance>] [--difficulty <beginner|intermediate|advanced>]"""
        parser = argparse.ArgumentParser(prog="list-projects", add_help=False)
        parser.add_argument("--domain", choices=["ecommerce", "healthcare", "finance"])
        parser.add_argument("--difficulty", choices=["beginner", "intermediate", "advanced"])
        try:
            parsed_args = parser.parse_args(args)
            result = list_projects(dojo=self.dojo, domain=parsed_args.domain, difficulty=parsed_args.difficulty, format_output="raw")
            if result.success:
                projects = result.output
                if not projects or isinstance(projects, str):
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
        """Start a learning project and track your progress.
Usage: start <project_id> [--student <student_name>]"""
        parser = argparse.ArgumentParser(prog="start", add_help=False)
        parser.add_argument("project_id")
        parser.add_argument("--student")
        try:
            parsed_args = parser.parse_args(args)
            student_id = parsed_args.student or self.current_student_id
            if not student_id:
                self.console.print("Student ID is required. Use 'set-student <name>' or specify with --student.", style="danger")
                return

            result = start_project(dojo=self.dojo, project_id=parsed_args.project_id, student_id=student_id)
            if result.success:
                self.current_project_id = parsed_args.project_id
                self.current_student_id = student_id
                self.prompt_text = f"({self.current_project_id}) > "
                self.console.print(result.output)
            else:
                self.console.print(f"Error: {result.error_message}", style="danger")
        except SystemExit:
            self.console.print(parser.format_help(), style="warning")

    def _explain_command(self, args: List[str]):
        """Get detailed explanations of data science concepts.
Usage: explain <concept_id>

Example concepts: missing_values, outliers, normalization, encoding"""
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
        """View your learning progress, XP, and achievements.
Usage: progress [--project <project_id>]"""
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
        """Mark a project step as completed to track progress.
Usage: complete-step <step_id>"""
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
        """Practice a specific data science concept with guided exercises.
Usage: practice <concept_id>"""
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

    def _generate_data_command(self, args: List[str]):
        """Generate synthetic datasets for learning.
Usage: generate-data [--domain <healthcare|ecommerce|finance>] [--size <small|medium|large>] [--output <path>]"""
        parser = argparse.ArgumentParser(prog="generate-data", add_help=False)
        parser.add_argument("--domain", choices=["healthcare", "ecommerce", "finance"], help="Specific domain to generate")
        parser.add_argument("--size", choices=["small", "medium", "large"], default="medium", help="Dataset size")
        parser.add_argument("--output", default="datasets", help="Output directory")
        
        try:
            parsed_args = parser.parse_args(args)
            
            self.console.print("🚀 [title]Generating Synthetic Data[/title]", style="bold")
            
            generator = SyntheticDataGenerator(seed=42)
            
            # Size configurations
            size_configs = {
                "small": {"patients": 500, "transactions": 2000, "bank_txns": 1000, "credit_apps": 500},
                "medium": {"patients": 1000, "transactions": 5000, "bank_txns": 3000, "credit_apps": 1000},
                "large": {"patients": 2000, "transactions": 10000, "bank_txns": 8000, "credit_apps": 2000}
            }
            config = size_configs[parsed_args.size]
            
            output_path = Path(parsed_args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if parsed_args.domain:
                # Generate specific domain
                if parsed_args.domain == "healthcare":
                    self.console.print("🏥 [bold cyan]Generating Healthcare datasets...[/bold cyan]")
                    patients = generator.generate_patient_demographics(config["patients"])
                    lab_results = generator.generate_lab_results(config["patients"] * 3, patients['patient_id'].tolist())
                    
                    healthcare_path = output_path / "healthcare"
                    healthcare_path.mkdir(exist_ok=True, parents=True)
                    
                    patient_file = healthcare_path / "patient_demographics.csv"
                    lab_file = healthcare_path / "lab_results.csv"
                    
                    patients.to_csv(patient_file, index=False)
                    lab_results.to_csv(lab_file, index=False)
                    
                    self.console.print(f"\n✅ [bold green]Healthcare datasets created:[/bold green]")
                    self.console.print(f"   📄 {patient_file.name} ({len(patients):,} rows)")
                    self.console.print(f"   📄 {lab_file.name} ({len(lab_results):,} rows)")
                    self.console.print(f"   📁 Location: {patient_file.parent.resolve()}")
                    
                elif parsed_args.domain == "ecommerce":
                    self.console.print("🛒 [bold cyan]Generating E-commerce datasets...[/bold cyan]")
                    customers = generator.generate_customers(config["patients"])
                    transactions = generator.generate_transactions(config["transactions"], customers['customer_id'].tolist())
                    
                    ecommerce_path = output_path / "ecommerce"
                    ecommerce_path.mkdir(exist_ok=True, parents=True)
                    
                    customer_file = ecommerce_path / "customers_messy.csv"
                    transaction_file = ecommerce_path / "transactions.csv"
                    
                    customers.to_csv(customer_file, index=False)
                    transactions.to_csv(transaction_file, index=False)
                    
                    self.console.print(f"\n✅ [bold green]E-commerce datasets created:[/bold green]")
                    self.console.print(f"   📄 {customer_file.name} ({len(customers):,} rows)")
                    self.console.print(f"   📄 {transaction_file.name} ({len(transactions):,} rows)")
                    self.console.print(f"   📁 Location: {customer_file.parent.resolve()}")
                    
                elif parsed_args.domain == "finance":
                    self.console.print("💰 [bold cyan]Generating Finance datasets...[/bold cyan]")
                    bank_txns = generator.generate_bank_transactions(config["bank_txns"])
                    credit_apps = generator.generate_credit_applications(config["credit_apps"])
                    
                    finance_path = output_path / "finance"
                    finance_path.mkdir(exist_ok=True, parents=True)
                    
                    bank_file = finance_path / "bank_transactions.csv"
                    credit_file = finance_path / "credit_applications.csv"
                    
                    bank_txns.to_csv(bank_file, index=False)
                    credit_apps.to_csv(credit_file, index=False)
                    
                    self.console.print(f"\n✅ [bold green]Finance datasets created:[/bold green]")
                    self.console.print(f"   📄 {bank_file.name} ({len(bank_txns):,} rows)")
                    self.console.print(f"   📄 {credit_file.name} ({len(credit_apps):,} rows)")
                    self.console.print(f"   📁 Location: {bank_file.parent.resolve()}")
                    
            else:
                # Generate all domains
                self.console.print("🌐 [bold cyan]Generating datasets for ALL domains...[/bold cyan]\n")
                
                # Healthcare
                self.console.print("🏥 Healthcare...")
                patients = generator.generate_patient_demographics(config["patients"])
                lab_results = generator.generate_lab_results(config["patients"] * 3, patients['patient_id'].tolist())
                healthcare_path = output_path / "healthcare"
                healthcare_path.mkdir(exist_ok=True, parents=True)
                patients.to_csv(healthcare_path / "patient_demographics.csv", index=False)
                lab_results.to_csv(healthcare_path / "lab_results.csv", index=False)
                
                # E-commerce
                self.console.print("🛒 E-commerce...")
                customers = generator.generate_customers(config["patients"])
                transactions = generator.generate_transactions(config["transactions"], customers['customer_id'].tolist())
                ecommerce_path = output_path / "ecommerce"
                ecommerce_path.mkdir(exist_ok=True, parents=True)
                customers.to_csv(ecommerce_path / "customers_messy.csv", index=False)
                transactions.to_csv(ecommerce_path / "transactions.csv", index=False)
                
                # Finance
                self.console.print("💰 Finance...")
                bank_txns = generator.generate_bank_transactions(config["bank_txns"])
                credit_apps = generator.generate_credit_applications(config["credit_apps"])
                finance_path = output_path / "finance"
                finance_path.mkdir(exist_ok=True, parents=True)
                bank_txns.to_csv(finance_path / "bank_transactions.csv", index=False)
                credit_apps.to_csv(finance_path / "credit_applications.csv", index=False)
                
                self.console.print(f"\n✅ [bold green]All datasets created:[/bold green]")
                self.console.print(f"   🏥 Healthcare: patient_demographics.csv, lab_results.csv")
                self.console.print(f"   🛒 E-commerce: customers_messy.csv, transactions.csv")
                self.console.print(f"   💰 Finance: bank_transactions.csv, credit_applications.csv")
                self.console.print(f"   📁 Location: {output_path.resolve()}")
            
            self.console.print(f"\n🎉 [success]Generation completed![/success]")
            self.console.print(f"💡 [info]Tip: Use 'profile-data --file <path>' to analyze the datasets[/info]")
            
        except SystemExit:
            self.console.print(parser.format_help(), style="warning")
        except Exception as e:
            self.console.print(f"❌ Error generating data: {str(e)}", style="danger")

    def _profile_data_command(self, args: List[str]):
        """Profile and analyze datasets with AI-powered insights.
Usage: profile-data --file <path> [--output <report_path>] [--format <text|json>]"""
        parser = argparse.ArgumentParser(prog="profile-data", add_help=False)
        parser.add_argument("--file", required=True, help="Path to CSV file to profile")
        parser.add_argument("--output", help="Output path for report")
        parser.add_argument("--format", choices=["text", "json"], default="text", help="Report format")
        
        try:
            parsed_args = parser.parse_args(args)
            
            # Resolve path properly, handling relative paths
            file_path = Path(parsed_args.file).resolve()
            if not file_path.exists():
                self.console.print(f"❌ File not found: {file_path}", style="danger")
                self.console.print(f"💡 Current directory: {Path.cwd()}", style="info")
                return
                
            if not file_path.suffix.lower() == '.csv':
                self.console.print("❌ Only CSV files are supported", style="danger")
                return
            
            self.console.print(f"🔍 [title]Profiling Data: {file_path.name}[/title]")
            
            # Load and profile the data
            try:
                df = pd.read_csv(file_path)
                self.console.print(f"📊 Loaded {len(df):,} rows × {len(df.columns)} columns")
                
                profiler = IntelligentProfiler()
                profile = profiler.profile_dataset(df, file_path.stem)
                
                # Generate report
                if parsed_args.format == "json":
                    output_path = parsed_args.output or f"{file_path.stem}_profile.json"
                    profiler.export_profile_json(profile, output_path)
                else:
                    report = profiler.generate_report(profile, parsed_args.output)
                    if not parsed_args.output:
                        self.console.print(report)
                    else:
                        self.console.print(f"📄 Report saved to: {parsed_args.output}")
                
                # Show quick summary in console
                quality_emoji = "🟢" if profile.overall_quality_score > 0.8 else "🟡" if profile.overall_quality_score > 0.6 else "🔴"
                self.console.print(f"\n{quality_emoji} [title]Overall Quality Score: {profile.overall_quality_score:.1%}[/title]")
                
                if profile.recommendations:
                    self.console.print(f"\n🚀 [title]Top Recommendations:[/title]")
                    for i, rec in enumerate(profile.recommendations[:3], 1):
                        self.console.print(f"   {i}. {rec}", style="info")
                        
            except Exception as e:
                self.console.print(f"❌ Error loading CSV: {str(e)}", style="danger")
                
        except SystemExit:
            self.console.print(parser.format_help(), style="warning")
        except Exception as e:
            self.console.print(f"❌ Error profiling data: {str(e)}", style="danger")

    def _list_datasets_command(self, args: List[str]):
        """List all available datasets in the workspace.
Usage: list-datasets [--domain <healthcare|ecommerce|finance>] [--format <table|json|paths>] [--min-size <MB>]"""
        parser = argparse.ArgumentParser(prog="list-datasets", add_help=False)
        parser.add_argument("--domain", choices=["healthcare", "ecommerce", "finance"], help="Filter by domain")
        parser.add_argument("--format", choices=["table", "json", "paths"], default="table", help="Output format")
        parser.add_argument("--min-size", type=float, help="Minimum file size in MB")
        
        try:
            parsed_args = parser.parse_args(args)
            
            result = list_datasets(
                domain_filter=parsed_args.domain,
                min_size_mb=parsed_args.min_size,
                format_type=parsed_args.format
            )
            
            if result.success:
                self.console.print(result.output)
            else:
                self.console.print(f"❌ Error: {result.error_message}", style="danger")
                
        except SystemExit:
            self.console.print(parser.format_help(), style="warning")
        except Exception as e:
            self.console.print(f"❌ Error listing datasets: {str(e)}", style="danger")

    def _profile_all_command(self, args: List[str]):
        """Profile all datasets in the workspace and generate summary reports.
Usage: profile-all [--domain <healthcare|ecommerce|finance>] [--output-dir <path>] [--format <text|json>]"""
        parser = argparse.ArgumentParser(prog="profile-all", add_help=False)
        parser.add_argument("--domain", choices=["healthcare", "ecommerce", "finance"], help="Filter by domain")
        parser.add_argument("--output-dir", default="profiles", help="Output directory for reports")
        parser.add_argument("--format", choices=["text", "json"], default="text", help="Report format")
        
        try:
            parsed_args = parser.parse_args(args)
            
            self.console.print("🔍 [title]Profiling All Datasets[/title]", style="bold")
            
            # Discover datasets
            datasets = discover_datasets()
            
            if parsed_args.domain:
                datasets = [d for d in datasets if d.domain.lower() == parsed_args.domain.lower()]
            
            if not datasets:
                self.console.print("No datasets found to profile.", style="warning")
                return
            
            # Create output directory
            output_path = Path(parsed_args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            profiled_count = 0
            total_datasets = len(datasets)
            
            self.console.print(f"📊 Found {total_datasets} datasets to profile...")
            
            for i, dataset_info in enumerate(datasets, 1):
                try:
                    self.console.print(f"[{i}/{total_datasets}] Profiling {dataset_info.name}...")
                    
                    # Load and profile dataset
                    df = pd.read_csv(dataset_info.path)
                    profiler = IntelligentProfiler()
                    profile = profiler.profile_dataset(df, dataset_info.name)
                    
                    # Generate report
                    if parsed_args.format == "json":
                        output_file = output_path / f"{dataset_info.name.replace('.csv', '_profile.json')}"
                        profiler.export_profile_json(profile, str(output_file))
                    else:
                        output_file = output_path / f"{dataset_info.name.replace('.csv', '_profile.txt')}"
                        profiler.generate_report(profile, str(output_file))
                    
                    # Show quick summary
                    quality_emoji = "🟢" if profile.overall_quality_score > 0.8 else "🟡" if profile.overall_quality_score > 0.6 else "🔴"
                    self.console.print(f"   {quality_emoji} Quality: {profile.overall_quality_score:.1%} | {profile.shape[0]:,} rows × {profile.shape[1]} cols")
                    
                    profiled_count += 1
                    
                except Exception as e:
                    self.console.print(f"   ❌ Failed to profile {dataset_info.name}: {str(e)}", style="danger")
                    continue
            
            # Generate summary report
            summary_file = output_path / "summary_report.txt"
            self._generate_summary_report(datasets, summary_file)
            
            self.console.print(f"\n🎉 [success]Profiling completed![/success]")
            self.console.print(f"📁 Reports saved to: {output_path.absolute()}")
            self.console.print(f"✅ Successfully profiled: {profiled_count}/{total_datasets} datasets")
            
        except SystemExit:
            self.console.print(parser.format_help(), style="warning")
        except Exception as e:
            self.console.print(f"❌ Error during batch profiling: {str(e)}", style="danger")

    def _generate_summary_report(self, datasets, output_file):
        """Generate a summary report of all datasets."""
        try:
            lines = []
            lines.append("=" * 80)
            lines.append("📊 DATADOJO WORKSPACE SUMMARY REPORT")
            lines.append("=" * 80)
            lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
            
            # Group by domain
            domains = {}
            total_size = 0
            total_rows = 0
            
            for ds in datasets:
                if ds.domain not in domains:
                    domains[ds.domain] = []
                domains[ds.domain].append(ds)
                total_size += ds.size_mb
                total_rows += ds.rows
            
            lines.append("🏷️  DATASETS BY DOMAIN")
            lines.append("-" * 40)
            for domain, domain_datasets in domains.items():
                lines.append(f"{domain.upper()}: {len(domain_datasets)} datasets")
                domain_size = sum(ds.size_mb for ds in domain_datasets)
                domain_rows = sum(ds.rows for ds in domain_datasets)
                lines.append(f"  • Total size: {domain_size:.1f}MB")
                lines.append(f"  • Total rows: {domain_rows:,}")
                lines.append("")
            
            lines.append(f"📈 WORKSPACE TOTALS")
            lines.append(f"  • Total datasets: {len(datasets)}")
            lines.append(f"  • Total size: {total_size:.1f}MB")  
            lines.append(f"  • Total rows: {total_rows:,}")
            lines.append("")
            
            lines.append("📋 DATASET DETAILS")
            lines.append("-" * 40)
            for ds in sorted(datasets, key=lambda x: (x.domain, x.name)):
                lines.append(f"• {ds.name}")
                lines.append(f"  Domain: {ds.domain} | Rows: {ds.rows:,} | Size: {ds.size_mb:.1f}MB")
                lines.append(f"  Path: {ds.path}")
                lines.append("")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                
        except Exception as e:
            self.console.print(f"Warning: Could not generate summary report: {str(e)}", style="warning")

    def run(self):
        self.console.print(ASCII_ART)
        self.console.print(Panel("Welcome to the [title]DataDojo Interactive Session[/title]!\nType 'help' for commands or 'exit' to quit.", border_style="border"))
        while True:
            try:
                user_input = prompt(self.prompt_text, completer=self.completer)
                if not user_input.strip():
                    continue
                # Use shlex.split with posix=False for Windows path compatibility
                parts = shlex.split(user_input, posix=False)
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
