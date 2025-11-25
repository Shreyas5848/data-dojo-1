from abc import ABC, abstractmethod
from typing import List, Optional

class CLIResult:
    def __init__(self, success: bool, output: str, exit_code: int, error_message: Optional[str] = None):
        self.success = success
        self.output = output
        self.exit_code = exit_code
        self.error_message = error_message

class CLIInterface(ABC):
    @abstractmethod
    def list_projects_cmd(self, domain: Optional[str] = None, difficulty: Optional[str] = None, format_output: str = "table") -> CLIResult:
        pass

    @abstractmethod
    def start_project_cmd(self, project_id: str, student_id: str, guidance_level: str = "detailed", interactive: bool = True) -> CLIResult:
        pass

    @abstractmethod
    def pipeline_cmd(self, operations: List[str], input_file: str, output_file: Optional[str] = None, config_file: Optional[str] = None, educational_mode: bool = False) -> CLIResult:
        pass

    @abstractmethod
    def show_progress_cmd(self, student_id: str, project_id: Optional[str] = None, format_output: str = "summary") -> CLIResult:
        pass

    @abstractmethod
    def explain_concept_cmd(self, concept_id: str, detail_level: str = "basic", include_examples: bool = False) -> CLIResult:
        pass

    @abstractmethod
    def validate_data_cmd(self, data_file: str, validation_rules: Optional[str] = None, report_format: str = "summary") -> CLIResult:
        pass

    @abstractmethod
    def create_project_cmd(self, name: str, domain: str, difficulty: str, dataset_path: str) -> CLIResult:
        pass

class CLI(CLIInterface):
    def list_projects_cmd(self, domain: Optional[str] = None, difficulty: Optional[str] = None, format_output: str = "table") -> CLIResult:
        return CLIResult(success=True, output="projects", exit_code=0)

    def start_project_cmd(self, project_id: str, student_id: str, guidance_level: str = "detailed", interactive: bool = True) -> CLIResult:
        if not project_id or not student_id:
            return CLIResult(success=False, output="", exit_code=1, error_message="Missing required arguments")
        return CLIResult(success=True, output="started", exit_code=0)

    def pipeline_cmd(self, operations: List[str], input_file: str, output_file: Optional[str] = None, config_file: Optional[str] = None, educational_mode: bool = False) -> CLIResult:
        if not operations or not input_file:
            return CLIResult(success=False, output="", exit_code=1, error_message="Missing required arguments")
        return CLIResult(success=True, output="pipeline", exit_code=0)

    def show_progress_cmd(self, student_id: str, project_id: Optional[str] = None, format_output: str = "summary") -> CLIResult:
        return CLIResult(success=True, output="progress", exit_code=0)

    def explain_concept_cmd(self, concept_id: str, detail_level: str = "basic", include_examples: bool = False) -> CLIResult:
        return CLIResult(success=True, output="explanation", exit_code=0)

    def validate_data_cmd(self, data_file: str, validation_rules: Optional[str] = None, report_format: str = "summary") -> CLIResult:
        return CLIResult(success=True, output="validation", exit_code=0)
    
    def create_project_cmd(self, name: str, domain: str, difficulty: str, dataset_path: str) -> CLIResult:
        return CLIResult(success=True, output="created", exit_code=0)
