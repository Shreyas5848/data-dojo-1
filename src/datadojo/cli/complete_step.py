from typing import Optional

class CLIResult:
    def __init__(self, success: bool, output: str, exit_code: int, error_message: Optional[str] = None):
        self.success = success
        self.output = output
        self.exit_code = exit_code
        self.error_message = error_message

def complete_step(dojo, student_id: str, project_id: str, step_id: str) -> CLIResult:
    """Marks a step as complete for a student in a project."""
    try:
        project = dojo.load_project(project_id)
        project.track_progress(student_id, step_id, [])
        return CLIResult(success=True, output=f"Step '{step_id}' marked as complete.", exit_code=0)
    except Exception as e:
        return CLIResult(success=False, output="", exit_code=1, error_message=str(e))
