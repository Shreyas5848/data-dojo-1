from typing import Optional
import json
import webbrowser
from pathlib import Path

class CLIResult:
    def __init__(self, success: bool, output: str, exit_code: int, error_message: Optional[str] = None):
        self.success = success
        self.output = output
        self.exit_code = exit_code
        self.error_message = error_message

def practice(dojo, student_id: str, project_id: str, concept_id: str) -> CLIResult:
    """Starts an interactive practice session for a concept."""
    try:
        # Get concept explanation to show as an example
        educational = dojo.get_educational_interface()
        explanation = educational.get_concept_explanation(concept_id)
        project = dojo.load_project(project_id)
        dataset_path = project.dataset_path

        output = []
        output.append("="*80)
        output.append(f"PRACTICE SESSION: {explanation['title']}")
        output.append("="*80)
        output.append("\n--- [ I DO: Here's an example ] ---\n")
        
        if explanation['examples']:
            output.append("Example Code:\n")
            output.append(explanation['examples'][0])
        else:
            output.append("No code example available for this concept.")
            
        output.append("\n--- [ YOU DO: Your Turn! ] ---\n")
        
        # Get user confirmation
        while True:
            response = input("I will now create a Jupyter Notebook for you to practice. Proceed? (y/n/skip): ").lower()
            if response in ['y', 'yes']:
                notebook_path = _create_practice_notebook(project, concept_id, dataset_path)
                output.append(f"\nJupyter Notebook created at: {notebook_path}")
                output.append("Opening the notebook now...")
                webbrowser.open(f"file://{notebook_path.resolve()}")
                break
            elif response in ['n', 'no']:
                output.append("\nPractice session cancelled.")
                break
            elif response == 'skip':
                output.append("\nPractice skipped. You can come back to it later.")
                # Optionally, mark concept as 'reviewed'
                project.track_progress(student_id, f"reviewed_{concept_id}", [concept_id])
                break
            else:
                print("Invalid input. Please enter 'y', 'n', or 'skip'.")
        
        return CLIResult(success=True, output="\n".join(output), exit_code=0)

    except KeyError:
        return CLIResult(success=False, output="", exit_code=1, error_message=f"Concept '{concept_id}' not found.")
    except Exception as e:
        return CLIResult(success=False, output="", exit_code=1, error_message=str(e))

def _create_practice_notebook(project, concept_id: str, dataset_path: str) -> Path:
    """Creates a pre-configured Jupyter Notebook for a practice session."""
    
    notebook_filename = f"practice_{concept_id}.ipynb"
    notebook_path = Path(notebook_filename)

    # Define the notebook structure
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Practice Session: {project.info.name}\n\n",
                    f"## Concept: {concept_id.replace('_', ' ').title()}\n\n",
                    f"**Your Task:** Use the techniques you've learned to work with the dataset for this project. The dataset is located at `{dataset_path}`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    f"dataset_path = '{dataset_path}'\n",
                    "df = pd.read_csv(dataset_path)\n",
                    "print('Dataset loaded successfully!')\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Your Workspace\n\n",
                    "Use the cell below to write your Python code to practice the concept."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Your code here...\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Validation\n\n",
                    "Run the cell below to check your work (Note: this is a placeholder for a real validation step)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Example validation: check if missing values have been handled\n",
                    "if df.isnull().sum().sum() == 0:\n",
                    "    print('Great job! No missing values found.')\n",
                    "else:\n",
                    "    print('Looks like there are still some missing values. Keep going!')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11" # This should ideally match the user's environment
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    # Write the notebook to a file
    with open(notebook_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)
        
    return notebook_path
