"""CLI command for explaining data preprocessing concepts."""

from typing import Optional

class CLIResult:
    def __init__(self, success: bool, output: str, exit_code: int, error_message: Optional[str] = None):
        self.success = success
        self.output = output
        self.exit_code = exit_code
        self.error_message = error_message


def explain_concept(
    dojo,
    concept_id: str,
    detail_level: str = "basic",
    include_examples: bool = False
) -> CLIResult:
    """Explain a data preprocessing concept.

    Args:
        dojo: Dojo instance
        concept_id: Concept to explain
        detail_level: Explanation depth (basic|detailed|expert)
        include_examples: Include code examples

    Returns:
        CLI result with concept explanation
    """
    try:
        if not concept_id:
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message="Concept ID is required"
            )

        educational = dojo.get_educational_interface()

        # Get concept explanation
        try:
            explanation = educational.get_concept_explanation(concept_id)
        except KeyError as e:
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message=str(e)
            )

        # Format output based on detail level
        output_lines = []
        output_lines.append("=" * 80)
        output_lines.append(f"Concept: {explanation['title']}")
        output_lines.append("=" * 80)
        output_lines.append(f"ID: {explanation['concept_id']}")
        output_lines.append(f"Difficulty Level: {explanation['difficulty_level']}")
        output_lines.append("")
        output_lines.append("Explanation:")
        output_lines.append(_wrap_text(explanation['explanation'], 78))
        output_lines.append("")

        # Add analogies for basic and detailed levels
        if detail_level in ["basic", "detailed"] and explanation.get('analogies'):
            output_lines.append("Real-World Analogies:")
            for i, analogy in enumerate(explanation['analogies'], 1):
                output_lines.append(f"  {i}. {_wrap_text(analogy, 75, indent='     ')}")
            output_lines.append("")

        # Add examples if requested or detail level is detailed/expert
        if (include_examples or detail_level in ["detailed", "expert"]) and explanation.get('examples'):
            output_lines.append("Examples:")
            for i, example in enumerate(explanation['examples'], 1):
                output_lines.append(f"  Example {i}:")
                for line in example.split('\n'):
                    output_lines.append(f"    {line}")
                output_lines.append("")

        # Add related concepts for detailed and expert levels
        if detail_level in ["detailed", "expert"] and explanation.get('related_concepts'):
            output_lines.append("Related Concepts:")
            for concept in explanation['related_concepts']:
                output_lines.append(f"  • {concept}")
            output_lines.append("")

        output_lines.append("=" * 80)

        return CLIResult(
            success=True,
            output="\n".join(output_lines),
            exit_code=0
        )

    except Exception as e:
        return CLIResult(
            success=False,
            output="",
            exit_code=1,
            error_message=f"Failed to explain concept: {str(e)}"
        )


def _wrap_text(text: str, width: int = 78, indent: str = "") -> str:
    """Wrap text to specified width with optional indent.

    Args:
        text: Text to wrap
        width: Maximum line width
        indent: String to prepend to continuation lines

    Returns:
        Wrapped text
    """
    if not text:
        return ""

    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_length = len(word)
        # +1 for the space before the word
        if current_length + word_length + 1 > width and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(indent) + word_length
        else:
            current_line.append(word)
            current_length += word_length + (1 if current_line else 0)

    if current_line:
        lines.append(" ".join(current_line))

    # Add indent to continuation lines
    if indent and len(lines) > 1:
        return lines[0] + "\n" + "\n".join(indent + line for line in lines[1:])

    return "\n".join(lines)
