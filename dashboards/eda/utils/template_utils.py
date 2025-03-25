"""
Template utilities for the dashboard application.
Provides functions for loading and rendering HTML templates.
"""

import re
from pathlib import Path


def load_template(template_path):
    """
    Load an HTML template from file

    Parameters:
    -----------
    template_path : str or Path
        Path to the template file

    Returns:
    --------
    str
        The template content as a string
    """
    with open(template_path, "r", encoding="utf-8") as file:
        return file.read()


def render_template(template, context):
    """
    Simple template rendering function that replaces placeholders with values

    Parameters:
    -----------
    template : str
        The template string with placeholders
    context : dict
        A dictionary of placeholder names and their values

    Returns:
    --------
    str
        The rendered template

    Notes:
    ------
    Supports:
    - Variable interpolation: {{variable}}
    - Conditional blocks: {{#variable}}content{{/variable}}
    - Comments: {{!comment}}
    """
    # Remove comments
    template = re.sub(r"{{!.*?}}", "", template, flags=re.DOTALL)

    # Replace all {{variable}} placeholders with their values
    for key, value in context.items():
        placeholder = f"{{{{{key}}}}}"
        replacement = str(value) if value is not None else ""
        # Simple string replacement instead of regex to avoid escape sequence issues
        template = template.replace(placeholder, replacement)

    # Handle conditional blocks {{#variable}} content {{/variable}}
    for key, value in context.items():
        start_tag = f"{{{{#{key}}}}}"
        end_tag = f"{{{{/{key}}}}}"

        # If the value exists and is truthy, remove just the conditional markers
        if value:
            # Find all occurrences of this conditional block
            start_pos = 0
            while True:
                start_idx = template.find(start_tag, start_pos)
                if start_idx == -1:
                    break

                end_idx = template.find(end_tag, start_idx)
                if end_idx == -1:
                    break

                # Extract the content between tags
                content = template[start_idx + len(start_tag) : end_idx]

                # Replace the entire block with just the content
                template = (
                    template[:start_idx] + content + template[end_idx + len(end_tag) :]
                )

                # Update start position for next iteration
                start_pos = start_idx + len(content)
        else:
            # If the value doesn't exist or is falsy, remove the entire block
            while True:
                start_idx = template.find(start_tag)
                if start_idx == -1:
                    break

                end_idx = template.find(end_tag)
                if end_idx == -1:
                    break

                # Remove the entire block including tags
                template = template[:start_idx] + template[end_idx + len(end_tag) :]

    # Clean up any remaining template tags (useful for optional content)
    while True:
        start_idx = template.find("{{")
        if start_idx == -1:
            break

        end_idx = template.find("}}", start_idx)
        if end_idx == -1:
            break

        template = template[:start_idx] + template[end_idx + 2 :]

    return template


def get_template_path(template_name, template_dir=None):
    """
    Get the full path to a template file

    Parameters:
    -----------
    template_name : str
        The name of the template file
    template_dir : str or Path, optional
        The directory containing templates. If None, uses the default 'templates' directory

    Returns:
    --------
    Path
        Full path to the template file
    """
    if template_dir is None:
        # Navigate to the templates directory relative to this file
        # Going up to utils directory, then up to the project root, then to templates
        template_dir = Path(__file__).parent.parent / "templates"
    else:
        template_dir = Path(template_dir)

    return template_dir / template_name
