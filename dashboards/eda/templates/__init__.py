# Templates package initialization
# This file is used to mark the directory as a Python package

from pathlib import Path


def get_template_path(template_name):
    """
    Get the full path to a template file

    Parameters:
    -----------
    template_name : str
        The name of the template file

    Returns:
    --------
    Path
        Full path to the template file
    """
    return Path(__file__).parent / template_name
