import os
from pathlib import Path


def get_last_modified_folder(base_dir: str) -> str:
    """
    Returns the absolute path of the last modified subdirectory in base_dir.
    """
    # Use Path for modern path handling
    base_path = Path(base_dir)

    # Get all subdirectories, excluding files
    subdirs = [p for p in base_path.iterdir() if p.is_dir()]

    if not subdirs:
        # Return the base directory itself or raise an error if empty
        return str(base_path.resolve())

    # Find the directory with the maximum modification time (mtime)
    latest_dir = max(subdirs, key=os.path.getmtime)

    # Return the absolute path
    print("latest_dir", str(latest_dir.resolve()))
    return str(latest_dir.resolve())

