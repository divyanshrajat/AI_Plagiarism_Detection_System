from pathlib import Path
from typing import Iterable
import os


def iter_reference_documents(dataset_root: Path) -> Iterable[tuple[Path, str]]:
    """
    Iterate over all reference documents in the dataset directory.
    Now includes ALL subdirectories for maximum coverage.
    """
    if not dataset_root.exists():
        return

    # Get all subdirectories in dataset_root
    all_dirs = [dataset_root]
    if dataset_root.is_dir():
        for item in dataset_root.iterdir():
            if item.is_dir():
                all_dirs.append(item)
    
    # Also check parent directory for any sibling datasets
    parent = dataset_root.parent
    if parent.exists() and parent != dataset_root:
        for item in parent.iterdir():
            if item.is_dir() and item != dataset_root:
                all_dirs.append(item)

    # Now iterate through all directories
    for dir_path in all_dirs:
        for file_path in dir_path.rglob("*"):
            # Skip hidden files and directories
            if any(part.startswith(".") for part in file_path.parts):
                continue
            # Only process text files
            if file_path.suffix.lower() not in {".txt", ".docx", ".pdf"}:
                continue
            # Skip files that are too large (> 1MB)
            try:
                if file_path.stat().st_size > 1024 * 1024:
                    continue
            except (OSError, FileNotFoundError):
                continue
            yield file_path, file_path.suffix.lower()
