"""Utilities for validating and saving uploaded files."""
# pylint: disable=duplicate-code

from pathlib import Path
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}


def allowed_file(filename: str) -> bool:
    """Return True only for supported document extensions."""
    if not filename or "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(upload: FileStorage, upload_dir: Path) -> Path:
    """Save an uploaded file safely and return the destination path."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = secure_filename(upload.filename or "uploaded_file.txt")
    destination = upload_dir / safe_name

    # Avoid collisions by appending a number.
    if destination.exists():
        stem, suffix = destination.stem, destination.suffix
        counter = 1
        while destination.exists():
            destination = upload_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    upload.save(destination)
    return destination


# Backward-compatible alias (common naming mismatch in starter templates).
def save_upload_file(upload: FileStorage, upload_dir: Path) -> Path:
    return save_uploaded_file(upload, upload_dir)


__all__ = ["ALLOWED_EXTENSIONS", "allowed_file", "save_uploaded_file", "save_upload_file"]
