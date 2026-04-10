import io
import json
from pathlib import Path

import pytest

from app import app, UPLOAD_DIR, UPLOAD_INDEX, REPORT_DIR


@pytest.fixture(autouse=True)
def clean_workspace(tmp_path, monkeypatch):
    # Use real paths but ensure a clean state before each test
    # Remove index files if present
    if UPLOAD_INDEX.exists():
        UPLOAD_INDEX.unlink()
    idx = REPORT_DIR / "index.json"
    if idx.exists():
        idx.unlink()
    # Ensure folders exist
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    yield


def login(client, username, password):
    return client.post("/login", data={"username": username, "password": password}, follow_redirects=True)


def test_student_upload_and_teacher_generate_report():
    # Student uploads
    student_client = app.test_client()
    r = login(student_client, "student1", "studentpass")
    assert r.status_code == 200

    data = {
        "document": (io.BytesIO(b"This is a test document for pytest."), "pytest_sample.txt")
    }
    r = student_client.post("/", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert r.status_code in (200, 302)

    # Verify upload index created and contains student entry
    assert UPLOAD_INDEX.exists(), "uploads/index.json should exist"
    uploads = json.loads(UPLOAD_INDEX.read_text(encoding="utf-8") or "[]")
    assert any(u.get("uploader") == "student1" for u in uploads)
    filename = uploads[0].get("filename")
    assert filename

    # Teacher generates report
    teacher_client = app.test_client()
    r = login(teacher_client, "teacher1", "teacherpass")
    assert r.status_code == 200

    r = teacher_client.post("/teacher/generate_report", data={"filename": filename, "owner": "student1"}, follow_redirects=True)
    assert r.status_code in (200, 302)

    # Verify report index updated
    reports_idx = REPORT_DIR / "index.json"
    assert reports_idx.exists(), "reports/index.json should exist"
    reports = json.loads(reports_idx.read_text(encoding="utf-8") or "[]")
    assert any(r.get("owner") == "student1" for r in reports)

    # Student dashboard shows report link
    r = login(student_client, "student1", "studentpass")
    r = student_client.get("/student")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    # Ensure report filename appears in dashboard HTML
    assert reports[0]["path"] in html
