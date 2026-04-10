"""Flask application entrypoint for the plagiarism detection service."""

from importlib import import_module
from pathlib import Path
import re
from datetime import datetime
import shutil
import json

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_from_directory,
)
from werkzeug.utils import secure_filename
from modules import user_store

# Optional OCR libs
try:
    import pytesseract
except (ImportError, ModuleNotFoundError):
    pytesseract = None

try:
    import pdf2image
except (ImportError, ModuleNotFoundError):
    pdf2image = None

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
DATASET_DIR = BASE_DIR / "datasets"
UPLOAD_INDEX = UPLOAD_DIR / "index.json"

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.secret_key = "dev-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.config["SESSION_PERMANENT"] = False

# ---------- local fallbacks ----------
def _default_allowed_file(filename: str) -> bool:
    if not filename or "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in {"pdf", "docx", "txt"}

def _default_save_uploaded_file(upload, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = secure_filename(upload.filename or "uploaded_file.txt")
    destination = upload_dir / safe_name
    if destination.exists():
        stem, suffix = destination.stem, destination.suffix
        counter = 1
        while destination.exists():
            destination = upload_dir / f"{stem}_{counter}{suffix}"
            counter += 1
    upload.save(destination)
    return destination

def _default_extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""

def _default_clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.,;:!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _default_detect_ai_assistance(_text: str) -> dict:
    return {
        "probability": 0.0,
        "label": "Unavailable (fallback)",
        "explanation": "AI-assisted indicator module not available.",
    }

def _default_generate_report(
    original_filename: str,
    plagiarism_score: float,
    _top_matches: list,
    ai_indicator: dict,
    report_dir: Path,
):
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.write_text(
        "\n".join([
            "AI-Based Plagiarism Detection Report",
            f"Submitted file: {original_filename}",
            f"Overall plagiarism score: {plagiarism_score:.2f}%",
            f"AI indicator: {ai_indicator.get('label', 'N/A')}",
        ]),
        encoding="utf-8",
    )
    return report_path, None

class _DefaultSimilarityEngine:
    def __init__(self, _dataset_dir: Path):
        self.dataset_dir = _dataset_dir

    def check_plagiarism(self, _submitted_text: str):
        return 0.0, []

# ---------- compatibility resolver ----------
def _safe_import(module_name: str):
    try:
        return import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None

_upload_handler = _safe_import("modules.upload_handler")
_text_extractor = _safe_import("modules.text_extractor")
_text_preprocessor = _safe_import("modules.text_preprocessor")
_similarity_module = _safe_import("modules.semantic_similarity")
_ai_detector_module = _safe_import("modules.ai_text_detector_improved")
_report_module = _safe_import("modules.report_generator")

allowed_file = (
    getattr(_upload_handler, "allowed_file", None) if _upload_handler else None
) or _default_allowed_file

save_uploaded_file = None
if _upload_handler:
    save_uploaded_file = getattr(_upload_handler, "save_upload_file", None) or getattr(
        _upload_handler, "save_upload_file", None
    )

if save_uploaded_file is None:
    save_uploaded_file = _default_save_uploaded_file

extract_text = (
    getattr(_text_extractor, "extract_text", None) if _text_extractor else None
) or _default_extract_text

clean_text = (
    getattr(_text_preprocessor, "clean_text", None) if _text_preprocessor else None
) or _default_clean_text

detect_ai_assistance = (
    getattr(_ai_detector_module, "detect_ai_assistance", None) if _ai_detector_module else None
) or _default_detect_ai_assistance

generate_report_func = (
    getattr(_report_module, "generate_report", None) if _report_module else None
) or _default_generate_report

SimilarityEngineClass = (
    getattr(_similarity_module, "SemanticSimilarityEngine", None) if _similarity_module else None
) or _DefaultSimilarityEngine

def get_similarity_engine():
    if getattr(get_similarity_engine, "instance", None) is None:
        get_similarity_engine.instance = SimilarityEngineClass(DATASET_DIR)
    return get_similarity_engine.instance

# Custom Jinja2 filter for formatting timestamp
def format_timestamp(timestamp_str):
    """Format timestamp from YYYYMMDD_HHMMSS to readable format."""
    if not timestamp_str:
        return "N/A"
    try:
        dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        return dt.strftime("%B %d, %Y %I:%M:%S %p")
    except ValueError:
        return timestamp_str

app.jinja_env.filters["format_timestamp"] = format_timestamp

# Initialize SQLite DB
user_store.init_db()

def login_required(role: str | list | None = None):
    """Decorator to require login, optional role check."""
    def _decorator(fn):
        def _wrapped(*args, **kwargs):
            user = session.get("user")
            if not user:
                flash("Please log in to access that page.", "error")
                return redirect(url_for("login", next=request.path))
            if role:
                allowed_roles = [role] if isinstance(role, str) else role
                if user.get("role") not in allowed_roles:
                    flash("Insufficient permissions.", "error")
                    return redirect(url_for("index"))
            return fn(*args, **kwargs)
        _wrapped.__name__ = fn.__name__
        return _wrapped
    return _decorator

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        email = request.form.get("email")
        
        if not username or not password:
            flash("Username and password are required.", "error")
            return redirect(url_for("signup"))
        
        success = user_store.create_user(username, password, role="student", email=email)
        
        if success:
            flash("Account created successfully! Please log in.", "info")
            return redirect(url_for("login"))
        else:
            flash("Username already exists. Please choose a different one.", "error")
            return redirect(url_for("signup"))
    
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    next_url = request.args.get("next")
    
    if request.method == "POST":
        username = request.form.get("username")
        passwd = request.form.get("password")
        user = user_store.authenticate(username, passwd)
        if user:
            session["user"] = {"username": user["username"], "role": user["role"]}
            flash("Logged in successfully.", "info")
            
            if next_url and next_url.startswith("/"):
                return redirect(next_url)
            
            if user["role"] == "admin":
                return redirect(url_for("admin_list_users"))
            elif user["role"] == "teacher":
                return redirect(url_for("teacher_dashboard"))
            else:
                return redirect(url_for("student_dashboard"))
        
        flash("Invalid credentials", "error")
    
    return render_template("login.html", next=next_url or "")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out.", "info")
    return redirect(url_for("index"))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "document" not in request.files:
            flash("No file part", "error")
            return redirect(request.url)
        
        file = request.files["document"]
        if file.filename == "":
            flash("No selected file", "error")
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            try:
                dest_path = save_uploaded_file(file, UPLOAD_DIR)
                
                # Update uploads index
                uploads = []
                if UPLOAD_INDEX.exists():
                    try:
                        uploads = json.loads(UPLOAD_INDEX.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pass
                
                user = session.get("user")
                uploader = user["username"] if user else "anonymous"
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # --- Run analysis immediately ---
                try:
                    raw_text = extract_text(dest_path)
                    if not raw_text:
                        flash("Could not extract text from file or file is empty.", "error")

                        # Still record upload as pending
                        upload_record = {
                            "filename": dest_path.name,
                            "uploader": uploader,
                            "timestamp": timestamp,
                            "status": "pending"
                        }
                        uploads.append(upload_record)
                        UPLOAD_INDEX.parent.mkdir(parents=True, exist_ok=True)
                        UPLOAD_INDEX.write_text(json.dumps(uploads, indent=4), encoding="utf-8")

                        return redirect(request.url)

                    clean_txt = clean_text(raw_text)
                    engine = get_similarity_engine()
                    plagiarism_score, matches = engine.check_plagiarism(clean_txt)
                    ai_indicator = detect_ai_assistance(clean_txt)

                    owner = uploader
                    if user and user.get("role") in ["teacher", "admin"]:
                        owner = request.form.get("owner") or uploader

                    report_path, pdf_path = generate_report_func(
                        dest_path.name,
                        plagiarism_score,
                        matches,
                        ai_indicator,
                        REPORT_DIR,
                        owner=owner,
                    )

                    # Record upload as done with report path
                    upload_record = {
                        "filename": dest_path.name,
                        "uploader": uploader,
                        "timestamp": timestamp,
                        "status": "done",
                        "report_path": report_path.name,
                    }
                    uploads.append(upload_record)
                    UPLOAD_INDEX.parent.mkdir(parents=True, exist_ok=True)
                    UPLOAD_INDEX.write_text(json.dumps(uploads, indent=4), encoding="utf-8")

                    # Read report text for display
                    report_text = ""
                    if report_path.exists():
                        report_text = report_path.read_text(encoding="utf-8", errors="ignore")

                    return render_template(
                        "result.html",
                        filename=dest_path.name,
                        plagiarism_score=round(plagiarism_score, 2),
                        top_matches=matches,
                        ai_indicator=ai_indicator,
                        report_text=report_text,
                        pdf_available=pdf_path.name if pdf_path else None,
                    )

                except Exception as inner_e:
                    # If analysis fails, still save upload as pending
                    upload_record = {
                        "filename": dest_path.name,
                        "uploader": uploader,
                        "timestamp": timestamp,
                        "status": "pending"
                    }
                    uploads.append(upload_record)
                    UPLOAD_INDEX.parent.mkdir(parents=True, exist_ok=True)
                    UPLOAD_INDEX.write_text(json.dumps(uploads, indent=4), encoding="utf-8")

                    flash(f"File uploaded but analysis failed: {str(inner_e)}", "error")
                    if user:
                        if user.get("role") == "student":
                            return redirect(url_for("student_uploads"))
                        elif user.get("role") in ["teacher", "admin"]:
                            return redirect(url_for("teacher_dashboard"))
                    return redirect(url_for("index"))
                
            except Exception as e:
                flash(f"Error uploading file: {str(e)}", "error")
                return redirect(request.url)
        else:
            flash("Invalid file type. Allowed types: PDF, DOCX, TXT", "error")
            return redirect(request.url)
            
    return render_template("index.html")

@app.route("/student")
@login_required("student")
def student_dashboard():
    user = session.get("user")
    
    uploads = []
    if UPLOAD_INDEX.exists():
        try:
            all_uploads = json.loads(UPLOAD_INDEX.read_text(encoding="utf-8"))
            uploads = [u for u in all_uploads if u.get("uploader") == user["username"]]
        except json.JSONDecodeError:
            pass
            
    reports = []
    reports_idx = REPORT_DIR / "index.json"
    if reports_idx.exists():
        try:
            all_reports = json.loads(reports_idx.read_text(encoding="utf-8"))
            reports = [r for r in all_reports if r.get("owner") == user["username"]]
        except json.JSONDecodeError:
            pass
            
    return render_template("student_dashboard.html", user=user, uploads=uploads, reports=reports)


@app.route("/student/uploads")
@login_required("student")
def student_uploads():
    return student_dashboard() # Route mentioned in template


@app.route("/student/reports")
@login_required("student")
def student_reports():
    return student_dashboard() # Route mentioned in template


@app.route("/teacher")
@login_required(["teacher", "admin"])
def teacher_dashboard():
    user = session.get("user")
    
    uploads = []
    if UPLOAD_INDEX.exists():
        try:
            uploads = json.loads(UPLOAD_INDEX.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
            
    reports = []
    reports_idx = REPORT_DIR / "index.json"
    if reports_idx.exists():
        try:
            reports = json.loads(reports_idx.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
            
    return render_template("teacher_dashboard.html", user=user, uploads=uploads, reports=reports)

@app.route("/admin/users")
@login_required("admin")
def admin_list_users():
    user = session.get("user")
    users = user_store.list_users()
    return render_template("admin_users.html", user=user, users=users)

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/generate_report", methods=["POST"])
@app.route("/teacher/generate_report", methods=["POST"])
@login_required(["student", "teacher", "admin"])
def generate_report():
    user = session.get("user")
    filename = request.form.get("filename")
    owner = request.form.get("owner") or (user["username"] if user else "anonymous")
    
    # Determine where to redirect on error based on role
    if user and user.get("role") == "student":
        error_redirect = url_for("student_dashboard")
    else:
        error_redirect = url_for("teacher_dashboard")
    
    if not filename:
        flash("Filename is required to generate report.", "error")
        return redirect(error_redirect)
        
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        flash("Uploaded file not found.", "error")
        return redirect(error_redirect)
        
    try:
        # Extract and clean text
        raw_text = extract_text(file_path)
        if not raw_text:
            flash("Could not extract text from file or file is empty.", "error")
            return redirect(error_redirect)
            
        clean_txt = clean_text(raw_text)
        
        # Check Plagiarism
        engine = get_similarity_engine()
        plagiarism_score, matches = engine.check_plagiarism(clean_txt)
        
        # Check AI Assistance
        ai_indicator = detect_ai_assistance(clean_txt)
        
        # Generate Report (report_generator.py writes index.json internally)
        report_path, pdf_path = generate_report_func(
            filename,
            plagiarism_score,
            matches,
            ai_indicator,
            REPORT_DIR,
            owner=owner,
        )
        
        # Update upload index: mark as "done" with report path
        if UPLOAD_INDEX.exists():
            try:
                uploads = json.loads(UPLOAD_INDEX.read_text(encoding="utf-8"))
                for u in uploads:
                    if u.get("filename") == filename:
                        u["status"] = "done"
                        u["report_path"] = report_path.name
                        break
                UPLOAD_INDEX.write_text(json.dumps(uploads, indent=4), encoding="utf-8")
            except Exception:
                pass
        
        # Read report text for display
        report_text = ""
        if report_path.exists():
            report_text = report_path.read_text(encoding="utf-8", errors="ignore")
        
        # Render the beautiful result page
        return render_template(
            "result.html",
            filename=filename,
            plagiarism_score=round(plagiarism_score, 2),
            top_matches=matches,
            ai_indicator=ai_indicator,
            report_text=report_text,
            pdf_available=pdf_path.name if pdf_path else None,
        )
        
    except Exception as e:
        flash(f"Error generating report: {str(e)}", "error")
        
    return redirect(error_redirect)

@app.route("/report/<path:filename>")
def download_report(filename):
    user = session.get("user")
    if not user:
        flash("Please log in to view reports.", "error")
        return redirect(url_for("login"))
        
    # Security: check if student owns the report, teachers/admins can view all
    if user["role"] == "student":
        reports_idx = REPORT_DIR / "index.json"
        owns_report = False
        if reports_idx.exists():
            try:
                reports = json.loads(reports_idx.read_text(encoding="utf-8"))
                for r in reports:
                    if r.get("path") == filename and r.get("owner") == user["username"]:
                        owns_report = True
                        break
            except Exception:
                pass
        if not owns_report:
            flash("Unauthorized access to report.", "error")
            return redirect(url_for("student_dashboard"))
            
    return send_from_directory(REPORT_DIR, filename)

@app.route("/reset_request", methods=["GET", "POST"])
def reset_request():
    if request.method == "POST":
        email = request.form.get("email")
        if email:
            flash("Password reset link sent to your email.", "success")
        else:
            flash("Please enter an email address.", "error")
        return redirect(url_for("login"))
    return render_template("reset_request.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

