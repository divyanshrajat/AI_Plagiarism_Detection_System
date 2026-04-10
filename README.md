# AI-Based Plagiarism Detection System for Academic Integrity

A Flask-based web application that detects plagiarism in submitted documents using semantic similarity and AI-generated text detection.

## Features

- **Multi-format Support**: Upload PDF, DOCX, or TXT files
- **Semantic Similarity**: Uses sentence-transformers for accurate plagiarism detection
- **AI Text Detection**: Identifies AI-assisted writing patterns
- **User Roles**: Student, Teacher, and Admin roles
- **Report Generation**: Detailed plagiarism reports with similarity scores

## Quick Start

**Note**: The app uses MongoDB for user management. It starts without MongoDB (use /signup), but login requires it.

```bash
# Create/activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Option 1: Start MongoDB (recommended for full features)
scripts/start_mongo.bat

# Option 2: Manual MongoDB
# Download: https://www.mongodb.com/try/download/community
# Run: mongod --dbpath ./data/db

# In NEW terminal: Run app
python app.py
```

Open `http://localhost:5000`. Use `/signup` to create users if no defaults.

## Default Login Credentials

| Role | Username | Password |
|------|----------|----------|
| Student | student1 | studentpass |
| Teacher | teacher1 | teacherpass |
| Admin | admin | adminpass |

## Project Structure

```
AI_Plagiarism_Detection_System/
├── app.py                      # Main Flask application
├── modules/
│   ├── ai_text_detector.py     # AI-generated text detection
│   ├── dataset_loader.py       # Dataset loading utilities
│   ├── index_utils.py          # Index management utilities
│   ├── report_generator.py     # Report generation
│   ├── semantic_similarity.py # Semantic similarity scoring
│   ├── text_extractor.py       # PDF, DOCX, TXT extraction
│   ├── text_preprocessor.py    # Text cleaning/normalization
│   ├── upload_handler.py        # Upload validation and saving
│   └── user_store.py            # User authentication & management
├── templates/                  # HTML templates
├── statics/                    # CSS and static assets
├── scripts/
│   ├── train_final_fixed.py    # Model training script
│   └── index_dataset.py         # Dataset indexing script
├── tests/                      # Test files
├── datasets/                   # Reference corpora
├── models/                    # Trained models
├── data/                      # User database
├── reports/                   # Generated reports
└── uploads/                   # Uploaded files
```

## Key Modules

| Module | Description |
|--------|-------------|
| `app.py` | Flask web app, routing, and end-to-end workflow |
| `semantic_similarity.py` | Semantic similarity scoring with fallback mechanism |
| `ai_text_detector.py` | Detects AI-assisted writing patterns |
| `text_extractor.py` | Extracts text from PDF, DOCX, and TXT files |
| `text_preprocessor.py` | Cleans and normalizes text |
| `report_generator.py` | Generates plagiarism reports |

## Training the Model

To train or fine-tune the model:

```bash
python scripts/train_final_fixed.py
```

To build dataset embeddings:

```bash
python scripts/index_dataset.py
```

## Running Tests

```bash
pytest tests/
```

## OCR Support (Optional)

The app supports OCR for scanned/image PDFs using `pytesseract` + `pdf2image`.

**Requirements:**
- Tesseract OCR engine
- Poppler for PDF rendering

**Installation (Windows):**
1. Download and install Tesseract: https://github.com/tesseract-ocr/tesseract
2. Add Tesseract `bin` to PATH
3. Download Poppler: https://github.com/oschwartz10612/poppler-windows/releases
4. Add Poppler `bin` to PATH

If not installed, the app will use non-OCR extractors.

## Important Notes

- AI-assisted indicator is supportive and not definitive proof of plagiarism
- If sentence-transformers model download fails, the app falls back to token cosine similarity
- Reports are stored in the `reports/` directory
- Uploaded files are stored in the `uploads/` directory

