# Summarizationcode.py - Tender Analyzer

## Overview
This is a Streamlit-based Tender Analyzer application that uses LLM-driven semantic retrieval with map/reduce pattern for processing tender documents.

## Running the Application

### Method 1: Using the startup script (Recommended)
```bash
./run_summarization.sh
```

### Method 2: Direct Streamlit command
```bash
streamlit run Summarizationcode.py
```

### Method 3: With explicit configuration
```bash
streamlit run Summarizationcode.py \
    --server.fileWatcherType=auto \
    --runner.fastReruns=true \
    --server.address=0.0.0.0 \
    --server.port=8501
```

## Troubleshooting Code Changes Not Reflecting in UI

If you make changes to `Summarizationcode.py` and they don't appear in the UI:

### 1. Clear Python Cache
```bash
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete
```

### 2. Restart Streamlit
Press `Ctrl+C` to stop the Streamlit server, then restart it using one of the methods above.

### 3. Clear Browser Cache
- Press `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac) to hard refresh the page
- Or use Streamlit's "Rerun" button in the UI (top-right corner)

### 4. Check File Watcher
Ensure the file watcher is enabled in `.streamlit/config.toml`:
```toml
[server]
fileWatcherType = "auto"

[runner]
fastReruns = true
```

### 5. Verify File Name
The file must be named exactly `Summarizationcode.py` (capital S). Case sensitivity matters on Linux systems.

## Configuration

The application uses the configuration file `.streamlit/config.toml` which includes:
- Automatic file watching for code changes
- Fast reruns when changes are detected
- Proper server configuration for development

## Requirements

Install dependencies before running:
```bash
pip install -r requirements.txt
```

Key dependencies:
- streamlit
- pymupdf
- nltk
- google-genai
- pdfplumber
- docx
- pandas

## Environment Variables

Set your Google API key:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or enter it in the sidebar when the app is running.

## Features

- Multiple PDF upload support
- Semantic chunking with map/reduce
- Multiple analysis modes:
  - General Summary
  - Compliance Matrix
  - Risk Assessment
  - Entity Dashboard
  - Ambiguity Scrutiny
- Vision support for charts/images
- Benchmarking with CSV evaluation
- **Confidence Score System** (NEW): Provides quality metrics for each result
  - Snippet Coverage
  - Result Coherence
  - Information Density
  - Citation Quality
  - See [CONFIDENCE_SCORE_README.md](./CONFIDENCE_SCORE_README.md) for details
- Export to TXT, JSON, DOCX, and XLSX formats

## Port

The application runs on port 8501 by default. Access it at:
```
http://localhost:8501
```

## Development Notes

- The file uses `@st.cache_resource` for NLTK setup
- Changes to cached functions require a manual cache clear or app restart
- Use Streamlit's "Clear cache" option from the hamburger menu if needed
