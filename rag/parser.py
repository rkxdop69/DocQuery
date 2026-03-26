"""parser.py — Extract raw text from PDF or TXT files."""

def extract_text(filepath: str) -> str:
    if filepath.endswith(".pdf"):
        import fitz  # pip install pymupdf
        doc = fitz.open(filepath)
        return "\n".join(page.get_text() for page in doc)
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
