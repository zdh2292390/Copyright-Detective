import io
from pathlib import Path

import PyPDF2

def _reset_stream(stream):
    try:
        stream.seek(0)
    except Exception:
        pass


def _read_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return b""

    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()

    read_method = getattr(uploaded_file, "read", None)
    if callable(read_method):
        seek_method = getattr(uploaded_file, "seek", None)
        try:
            if callable(seek_method):
                seek_method(0)
        except Exception:
            pass
        data = read_method()
        return data

    return uploaded_file


def extract_text_from_pdf(pdf_file):
    """
    Extracts text from an uploaded PDF file.
    """
    try:
        _reset_stream(pdf_file)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF file: {e}"


def extract_text_from_txt(txt_file):
    """Extract text from a plain text file, attempting several encodings."""

    encoding_candidates = ("utf-8", "utf-8-sig", "utf-16", "latin-1")
    try:
        raw_data = _read_uploaded_file(txt_file)
        if isinstance(raw_data, str):
            return raw_data
        if not isinstance(raw_data, (bytes, bytearray)):
            raw_data = str(raw_data).encode("utf-8", errors="ignore")
        for encoding in encoding_candidates:
            try:
                return raw_data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw_data.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Error reading TXT file: {e}"


def extract_text_from_document(uploaded_file):
    """Extract text from either a PDF or TXT uploaded file."""

    if uploaded_file is None:
        return "Error reading file: no file provided."

    filename = str(getattr(uploaded_file, "name", "") or "").lower()
    mime_type = str(getattr(uploaded_file, "type", "") or "").lower()
    extension = Path(filename).suffix.lower() if filename else ""

    if extension == ".txt" or mime_type.startswith("text"):
        return extract_text_from_txt(uploaded_file)

    raw_data = _read_uploaded_file(uploaded_file)
    if isinstance(raw_data, str):
        raw_data = raw_data.encode("utf-8")
    pdf_buffer = io.BytesIO(raw_data)
    return extract_text_from_pdf(pdf_buffer)

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """
    Splits a long text into overlapping chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    # Create pairs of (upper, lower) chunks
    chunk_pairs = []
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            chunk_pairs.append((chunks[i], chunks[i+1]))
            
    return chunk_pairs
