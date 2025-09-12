import PyPDF2
import io

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from an uploaded PDF file.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF file: {e}"

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
