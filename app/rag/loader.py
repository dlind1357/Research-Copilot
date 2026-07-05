import fitz  # PyMuPDF

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file using PyMuPDF.
    
    Args:
        file_path: Path to the PDF file on disk.
        
    Returns:
        Extracted text as a single string.
    """
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text += page_text + "\n"
    return text
