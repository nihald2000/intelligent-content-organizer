# core/utils.py

import re
from datetime import datetime
import hashlib

def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing extra whitespace.
    """
    if not text:
        return ""
    # Collapse multiple whitespace into single spaces and strip ends
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()

def generate_doc_id(source: str) -> str:
    """
    Generate a unique document ID based on source identifier and timestamp.
    """
    timestamp = datetime.now().isoformat()
    raw_id = f"{source}-{timestamp}"
    return hashlib.md5(raw_id.encode()).hexdigest()
