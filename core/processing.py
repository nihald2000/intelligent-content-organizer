# core/processing.py

import requests
from unstructured.partition.html import partition_html
from unstructured.partition.auto import partition
import config

def fetch_web_content(url: str) -> str:
    """
    Fetch and parse web content from the given URL into structured text.
    """
    try:
        # Use Unstructured to fetch and parse HTML content directly from the URL
        elements = partition_html(url=url)
        text = "\n\n".join([elem.text for elem in elements if hasattr(elem, 'text') and elem.text])
        return text
    except Exception:
        # If Unstructured parsing fails, attempt a simple HTTP GET as a fallback
        try:
            response = requests.get(url)
            response.raise_for_status()
            html_text = response.text
            # Attempt parsing the fetched HTML text
            elements = partition(filename=None, file=html_text)
            text = "\n\n".join([elem.text for elem in elements if hasattr(elem, 'text') and elem.text])
            return text
        except Exception:
            # On failure, return empty string
            return ""

def parse_local_file(file_path: str) -> str:
    """
    Parse a local file into structured text using the Unstructured library.
    Supports various file formats (e.g., PDF, DOCX, TXT).
    """
    try:
        elements = partition(filename=file_path)
        text = "\n\n".join([elem.text for elem in elements if hasattr(elem, 'text') and elem.text])
        return text
    except Exception:
        # Return empty string on failure
        return ""
