import requests
from bs4 import BeautifulSoup
from unstructured.partition.auto import partition

def parse_document(file_path: str) -> str:
    """
    Parse a document file (PDF, DOCX, TXT, etc.) into text using Unstructured.
    """
    try:
        elements = partition(file_path)
        # Combine text elements into a single string
        text = "\n".join([elem.text for elem in elements if elem.text])
        return text
    except Exception as e:
        return f"Error parsing document: {e}"

def parse_url(url: str) -> str:
    """
    Fetch and parse webpage content at the given URL.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract visible text from paragraphs
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
        text = "\n".join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        return f"Error fetching URL: {e}"
