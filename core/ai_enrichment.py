# core/ai_enrichment.py

from mistralai import Mistral
import config

def generate_tags(text: str) -> list[str]:
    """
    Use Mistral AI to generate 5-7 relevant tags for the text.
    """
    with Mistral(api_key=config.MISTRAL_API_KEY) as client:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{
                "role": "user",
                "content": f"Generate 5-7 relevant tags (comma-separated) for the following text:\n\n{text}"
            }]
        )
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return []
    tags = [tag.strip() for tag in content.split(",") if tag.strip()]
    return tags

def summarize_text(text: str) -> str:
    """
    Use Mistral AI to generate a concise summary of the text.
    """
    with Mistral(api_key=config.MISTRAL_API_KEY) as client:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{
                "role": "user",
                "content": f"Summarize the following text in a concise manner:\n\n{text}"
            }]
        )
    try:
        summary = response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return ""
    return summary
