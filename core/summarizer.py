def summarize_content(text: str) -> str:
    """
    Generate a summary of the text. (This is a stub simulating a Claude 3 Haiku call.)
    """
    # In a real app, you might call the Anthropic Claude 3 API here.
    # We'll return the first 100 characters as a "summary".
    summary = text.strip().replace("\n", " ")
    summary = summary[:100] + ("..." if len(summary) > 100 else "")
    return f"Summary: {summary}"

def tag_content(text: str) -> list:
    """
    Generate tags for the text. (This is a stub simulating a Mistral 7B call.)
    """
    # In a real app, you might call a tag-generation model or use embeddings.
    # We'll simulate by picking some keywords.
    common_words = ["data", "analysis", "python", "research", "AI"]
    tags = []
    lower = text.lower()
    for word in common_words:
        if word in lower:
            tags.append(word)
    if not tags:
        tags = ["general"]
    return tags
