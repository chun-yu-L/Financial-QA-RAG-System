import tiktoken

def count_tokens(text: str) -> int:
    """Returns the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))