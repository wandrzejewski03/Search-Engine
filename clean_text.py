import re

def clean_text(text):

  text = text.lower()
  text = re.sub(r"\\", "", text)  # Remove backslashes
  text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters and punctuation
  text = re.sub(r"\s+", " ", text)  # Remove extra whitespace
  text = text.strip()
  return text



