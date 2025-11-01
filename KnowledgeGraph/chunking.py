"""
Fetch the text from the file and split it into chunks

Returns:
    list: A list of chunk dictionaries with metadata.
"""

import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Large chunks and overlap for legal documents to preserve context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    length_function=len,
    separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
)

def split_data_from_file(file):
    """
    Splits either a JSON or Markdown file into text chunks with metadata.
    Returns a list of chunk dicts.
    """
    chunks_with_metadata = []
    ext = os.path.splitext(file)[1].lower()

    if ext == ".json":
        # ✅ JSON handling
        with open(file, "r", encoding="utf-8") as f:
            try:
                file_as_object = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON file: {file}")

        if not isinstance(file_as_object, dict):
            raise ValueError(f"Expected a JSON object in {file}, got {type(file_as_object)}")

        for item, text in file_as_object.items():
            item_text_chunks = text_splitter.split_text(str(text))
            form_name = os.path.splitext(os.path.basename(file))[0]

            for seq_id, chunk in enumerate(item_text_chunks):
                chunks_with_metadata.append({
                    "text": chunk,
                    "Source": item,
                    "source": item,             # <--
                    "chunkSeqId": seq_id,
                    "chunkId": f"{form_name}-{item}-chunk{seq_id:04d}"
                })

    elif ext in {".md", ".txt"}:
        # ✅ Markdown or plain text handling
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        text_chunks = text_splitter.split_text(text)
        form_name = os.path.splitext(os.path.basename(file))[0]

        for seq_id, chunk in enumerate(text_chunks):
            chunks_with_metadata.append({
                "text": chunk,
                "Source": form_name,
                "source": form_name,          # <--
                "chunkSeqId": seq_id,
                "chunkId": f"{form_name}-chunk{seq_id:04d}"
            })
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return chunks_with_metadata

