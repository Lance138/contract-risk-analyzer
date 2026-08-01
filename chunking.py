# Fixed-size chunking since contract formatting is too inconsistent for reliable regex section-splitting.

import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """Collapse irregular whitespace (multiple spaces, stray line breaks)
    that come from the original PDF-to-text extraction."""
    text = re.sub(r"[ \t]+", " ", text)       # collapse repeated spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)    # collapse 3+ newlines to 2
    return text.strip()


def chunk_contracts(contracts: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for contract in contracts:
        cleaned_text = clean_text(contract["context"])
        doc = Document(page_content=cleaned_text, metadata={"title": contract["title"]})
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)

    return all_chunks


def main():
    with open("data/contracts.json") as f:
        contracts = json.load(f)

    chunks = chunk_contracts(contracts)

    print(f"Contracts: {len(contracts)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg chunks per contract: {len(chunks) / len(contracts):.1f}")
    print("\n--- Example chunk ---")
    print(f"Source: {chunks[0].metadata['title']}")
    print(chunks[0].page_content[:500])


if __name__ == "__main__":
    main()