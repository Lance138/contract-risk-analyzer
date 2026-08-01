"""
Embeds contract chunks into Chroma using a LOCAL embedding model (runs on
GPU if available, e.g. RTX 3070 Ti via CUDA). No API calls, no rate limits.

Run this on the machine with the GPU. Afterward, copy the resulting
chroma_db/ folder to wherever you run the Streamlit app.

IMPORTANT: app.py must use the SAME embedding model name below when
embedding user queries at search time, or retrieval will be meaningless
(vectors from different models aren't comparable).
"""

import json

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from chunking import chunk_contracts

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "contracts_collection"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

BATCH_SIZE = 200

MAX_CONTRACTS = None


def main():
    with open("data/contracts.json") as f:
        contracts = json.load(f)

    if MAX_CONTRACTS is not None:
        contracts = contracts[:MAX_CONTRACTS]
        print(f"Using first {MAX_CONTRACTS} contracts.")

    print("Chunking contracts...")
    chunks = chunk_contracts(contracts)
    print(f"Total chunks to embed: {len(chunks)}")

    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    num_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(
            f"Embedding batch {batch_num}/{num_batches} ({len(batch)} chunks)...")
        vectorstore.add_documents(batch)

    final_count = len(vectorstore.get(include=[])["ids"])
    print(f"\nDone. Chunks in vector store: {final_count}")
    print(f"Persisted at: {PERSIST_DIR}/")


if __name__ == "__main__":
    main()
