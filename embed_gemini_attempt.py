"""Embeds contract chunks into Chroma. Paces requests to respect the free-tier 100 RPM embedding quota."""

import json
import os
import time

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from chunking import chunk_contracts

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "contracts_collection"

BATCH_SIZE = 20          # chunks (= requests) per embedding call
SLEEP_BETWEEN_BATCHES = 15  # seconds — keeps us under 100 RPM with buffer

# Set to an integer (e.g. 60) to only embed the first N contracts for a
# faster demo-sized run. Set to None to embed everything.
MAX_CONTRACTS = 60


def get_api_key() -> str:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = input("Enter your Google API key: ").strip()
        os.environ["GOOGLE_API_KEY"] = api_key
    return api_key


def embed_with_retry(vectorstore: Chroma, batch, max_retries: int = 4):
    for attempt in range(1, max_retries + 1):
        try:
            vectorstore.add_documents(batch)
            return
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = 30 * attempt
            print(f"  Batch failed ({type(e).__name__}); waiting {wait}s (attempt {attempt}/{max_retries})...")
            time.sleep(wait)


def main():
    api_key = get_api_key()

    with open("data/contracts.json") as f:
        contracts = json.load(f)

    if MAX_CONTRACTS is not None:
        contracts = contracts[:MAX_CONTRACTS]
        print(f"Using first {MAX_CONTRACTS} contracts (demo-sized run).")

    print("Chunking contracts...")
    chunks = chunk_contracts(contracts)
    print(f"Total chunks to embed: {len(chunks)}")

    est_minutes = (len(chunks) / BATCH_SIZE) * (SLEEP_BETWEEN_BATCHES / 60)
    print(f"Estimated time at current pacing: ~{est_minutes:.0f} minutes\n")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", google_api_key=api_key)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    num_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"Embedding batch {batch_num}/{num_batches} ({len(batch)} chunks)...")
        embed_with_retry(vectorstore, batch)

        if batch_num < num_batches:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    final_count = len(vectorstore.get(include=[])["ids"])
    print(f"\nDone. Chunks in vector store: {final_count}")
    print(f"Persisted at: {PERSIST_DIR}/")


if __name__ == "__main__":
    main()
