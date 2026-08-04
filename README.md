# Contract Risk Analyzer (Project for Broadway Infosys)

## Disclaimer
>This is a portfolio/learning project, not legal advice. Outputs should not be relied on for real contract decisions without human legal review.


A RAG-powered Streamlit app that analyzes contracts, flags risky clauses with severity ratings, and cites the exact source text behind each flag. Also includes a reference explorer over 408 real commercial contracts for browsing how specific clause types are typically written.

> **Status:** Core app working — upload-and-analyze flow and CUAD reference explorer are both functional. Evaluation against held-out test data and deployment are still in progress. See [Roadmap](#roadmap).

## Why this exists

Manually reviewing contracts for risky clauses (auto-renewal, liability caps, non-competes, etc.) is slow and expensive. This project automates the first pass: upload a contract and get an automatic report of which risk-relevant clauses are present, how risky they are, and the exact source text — instead of having to know what to ask.

It's also a portfolio/course project, so a deliberate goal here is to go beyond a basic RAG demo by including:
- **Structured extraction, not just Q&A** — the app proactively scans for a defined set of risk-relevant clause types and returns structured data (clause type, risk level, source text, explanation), not free-text answers.
- **Evaluation against ground truth** (in progress) — precision/recall on clause detection, measured against CUAD's expert-labeled contracts, not just eyeballed outputs.
- **Honest "not found" handling** — the app clearly separates clauses it found from ones it didn't, rather than guessing.

## Stack

| Piece | Tool |
|---|---|
| Frontend | Streamlit |
| Orchestration | LangChain |
| Vector store | Chroma |
| Chat/generation LLM | Google Gemini (`gemini-3.5-flash-lite`) |
| Embeddings | Local, `BAAI/bge-base-en-v1.5` via `sentence-transformers` (CPU at query time) |
| Structured output | Pydantic schema + LangChain's `with_structured_output` |
| PDF parsing | pypdf |
| Dataset | [CUAD](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus Dataset) — 510 real commercial contracts with clause-level labels |
| Vector store hosting | Hugging Face Hub (dataset repo) — downloaded automatically at app startup |

## Setup

```bash
git clone <this-repo-url>
cd <repo-folder>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You'll need a free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/) — the app prompts for it in the sidebar (or set it as a `GOOGLE_API_KEY` environment variable).

```bash
streamlit run app.py
```

On first run, the app downloads the embedding model and the pre-built CUAD vector store (~244MB, from Hugging Face Hub) — this only happens once and is cached after that.

## Data Preparation

CUAD's raw format (via the `datasets` library) has one row per (contract, clause-type) pair, meaning each of the 510 contracts appears ~44 times — once per possible clause type it was checked against, with an empty answer if that clause type isn't present.

`data_prep.py` deduplicates and reshapes this into one record per contract: full contract text plus a clean map of which clause types are actually present and their exact text spans. This produces `data/contracts.json`, used both as the source documents for the knowledge base and as ground-truth labels for evaluation (planned). Only the `train` split (408 contracts) is used for the app; the `test` split (~102 contracts) is deliberately held out for evaluation later.

## Chunking Strategy

Contract formatting is inconsistent across the corpus — some have numbered sections, others don't (see `chunking.py`). Rather than a regex-based section splitter that would work on some contracts and silently fail on others, chunking uses `RecursiveCharacterTextSplitter` tuned for legal text: larger chunk size (1800 chars) and generous overlap (250 chars) to reduce the chance of cutting a clause mid-sentence.

## Embedding

Two embedding approaches were tried:
- `embed_gemini_attempt.py` — used Google's Gemini embeddings API. Hit real free-tier rate limits (100 requests/minute) and a harder daily cap (1,000 requests/day) partway through a full run. Kept in the repo to document the tradeoff.
- `embed_local.py` — the working approach. Embeds all 16,784 chunks locally using `BAAI/bge-base-en-v1.5` on a GPU (via `sentence-transformers`/PyTorch with CUDA), with no rate limits. This is what actually built the vector store used by the app.

Because GitHub blocks individual files over 100MB and the resulting `chroma.sqlite3` is ~182MB, the built vector store is hosted on Hugging Face Hub instead of committed to this repo, and is downloaded automatically at app startup.

## App Architecture

The app uses **two separate vector stores**, kept intentionally isolated:

1. **CUAD reference store** (persistent, read-only) — the pre-built 16,784-chunk knowledge base of 408 real contracts, downloaded from Hugging Face Hub. Powers the "Explore reference contracts" tab, letting users see how specific clause types are typically written across real contracts.
2. **Session store** (ephemeral, per-upload) — built fresh each time a user uploads their own contract, kept completely separate from the CUAD store so retrieval isn't polluted by unrelated contracts. Powers both the structured risk analysis and the reactive Q&A chat in the "Analyze a contract" tab.

**Structured clause extraction** (`clause_extraction.py`) checks an uploaded contract against 8 risk-relevant clause types (Governing Law, Anti-Assignment, Exclusivity, Non-Compete, Cap On Liability, Uncapped Liability, Termination For Convenience, Most Favored Nation — chosen from CUAD's 41 categories, using CUAD's exact naming so results can later be compared against ground truth). For each clause type, it retrieves relevant chunks and asks the LLM to return structured output (via a Pydantic schema) rather than free text: whether the clause is present, its risk level, the exact source quote, and a brief explanation.

## Roadmap

- [x] Load and inspect the CUAD dataset
- [x] Clean and reshape CUAD into per-contract records with ground-truth clause labels
- [x] Clause chunking strategy
- [x] Vector store built and hosted externally
- [x] Basic RAG Q&A (uploaded contract + CUAD reference explorer)
- [x] Structured clause extraction with risk levels and source citations
- [ ] Evaluation script against held-out CUAD test split (precision/recall on clause detection)
- [ ] Test set including unanswerable questions, to verify the app doesn't hallucinate when info isn't present
- [ ] Deploy to Streamlit Community Cloud (pending a RAM usage check against the free tier's ~1GB limit)
