# This file extracts structured clause data (type, risk, etc.) from contract text using LLM.

from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

CLAUSE_TYPES_TO_CHECK = [
    "Governing Law",
    "Anti-Assignment",
    "Exclusivity",
    "Non-Compete",
    "Cap On Liability",
    "Uncapped Liability",
    "Termination For Convenience",
    "Most Favored Nation",
]


class ClauseFinding(BaseModel):
    present: bool = Field(description="Whether this clause type appears in the contract")
    risk_level: Optional[str] = Field(default=None, description="Low, Medium, or High. Null if not present.")
    source_text: Optional[str] = Field(default=None, description="Exact quoted text, or null if not present.")
    explanation: str = Field(description="Brief explanation of the risk, or why it wasn't found.")


def get_structured_llm(api_key: str):
    base_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, google_api_key=api_key)
    return base_llm.with_structured_output(ClauseFinding)


def analyze_clause(retriever, structured_llm, clause_type: str, format_docs_fn) -> ClauseFinding:
    docs = retriever.invoke(clause_type)
    context = format_docs_fn(docs)
    prompt = (
        f"You are reviewing a contract for the clause type: '{clause_type}'.\n\n"
        f"Contract excerpts:\n{context}\n\n"
        f"Determine if this clause type is present. If present, quote the exact source "
        f"text, rate the risk level (Low/Medium/High) from the perspective of the party "
        f"signing, and briefly explain why. If not present, set present=false."
    )
    return structured_llm.invoke(prompt)