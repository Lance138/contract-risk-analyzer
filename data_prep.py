# Note: NOT USED IN THE FINAL PIPELINE - GEMINI FREE-TIER QUOTA EXCEEDED.
# LIMITS ON THIS DATASET SIZE. SEE EMBED_LOCAL.PY (LOCAL GPU BASED) INSTEAD

import json
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

# Matches the clause type name out of the templated question text, e.g.
# 'Highlight the parts (if any) of this contract related to "Governing Law"...'
CLAUSE_TYPE_PATTERN = re.compile(r'related to "([^"]+)"')


def extract_clause_type(question: str) -> str:
    match = CLAUSE_TYPE_PATTERN.search(question)
    if not match:
        raise ValueError(f"Could not parse clause type from question: {question!r}")
    return match.group(1)


def build_contracts(split_name: str = "train") -> list[dict]:
    dataset = load_dataset("theatticusproject/cuad-qa", revision="refs/pr/6")
    split = dataset[split_name]

    # title -> {"context": str, "clauses": {clause_type: [ {text, answer_start}, ... ]}}
    contracts: dict[str, dict] = {}

    mismatched_context_count = 0

    for row in split:
        title = row["title"]
        context = row["context"]
        clause_type = extract_clause_type(row["question"])

        if title not in contracts:
            contracts[title] = {"title": title, "context": context, "clauses": defaultdict(list)}
        elif contracts[title]["context"] != context:
            # Sanity check: every row for the same title should carry the
            # same contract text. Flag it if not, rather than silently
            # trusting mismatched data.
            mismatched_context_count += 1

        texts = row["answers"]["text"]
        starts = row["answers"]["answer_start"]
        for text, start in zip(texts, starts):
            contracts[title]["clauses"][clause_type].append(
                {"text": text, "answer_start": start}
            )

    if mismatched_context_count:
        print(
            f"WARNING: {mismatched_context_count} rows had a context that "
            f"didn't match earlier rows with the same title. Investigate "
            f"before trusting the data blindly."
        )

    # Convert defaultdicts to plain dicts and drop clause types with no
    # matches at all, so each contract only lists clauses it actually has.
    result = []
    for entry in contracts.values():
        clauses = {k: v for k, v in entry["clauses"].items() if v}
        result.append({"title": entry["title"], "context": entry["context"], "clauses": clauses})

    return result


def main():
    contracts = build_contracts(split_name="train")

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "contracts.json"
    with open(out_path, "w") as f:
        json.dump(contracts, f, indent=2)

    # Quick summary stats so we can sanity-check the result
    clause_counts = [len(c["clauses"]) for c in contracts]
    total_clause_type_names = set()
    for c in contracts:
        total_clause_type_names.update(c["clauses"].keys())

    print(f"Unique contracts: {len(contracts)}")
    print(f"Distinct clause types seen: {len(total_clause_type_names)}")
    print(f"Avg clause types present per contract: {sum(clause_counts) / len(clause_counts):.1f}")
    print(f"Min / max clause types in a single contract: {min(clause_counts)} / {max(clause_counts)}")
    print(f"Saved to: {out_path.resolve()}")

    # Show one real example
    example = contracts[0]
    print("\n--- Example contract ---")
    print(f"Title: {example['title']}")
    print(f"Context length: {len(example['context'])} chars")
    print(f"Clause types present: {list(example['clauses'].keys())}")


if __name__ == "__main__":
    main()