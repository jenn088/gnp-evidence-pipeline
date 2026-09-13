"""
pipeline.py
Ingestion, LLM extraction, and deterministic verification coordination.
"""

import json
from typing import List, Dict, Any
from openai import OpenAI
from verifier import verify_quote
from prompts import EXTRACTION_SYSTEM_PROMPT, QA_SYSTEM_PROMPT


def run_extraction_pipeline(files_dict: Dict[str, str], api_key: str) -> List[Dict[str, Any]]:
    client = OpenAI(api_key=api_key)
    all_evidence = []
    evidence_id = 1

    for filename, raw_text in files_dict.items():
        user_prompt = f"""DOCUMENT FILENAME: {filename}
CONTENT:
\"\"\"{raw_text}\"\"\"

Extract all distinct evidence points according to instructions. Return pure JSON."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )

            raw_resp = response.choices[0].message.content.strip()
            if raw_resp.startswith("```json"):
                raw_resp = raw_resp[7:]
            if raw_resp.endswith("```"):
                raw_resp = raw_resp[:-3]

            extracted_items = json.loads(raw_resp.strip())

            for item in extracted_items:
                quote = item.get("verbatim_quote", "")
                audit = verify_quote(quote=quote, raw_source=raw_text)

                all_evidence.append({
                    "id": f"EVD-{evidence_id:03d}",
                    "file": filename,
                    "speaker": item.get("speaker", "Unknown"),
                    "theme": item.get("theme", "General"),
                    "quote": quote,
                    "evidence_type": item.get("evidence_type", "Direct Quote"),
                    "context": item.get("context", ""),
                    "verified": audit["verified"],
                    "match_type": audit["match_type"],
                    "similarity_score": audit["similarity_score"],
                    "audit_details": audit["details"]
                })
                evidence_id += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return all_evidence


def ask_evidence_query(query: str, evidence_list: List[Dict[str, Any]], api_key: str) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)

    context_str = "\n".join([
        f"[{e['id']}] ({e['file']} - {e['speaker']}) Theme: {e['theme']} | \"{e['quote']}\""
        for e in evidence_list if e.get("verified", False)
    ])

    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context Evidence:\n{context_str}\n\nQuestion: {query}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0
    )

    answer_text = response.choices[0].message.content.strip()
    cited_evidence = [e for e in evidence_list if e['id'] in answer_text or e['speaker'].lower() in answer_text.lower()]

    return {
        "answer": answer_text,
        "cited_evidence": cited_evidence[:4]
    }
