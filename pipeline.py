"""
pipeline.py
Ingestion, Gemini REST extraction (supporting AQ. and AIza keys),
and deterministic verification coordination.
"""

import json
from typing import List, Dict, Any
import requests
from verifier import verify_quote
from prompts import EXTRACTION_SYSTEM_PROMPT, QA_SYSTEM_PROMPT


def call_gemini_api(prompt: str, system_prompt: str, api_key: str, json_mode: bool = False) -> str:
    """Calls Gemini REST API directly using header authentication for AQ. keys."""
    clean_key = api_key.strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={clean_key}"
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": clean_key
    }
    
    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.0
        }
    }
    
    if json_mode:
        payload["generationConfig"]["response_mime_type"] = "application/json"
        
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    
    if response.status_code != 200:
        raise RuntimeError(f"Gemini API Error {response.status_code}: {response.text}")
        
    res_json = response.json()
    return res_json["candidates"][0]["content"]["parts"][0]["text"]


def run_extraction_pipeline(files_dict: Dict[str, str], api_key: str) -> List[Dict[str, Any]]:
    all_evidence = []
    evidence_id = 1

    for filename, raw_text in files_dict.items():
        prompt = f"""DOCUMENT FILENAME: {filename}
CONTENT:
\"\"\"{raw_text}\"\"\"

Extract all distinct evidence points according to instructions. Return pure JSON."""

        try:
            raw_resp = call_gemini_api(
                prompt=prompt,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                api_key=api_key,
                json_mode=True
            ).strip()

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
    context_str = "\n".join([
        f"[{e['id']}] ({e['file']} - {e['speaker']}) Theme: {e['theme']} | \"{e['quote']}\""
        for e in evidence_list if e.get("verified", False)
    ])

    prompt = f"Context Evidence:\n{context_str}\n\nQuestion: {query}"
    
    answer_text = call_gemini_api(
        prompt=prompt,
        system_prompt=QA_SYSTEM_PROMPT,
        api_key=api_key,
        json_mode=False
    ).strip()

    cited_evidence = [
        e for e in evidence_list 
        if e['id'] in answer_text or e['speaker'].lower() in answer_text.lower()
    ]

    return {
        "answer": answer_text,
        "cited_evidence": cited_evidence[:4]
    }
