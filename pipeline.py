import json
from typing import List, Dict, Any
import requests
from verifier import verify_quote
from prompts import EXTRACTION_SYSTEM_PROMPT, QA_SYSTEM_PROMPT

# Active models as per Google AI Studio
CANDIDATE_MODELS = [
    "gemini-3.6-flash",
    "gemini-3-flash",
    "gemini-flash"
]


def call_gemini_api(prompt: str, system_prompt: str, api_key: str, json_mode: bool = False) -> str:
    clean_key = api_key.strip()
    headers = {"Content-Type": "application/json"}
    
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

    errors = []
    for model in CANDIDATE_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={clean_key}"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                errors.append(f"{model} -> HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            errors.append(f"{model} -> Exception: {str(e)}")

    raise RuntimeError("All Gemini endpoints failed:\n" + "\n".join(errors))


def run_extraction_pipeline(files_dict: Dict[str, str], api_key: str) -> List[Dict[str, Any]]:
    all_evidence = []
    evidence_id = 1

    for filename, raw_text in files_dict.items():
        user_prompt = f"""DOCUMENT FILENAME: {filename}
CONTENT:
\"\"\"{raw_text}\"\"\"

Extract structured evidence points from this document. Return pure JSON."""

        raw_response = call_gemini_api(
            prompt=user_prompt,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            api_key=api_key,
            json_mode=True
        ).strip()

        if raw_response.startswith("```json"):
            raw_response = raw_response[7:]
        if raw_response.startswith("```"):
            raw_response = raw_response[3:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]

        items = json.loads(raw_response.strip())

        for item in items:
            candidate_quote = item.get("verbatim_quote", "")
            audit = verify_quote(quote=candidate_quote, raw_source=raw_text)

            all_evidence.append({
                "id": f"EVD-{evidence_id:03d}",
                "file": filename,
                "speaker": item.get("speaker", "Unknown"),
                "theme": item.get("theme", "General"),
                "quote": candidate_quote,
                "evidence_type": item.get("evidence_type", "Direct Quote"),
                "context": item.get("context", ""),
                "verified": audit["verified"],
                "match_type": audit["match_type"],
                "similarity_score": audit["similarity_score"],
                "audit_details": audit["details"]
            })
            evidence_id += 1

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
