"""
pipeline.py
Linguistic-aware pipeline that separates true spoken quotes from notetaker summaries,
runs structured classification, and performs deterministic verification.
"""

import re
import json
import time
from typing import List, Dict, Any, Tuple
import requests
from verifier import verify_quote
from prompts import CLASSIFICATION_SYSTEM_PROMPT, QA_SYSTEM_PROMPT

ACTIVE_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.6-pro"
]

# Patterns indicating natural first-person conversational voice
FIRST_PERSON_PATTERN = re.compile(
    r'\b(I|we|my|our|us|I\'m|we\'re|I\'ve|we\'ve|I\'d|we\'d|don\'t|can\'t|won\'t)\b',
    re.IGNORECASE
)

# Common notetaker shorthand prefixes that signify paraphrasing
NOTE_PREFIX_PATTERN = re.compile(
    r'^(mantra of|shift in|barrier|success|challenge of|example:|report directly|willingness to|old paradigm:|e\.g\.)',
    re.IGNORECASE
)


def extract_speaker_from_header(text: str) -> str:
    lines = text.strip().splitlines()
    if lines:
        first_line = lines[0]
        if "|" in first_line:
            return first_line.split("|")[-1].strip()
    return "Executive Stakeholder"


def classify_bullet_linguistics(bullet: str) -> Tuple[str, str]:
    """
    Classifies a raw bullet into:
    1. Direct Quote (explicitly in quotation marks)
    2. Spoken Verbatim (natural spoken first-person voice)
    3. Paraphrase / Summary (notetaker fragment or third-person summary)
    """
    stripped = bullet.strip()
    
    # 1. Check for explicit quotation marks
    quote_matches = re.findall(r'"([^"]+)"', stripped)
    if quote_matches:
        # Extract the primary quote inside the quotes
        longest_quote = max(quote_matches, key=len)
        if len(longest_quote.split()) >= 3:
            return "Direct Quote", longest_quote

    # 2. Check for natural first-person spoken sentence structure
    # Must have first-person tokens and sufficient sentence length (>= 5 words)
    words = stripped.split()
    has_first_person = bool(FIRST_PERSON_PATTERN.search(stripped))
    is_notetaker_prefix = bool(NOTE_PREFIX_PATTERN.search(stripped))

    if has_first_person and len(words) >= 5 and not is_notetaker_prefix:
        return "Spoken Verbatim", stripped

    # 3. Everything else is a notetaker paraphrase/summary
    return "Notetaker Paraphrase", stripped


def parse_interview_bullets(raw_text: str) -> List[Dict[str, Any]]:
    items = []
    current_section = "GENERAL"
    lines = raw_text.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect section headings
        if not stripped.startswith("-") and not stripped.startswith("[source:") and not stripped.startswith("GNP FOUNDATION"):
            if stripped.isupper() or "—" in stripped or "-" in stripped:
                current_section = stripped.replace("—", "-").strip()
                continue

        # Parse bullet lines
        if stripped.startswith("-"):
            bullet_body = stripped.lstrip("-").strip()
            if not bullet_body:
                continue

            evidence_type, quote_candidate = classify_bullet_linguistics(bullet_body)

            items.append({
                "section": current_section,
                "raw_bullet": bullet_body,
                "quote_candidate": quote_candidate,
                "evidence_type": evidence_type,
                "is_quote": evidence_type in ["Direct Quote", "Spoken Verbatim"]
            })

    return items


def call_gemini_api(prompt: str, system_prompt: str, api_key: str, json_mode: bool = False) -> str:
    clean_key = api_key.strip()
    headers = {"Content-Type": "application/json"}

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0}
    }

    if json_mode:
        payload["generationConfig"]["response_mime_type"] = "application/json"

    max_retries = 3
    last_error = ""

    for model in ACTIVE_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={clean_key}"
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 200:
                    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                elif resp.status_code in [503, 429]:
                    time.sleep(2 * (attempt + 1))
                    last_error = f"{model} (HTTP {resp.status_code}): {resp.text}"
                    continue
                else:
                    last_error = f"{model} (HTTP {resp.status_code}): {resp.text}"
                    break
            except Exception as e:
                last_error = f"{model} (Exception): {str(e)}"
                time.sleep(1)

    raise RuntimeError(f"Gemini API Error: {last_error}")


def run_extraction_pipeline(files_dict: Dict[str, str], api_key: str) -> List[Dict[str, Any]]:
    all_evidence = []
    evidence_id = 1

    for filename, raw_text in sorted(files_dict.items()):
        speaker = extract_speaker_from_header(raw_text)
        parsed_bullets = parse_interview_bullets(raw_text)

        # Batch classify with Gemini
        bullets_payload = [
            {"index": idx, "section": b["section"], "text": b["quote_candidate"]}
            for idx, b in enumerate(parsed_bullets)
        ]

        prompt = f"""SPEAKER: {speaker}
DOCUMENT: {filename}
BULLETS:
{json.dumps(bullets_payload, indent=2)}

Assign each indexed bullet to exactly one of the 5 themes:
- Decision-Making & Bureaucracy
- Cross-Functional Silos & Alignment
- Grantee Experience & Responsiveness
- Workforce Capability & Change Readiness
- Leadership & Governance

Return JSON array: [{{"index": 0, "theme": "..."}}]"""

        try:
            raw_resp = call_gemini_api(
                prompt=prompt,
                system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
                api_key=api_key,
                json_mode=True
            ).strip()

            if raw_resp.startswith("```json"):
                raw_resp = raw_resp[7:]
            if raw_resp.endswith("```"):
                raw_resp = raw_resp[:-3]

            theme_map = {item["index"]: item.get("theme", "General") for item in json.loads(raw_resp.strip())}
        except Exception:
            theme_map = {}

        for idx, item in enumerate(parsed_bullets):
            theme = theme_map.get(idx, item["section"].title())
            quote = item["quote_candidate"]
            audit = verify_quote(quote=quote, raw_source=raw_text)

            all_evidence.append({
                "id": f"EVD-{evidence_id:03d}",
                "file": filename,
                "speaker": speaker,
                "theme": theme,
                "quote": quote,
                "raw_bullet": item["raw_bullet"],
                "evidence_type": item["evidence_type"],
                "is_quote": item["is_quote"],
                "context": f"Section: {item['section']}",
                "verified": audit["verified"],
                "match_type": audit["match_type"],
                "similarity_score": audit["similarity_score"],
                "audit_details": audit["details"]
            })
            evidence_id += 1

    return all_evidence


def ask_evidence_query(query: str, evidence_list: List[Dict[str, Any]], api_key: str) -> Dict[str, Any]:
    # Ground only on verified true quotes (Direct Quotes and Spoken Verbatims)
    context_str = "\n".join([
        f"[{e['id']}] ({e['file']} - {e['speaker']}) Theme: {e['theme']} | \"{e['quote']}\""
        for e in evidence_list if e.get("verified", False) and e.get("is_quote", False)
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
        if e.get("is_quote", False) and (e['id'] in answer_text or e['speaker'].lower() in answer_text.lower())
    ]

    return {
        "answer": answer_text,
        "cited_evidence": cited_evidence[:4]
    }
