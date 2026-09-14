"""
pipeline.py
Production-grade linguistic evidence extraction and deterministic verification pipeline.
Handles multi-bullet overflows, notetaker prefix stripping, spoken-voice detection,
and automated thematic classification.
"""

import re
import json
import time
from typing import List, Dict, Any, Tuple
import requests
from verifier import verify_quote
from prompts import CLASSIFICATION_SYSTEM_PROMPT, QA_SYSTEM_PROMPT

# Global working model cache
WORKING_MODEL = None

# --- LINGUISTIC REGEX DEFINITIONS ---
FIRST_PERSON_PATTERN = re.compile(
    r'\b(I|we|my|our|us|me|I\'m|we\'re|I\'ve|we\'ve|I\'d|we\'d|don\'t|can\'t|won\'t)\b',
    re.IGNORECASE
)

FINITE_VERB_PATTERN = re.compile(
    r'\b(is|are|was|were|have|has|had|do|does|did|can|could|will|would|should|may|might|must|'
    r'want|need|deliver|struggle|work|make|get|take|know|think|feel|see|show|say|tell|start|'
    r'evolve|rely|refuse|resist|scarred|holding|pulling|facing|moving)\b',
    re.IGNORECASE
)

# Common notetaker structural prefixes that precede the actual quote
NOTE_PREFIX_PATTERN = re.compile(
    r'^([A-Za-z0-9\s&/,\(\)]+[:—\?]\s*)',
    re.IGNORECASE
)

# Continuations indicating multi-bullet thought overflow
CONTINUATION_START_PATTERN = re.compile(
    r'^(and|but|because|so|e\.g\.|for example|which|for resistors|or|plus)\b',
    re.IGNORECASE
)


def get_available_model(api_key: str) -> str:
    """Discovers available generateContent models on Google AI Studio."""
    global WORKING_MODEL
    if WORKING_MODEL:
        return WORKING_MODEL

    clean_key = api_key.strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={clean_key}"

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            models_data = resp.json().get("models", [])
            compatible = [
                m["name"].replace("models/", "")
                for m in models_data
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            flash_candidates = [m for m in compatible if "flash" in m]
            if flash_candidates:
                WORKING_MODEL = flash_candidates[0]
                return WORKING_MODEL
            elif compatible:
                WORKING_MODEL = compatible[0]
                return WORKING_MODEL
    except Exception as e:
        print(f"Model discovery fallback: {e}")

    WORKING_MODEL = "gemini-3.6-flash"
    return WORKING_MODEL


def extract_speaker_from_header(text: str) -> str:
    """Extracts speaker role from document header, supporting multiple formats."""
    lines = text.strip().splitlines()
    if not lines:
        return "Executive Stakeholder"
    first_line = lines[0]
    if "|" in first_line:
        return first_line.split("|")[-1].strip()
    if "—" in first_line or "-" in first_line:
        parts = re.split(r'[—\-]', first_line)
        return parts[-1].strip()
    return "Executive Stakeholder"


def parse_and_stitch_document(raw_text: str) -> List[Dict[str, str]]:
    """
    Ingests text, tracks headers, and stitches multi-bullet sentence overflows.
    """
    lines = raw_text.splitlines()
    bullets = []
    current_sec = "GENERAL"

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # Header detection: lines without bullet dashes that are uppercase or section-like
        if not s.startswith("-") and not s.startswith("*") and not s.startswith("[source:") and not s.startswith("GNP FOUNDATION"):
            if s.isupper() or "—" in s or "-" in s:
                current_sec = s.replace("—", "-").strip()
                continue

        # Bullet ingestion (supports '-' and '*' bullets)
        if s.startswith("-") or s.startswith("*"):
            body = s.lstrip("-*").strip()
            if body:
                bullets.append({"section": current_sec, "raw": body})

    # Stitch thought overflows
    stitched_items = []
    i = 0
    while i < len(bullets):
        curr = bullets[i]
        curr_text = curr["raw"]
        curr_sec = curr["section"]

        while i + 1 < len(bullets):
            nxt = bullets[i + 1]
            if nxt["section"] != curr_sec:
                break
            nxt_text = nxt["raw"]
            if CONTINUATION_START_PATTERN.search(nxt_text) or nxt_text[0].islower():
                curr_text = f"{curr_text} {nxt_text}"
                i += 1
            else:
                break

        stitched_items.append({"section": curr_sec, "text": curr_text})
        i += 1

    return stitched_items


def classify_linguistic_element(item_text: str) -> Tuple[str, str]:
    """
    Determines whether an item is a Direct Quote, Spoken Verbatim, or Notetaker Note.
    Returns: (evidence_type, clean_quote_candidate)
    """
    text = item_text.strip()

    # Rule 1: Substantive quote enclosed in quotation marks
    explicit_matches = re.findall(r'"([^"]+)"', text)
    for q in explicit_matches:
        if len(q.split()) >= 4 and FINITE_VERB_PATTERN.search(q):
            return "Direct Quote", q

    # Rule 2: Prefix stripping for shorthand labels (e.g., 'PAIN POINTS — We just...')
    candidate = text
    prefix_match = NOTE_PREFIX_PATTERN.match(text)
    if prefix_match:
        prefix = prefix_match.group(1)
        remainder = text[len(prefix):].strip()
        if len(remainder.split()) >= 4:
            candidate = remainder

    # Rule 3: Spoken Voice Evaluation (First-Person deictic markers + Finite Verb)
    has_first_person = bool(FIRST_PERSON_PATTERN.search(candidate))
    has_finite_verb = bool(FINITE_VERB_PATTERN.search(candidate))
    word_count = len(candidate.split())

    # Rhetorical questions spoken by leadership
    is_rhetorical = candidate.endswith("?") and any(
        candidate.lower().startswith(w) for w in ["how do we", "how does", "what are", "why"]
    )

    if (has_first_person and has_finite_verb and word_count >= 5) or (is_rhetorical and word_count >= 4):
        # Exclude notetaker meta-commentary
        if not candidate.lower().startswith("mantra of"):
            return "Spoken Verbatim", candidate

    # Rule 4: Otherwise, notetaker paraphrase/summary
    return "Notetaker Note", text


def call_gemini_api(prompt: str, system_prompt: str, api_key: str, json_mode: bool = False) -> str:
    """Calls Gemini REST endpoint with dynamic model resolution and automatic retries."""
    clean_key = api_key.strip()
    model_name = get_available_model(clean_key)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={clean_key}"
    headers = {"Content-Type": "application/json"}

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0}
    }

    if json_mode:
        payload["generationConfig"]["response_mime_type"] = "application/json"

    max_retries = 4
    last_error = ""

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif resp.status_code in [503, 429]:
                time.sleep(3 * (attempt + 1))
                last_error = f"{model_name} (HTTP {resp.status_code}): {resp.text}"
                continue
            else:
                last_error = f"{model_name} (HTTP {resp.status_code}): {resp.text}"
                break
        except Exception as e:
            last_error = f"{model_name} (Exception): {str(e)}"
            time.sleep(2)

    raise RuntimeError(f"Gemini API Error after retries: {last_error}")


def run_extraction_pipeline(files_dict: Dict[str, str], api_key: str) -> List[Dict[str, Any]]:
    """Runs deterministic parsing and verification across arbitrary sets of interview files."""
    all_evidence = []
    evidence_id = 1

    for filename, raw_text in sorted(files_dict.items()):
        speaker = extract_speaker_from_header(raw_text)
        stitched_items = parse_and_stitch_document(raw_text)

        parsed_records = []
        for it in stitched_items:
            ev_type, quote_cand = classify_linguistic_element(it["text"])
            parsed_records.append({
                "section": it["section"],
                "full_text": it["text"],
                "quote_candidate": quote_cand,
                "evidence_type": ev_type,
                "is_quote": ev_type in ["Direct Quote", "Spoken Verbatim"]
            })

        # Batch classify themes using the LLM
        classification_payload = [
            {"index": idx, "section": r["section"], "text": r["quote_candidate"]}
            for idx, r in enumerate(parsed_records)
        ]

        prompt = f"""SPEAKER: {speaker}
DOCUMENT: {filename}
EVIDENCE ITEMS:
{json.dumps(classification_payload, indent=2)}

Assign each indexed item to exactly one strategic theme:
- Decision-Making & Bureaucracy
- Cross-Functional Silos & Alignment
- Grantee Experience & Responsiveness
- Workforce Capability & Change Readiness
- Leadership & Governance

Return JSON array of objects: [{{"index": 0, "theme": "..."}}]"""

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

        for idx, rec in enumerate(parsed_records):
            theme = theme_map.get(idx, rec["section"].title())
            quote = rec["quote_candidate"]

            # Independent verification pass
            audit = verify_quote(quote=quote, raw_source=raw_text)

            all_evidence.append({
                "id": f"EVD-{evidence_id:03d}",
                "file": filename,
                "speaker": speaker,
                "theme": theme,
                "quote": quote,
                "full_text": rec["full_text"],
                "evidence_type": rec["evidence_type"],
                "is_quote": rec["is_quote"],
                "context": f"Section: {rec['section']}",
                "verified": audit["verified"],
                "match_type": audit["match_type"],
                "similarity_score": audit["similarity_score"],
                "audit_details": audit["details"]
            })
            evidence_id += 1

    return all_evidence


def ask_evidence_query(query: str, evidence_list: List[Dict[str, Any]], api_key: str) -> Dict[str, Any]:
    """Answers queries grounded strictly in verified quotes."""
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
