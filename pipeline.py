"""
pipeline.py
Linguistic extraction with robust transformational theme classification.
Guarantees strategic recommendation themes instead of raw document subheadings.
"""

import re
import json
import time
from typing import List, Dict, Any, Tuple
import requests
from verifier import verify_quote
from prompts import CLASSIFICATION_SYSTEM_PROMPT, QA_SYSTEM_PROMPT

ACTIVE_MODEL = "gemini-3.6-flash"

# Core Transformational Recommendation Pillars
TRANSFORMATION_THEMES = [
    "Governance & Decision-Making Authority",
    "Operating Model & Cross-Functional Silos",
    "Grantee Experience & Service Delivery",
    "Workforce Capabilities & Cultural Inertia",
    "Leadership Alignment & Strategy Execution"
]

PROXY_SECTION_PATTERNS = re.compile(
    r'(GRANTEE|SURVEY|COMMUNITY REQUESTS|EXTERNAL|FEEDBACK|FUTURE DEMANDS)',
    re.IGNORECASE
)

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

NOTE_PREFIX_PATTERN = re.compile(
    r'^([A-Za-z0-9\s&/,\(\)]+[:—\?]\s*)',
    re.IGNORECASE
)

CONTINUATION_START_PATTERN = re.compile(
    r'^(and|but|because|so|e\.g\.|for example|which|for resistors|or|plus)\b',
    re.IGNORECASE
)


def fallback_strategic_theme(text: str, section: str) -> str:
    """
    Intelligent safety net: If the LLM call fails, maps content into a 
    transformational theme rather than falling back to the raw subheading.
    """
    blob = f"{section} {text}".lower()

    if any(k in blob for k in ["decision", "approval", "bureaucracy", "hierarchy", "sign off", "escalat", "defer", "pyramid"]):
        return "Governance & Decision-Making Authority"
    elif any(k in blob for k in ["silo", "pcka", "policy", "community team", "department", "cross-functional", "handoff", "rumor", "matrix"]):
        return "Operating Model & Cross-Functional Silos"
    elif any(k in blob for k in ["grantee", "customer", "disburs", "responsive", "speed", "funding", "cycle", "grant"]):
        return "Grantee Experience & Service Delivery"
    elif any(k in blob for k in ["skill", "resistor", "turnover", "generalist", "tenure", "coach", "clique", "hiring", "inertia", "contract"]):
        return "Workforce Capabilities & Cultural Inertia"
    else:
        return "Leadership Alignment & Strategy Execution"


def extract_primary_speaker(filename: str, text: str) -> str:
    lines = text.strip().splitlines()
    if lines and "|" in lines[0]:
        return lines[0].split("|")[-1].strip()
    clean_name = filename.replace(".txt", "").replace("interview_", "")
    clean_name = re.sub(r'^\d+_', '', clean_name)
    return clean_name.replace("_", " ").title()


def parse_document_structure(raw_text: str) -> List[Dict[str, Any]]:
    lines = raw_text.splitlines()
    bullets = []
    current_sec = "GENERAL"

    for line in lines:
        s = line.strip()
        if not s:
            continue

        if not s.startswith("-") and not s.startswith("*") and not s.startswith("[source:") and not s.startswith("GNP FOUNDATION"):
            if s.isupper() or "—" in s or "-" in s:
                current_sec = s.replace("—", "-").strip()
                continue

        if s.startswith("-") or s.startswith("*"):
            body = s.lstrip("-*").strip()
            if body:
                bullets.append({"section": current_sec, "raw": body})

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


def classify_bullet_linguistics(item_text: str, section: str, primary_speaker: str) -> Tuple[str, str, str, str, bool]:
    text = item_text.strip()
    is_proxy_section = bool(PROXY_SECTION_PATTERNS.search(section))

    if is_proxy_section:
        attributed_speaker = f"Grantee Feedback (via {primary_speaker})"
    else:
        attributed_speaker = primary_speaker

    # Explicit quote marks
    explicit_matches = re.findall(r'"([^"]+)"', text)
    for q in explicit_matches:
        if len(q.split()) >= 4 and FINITE_VERB_PATTERN.search(q):
            context_desc = f"[{section}] {text}"
            if is_proxy_section:
                return attributed_speaker, "Grantee Voice", q, context_desc, False
            return attributed_speaker, "Direct Quote", q, context_desc, True

    # Prefix stripping
    candidate = text
    prefix_match = NOTE_PREFIX_PATTERN.match(text)
    if prefix_match:
        prefix = prefix_match.group(1)
        remainder = text[len(prefix):].strip()
        if len(remainder.split()) >= 4:
            candidate = remainder

    context_desc = f"[{section}] {text}"

    has_first_person = bool(FIRST_PERSON_PATTERN.search(candidate))
    has_finite_verb = bool(FINITE_VERB_PATTERN.search(candidate))
    word_count = len(candidate.split())

    is_rhetorical = candidate.endswith("?") and any(
        candidate.lower().startswith(w) for w in ["how do we", "how does", "what are", "why"]
    )

    if (has_first_person and has_finite_verb and word_count >= 4) or (is_rhetorical and word_count >= 3):
        if not candidate.lower().startswith("mantra of"):
            if is_proxy_section:
                return attributed_speaker, "Grantee Voice", candidate, context_desc, False
            return attributed_speaker, "Spoken Verbatim", candidate, context_desc, True

    if is_proxy_section:
        return attributed_speaker, "Grantee Summary", text, context_desc, False
    return attributed_speaker, "Notetaker Note", text, context_desc, False


def call_gemini_api(prompt: str, system_prompt: str, api_key: str, json_mode: bool = False) -> str:
    clean_key = api_key.strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{ACTIVE_MODEL}:generateContent?key={clean_key}"
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

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=45)
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif resp.status_code in [503, 429]:
                time.sleep(2 * (attempt + 1))
                last_error = f"{ACTIVE_MODEL} (HTTP {resp.status_code}): {resp.text}"
                continue
            else:
                last_error = f"{ACTIVE_MODEL} (HTTP {resp.status_code}): {resp.text}"
                break
        except Exception as e:
            last_error = str(e)
            time.sleep(1)

    raise RuntimeError(f"Gemini API Error: {last_error}")


def run_extraction_pipeline(files_dict: Dict[str, str], api_key: str) -> List[Dict[str, Any]]:
    all_evidence = []
    evidence_id = 1

    for filename, raw_text in sorted(files_dict.items()):
        primary_speaker = extract_primary_speaker(filename, raw_text)
        stitched_items = parse_document_structure(raw_text)

        parsed_records = []
        for it in stitched_items:
            speaker, ev_type, quote_cand, context_desc, is_interviewee_quote = classify_bullet_linguistics(
                it["text"], it["section"], primary_speaker
            )
            parsed_records.append({
                "section": it["section"],
                "full_text": it["text"],
                "quote_candidate": quote_cand,
                "context_desc": context_desc,
                "speaker": speaker,
                "evidence_type": ev_type,
                "is_interviewee_quote": is_interviewee_quote
            })

        # Thematic classification payload
        classification_payload = [
            {"index": idx, "section": r["section"], "text": r["quote_candidate"]}
            for idx, r in enumerate(parsed_records)
        ]

        prompt = f"""SPEAKER: {primary_speaker}
DOCUMENT: {filename}
EVIDENCE ITEMS:
{json.dumps(classification_payload, indent=2)}

Assign each indexed item to exactly ONE of these 5 transformation recommendation themes:
1. Governance & Decision-Making Authority
2. Operating Model & Cross-Functional Silos
3. Grantee Experience & Service Delivery
4. Workforce Capabilities & Cultural Inertia
5. Leadership Alignment & Strategy Execution

Return a JSON array of objects: [{{"index": 0, "theme": "..."}}]"""

        theme_map = {}
        if api_key:
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

                parsed_json = json.loads(raw_resp.strip())
                theme_map = {item["index"]: item.get("theme") for item in parsed_json if item.get("theme") in TRANSFORMATION_THEMES}
            except Exception as e:
                # Log error silently and proceed to strategic fallback
                print(f"Theme classification fallback triggered for {filename}: {e}")

        for idx, rec in enumerate(parsed_records):
            # Prioritize LLM theme -> Strategic Fallback -> Never raw section header
            theme = theme_map.get(idx) or fallback_strategic_theme(rec["quote_candidate"], rec["section"])
            quote = rec["quote_candidate"]
            audit = verify_quote(quote=quote, raw_source=raw_text)

            all_evidence.append({
                "id": f"EVD-{evidence_id:03d}",
                "file": filename,
                "speaker": rec["speaker"],
                "theme": theme,
                "quote": quote,
                "context": rec["context_desc"],
                "full_text": rec["full_text"],
                "evidence_type": rec["evidence_type"],
                "is_interviewee_quote": rec["is_interviewee_quote"],
                "verified": audit["verified"],
                "match_type": audit["match_type"],
                "similarity_score": audit["similarity_score"],
                "audit_details": audit["details"]
            })
            evidence_id += 1

    return all_evidence


def ask_evidence_query(query: str, evidence_list: List[Dict[str, Any]], api_key: str) -> Dict[str, Any]:
    context_str = "\n".join([
        f"[{e['id']}] ({e['file']} - {e['speaker']}) Theme: {e['theme']} | Quote: \"{e['quote']}\" | Context: {e['context']}"
        for e in evidence_list if e.get("verified", False)
    ])

    prompt = f"Context Evidence:\n{context_str}\n\nQuestion: {query}"

    try:
        answer_text = call_gemini_api(
            prompt=prompt,
            system_prompt=QA_SYSTEM_PROMPT,
            api_key=api_key,
            json_mode=False
        ).strip()
    except Exception:
        # Fallback if API rate limits during Q&A
        query_tokens = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 3]
        matched = []
        for ev in evidence_list:
            blob = f"{ev['theme']} {ev['quote']} {ev['context']}".lower()
            if any(t in blob for t in query_tokens):
                matched.append(ev)
        
        if not matched:
            return {"answer": "Not found in the interviews or fact pack.", "cited_evidence": []}
            
        points = [f"• **{m['speaker']}**: \"{m['quote']}\"" for m in matched[:4]]
        return {
            "answer": "Synthesized evidence based on interview records:\n\n" + "\n".join(points),
            "cited_evidence": matched[:4]
        }

    cited_evidence = [
        e for e in evidence_list
        if e['id'] in answer_text or e['speaker'].lower() in answer_text.lower()
    ]

    return {
        "answer": answer_text,
        "cited_evidence": cited_evidence[:4]
    }
