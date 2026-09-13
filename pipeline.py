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
</code></pre>
  </Step>

  <Step subtitle="Commit & Test" title="Commit Changes and Test">
    1. Click the green <strong>Commit changes...</strong> button &rarr; <strong>Commit changes</strong>.<br/>
    2. Wait 30 seconds, then refresh your Streamlit app page in your browser.<br/>
    3. Paste your key, upload the 5 interview files, and click <strong>Run Evidence Pipeline</strong>.
    <br/><em>Verification:</em> The 404 error disappears, and a green success banner appears indicating all quotes have been extracted and verified.
  </Step>
</Steps>
