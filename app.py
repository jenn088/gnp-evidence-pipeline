"""
app.py
Streamlit web application for the GNP Evidence Pipeline.
Features:
1. Dynamic Evidence Matrix (streamlined layout without top theme boxes)
2. Deterministic Verification Audit
3. Grounded Q&A (gemini-3.6-flash)
4. Linguistic Methodology & Filter Logic
"""

import streamlit as st
import pandas as pd
from pipeline import run_extraction_pipeline, ask_evidence_query

st.set_page_config(
    page_title="Evidence & Transformation Pipeline",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Qualitative Evidence & Transformation Engine")
st.caption("Dynamic inductive thematic discovery, subheading-aware extraction, and deterministic verification.")

api_key = st.secrets.get("GEMINI_API_KEY", "").strip()

st.sidebar.header("Evidence Ingestion")
uploaded_files = st.sidebar.file_uploader(
    "Upload Interview Files (.txt)",
    type=["txt"],
    accept_multiple_files=True
)

if "evidence_data" not in st.session_state:
    st.session_state.evidence_data = []

if "discovered_themes" not in st.session_state:
    st.session_state.discovered_themes = []

if st.sidebar.button("Run Evidence Pipeline", type="primary"):
    if not api_key:
        st.sidebar.error("System configuration error: GEMINI_API_KEY is missing from app secrets.")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one .txt interview file.")
    else:
        with st.spinner("Analyzing uploaded corpus, dynamically discovering themes, and running audits..."):
            try:
                files_dict = {f.name: f.read().decode("utf-8") for f in uploaded_files}
                data, themes = run_extraction_pipeline(files_dict, api_key)
                st.session_state.evidence_data = data
                st.session_state.discovered_themes = themes
                
                quotes_count = sum(1 for e in st.session_state.evidence_data if e["is_interviewee_quote"])
                st.sidebar.success(
                    f"Discovered {len(themes)} dynamic transformation themes across "
                    f"{len(st.session_state.evidence_data)} total items ({quotes_count} verified quotes)!"
                )
            except Exception as ex:
                st.error(f"Pipeline Error: {str(ex)}")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dynamic Evidence Matrix",
    "🔍 Verification Report",
    "💬 Grounded Q&A",
    "📖 Linguistic Methodology & Filter Logic"
])

# Tab 1: Dynamic Evidence Matrix (Clean, Direct Table View)
with tab1:
    if not st.session_state.evidence_data:
        st.info("Upload interview files in the sidebar and click **Run Evidence Pipeline**.")
    else:
        df = pd.DataFrame(st.session_state.evidence_data)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            theme_choice = st.selectbox("Filter Dynamic Theme", ["All"] + sorted(df["theme"].unique().tolist()))
        with col2:
            speaker_choice = st.selectbox("Filter Speaker", ["All"] + sorted(df["speaker"].unique().tolist()))
        with col3:
            evidence_view = st.selectbox(
                "Filter Evidence Type",
                [
                    "Interviewee Quotes Only",
                    "Grantee Voice / Feedback",
                    "All Evidence (Incl. Notes)"
                ]
            )
        with col4:
            verified_filter = st.checkbox("Verified Word-for-Word Only", value=True)

        filtered = df.copy()
        if theme_choice != "All":
            filtered = filtered[filtered["theme"] == theme_choice]
        if speaker_choice != "All":
            filtered = filtered[filtered["speaker"] == speaker_choice]

        if evidence_view == "Interviewee Quotes Only":
            filtered = filtered[filtered["is_interviewee_quote"] == True]
        elif evidence_view == "Grantee Voice / Feedback":
            filtered = filtered[filtered["evidence_type"].isin(["Grantee Voice", "Grantee Summary"])]

        if verified_filter:
            filtered = filtered[filtered["verified"] == True]

        st.dataframe(
            filtered[["id", "theme", "speaker", "quote", "context", "file", "verified", "match_type"]],
            use_container_width=True,
            height=520,
            column_config={
                "theme": st.column_config.TextColumn("Dynamic Theme", width="medium"),
                "quote": st.column_config.TextColumn("Verbatim Quote", width="medium"),
                "context": st.column_config.TextColumn("Context / Notes", width="large"),
            }
        )

# Tab 2: Verification Report
with tab2:
    if not st.session_state.evidence_data:
        st.info("Run the pipeline to generate an integrity audit report.")
    else:
        df = pd.DataFrame(st.session_state.evidence_data)
        quotes_df = df[df["is_interviewee_quote"] == True]
        proxy_df = df[df["is_interviewee_quote"] == False]

        total_quotes = len(quotes_df)
        verified_quotes = int(quotes_df["verified"].sum())
        quote_pass_rate = round((verified_quotes / total_quotes) * 100, 1) if total_quotes > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Interviewee Quotes Audited", total_quotes)
        m2.metric("Verified Word-for-Word", verified_quotes)
        m3.metric("Grantee & Notetaker Records Separated", len(proxy_df))
        m4.metric("Executive Quote Integrity", f"{quote_pass_rate}%")

        st.divider()
        st.subheader("Audited Interviewee Verbatims")
        for _, row in quotes_df.iterrows():
            badge = "✅ VERIFIED" if row["verified"] else "❌ FAILED"
            with st.expander(f"{row['id']} | {row['speaker']} — {badge} ({row['match_type']})"):
                st.write(f"**Verbatim Quote:** \"{row['quote']}\"")
                st.write(f"**Assigned Dynamic Theme:** `{row['theme']}`")
                st.write(f"**Context:** `{row['context']}`")
                st.write(f"**Source Document:** `{row['file']}`")
                st.write(f"**Verification Match:** `{row['match_type']}` (Similarity: {row['similarity_score']})")

# Tab 3: Grounded Q&A
with tab3:
    st.subheader("Ask the Interview Evidence")
    query = st.text_input(
        "Enter your strategic question:",
        placeholder="e.g., What are the primary root causes driving the need for organizational change?"
    )

    if st.button("Submit Question"):
        if not api_key:
            st.error("System configuration error: GEMINI_API_KEY is missing from app secrets.")
        elif not st.session_state.evidence_data:
            st.error("Please run the pipeline on your interview files first.")
        elif not query:
            st.warning("Please type a question.")
        else:
            with st.spinner("Synthesizing answer grounded in verified evidence..."):
                try:
                    res = ask_evidence_query(query, st.session_state.evidence_data, api_key)
                    st.markdown("### Synthesized Finding")
                    st.write(res["answer"])

                    if res["cited_evidence"]:
                        st.markdown("#### Direct Verified Citations")
                        for ev in res["cited_evidence"]:
                            st.success(
                                f"**\"{ev['quote']}\"**\n\n— *{ev['speaker']} ({ev['file']})*\n\n"
                                f"*Dynamic Theme: {ev['theme']} | Context: {ev['context']}*"
                            )
                except Exception as ex:
                    st.error(f"Query Error: {str(ex)}")

# Tab 4: Linguistic Methodology
with tab4:
    st.header("📖 Quote Qualification Methodology & Linguistic Rules")
    st.markdown(
        """
        Consulting interview notes blend **verbatim quotes**, **relayed third-party feedback**, and **telegraphic notetaker summaries**. 
        This application executes a **deterministic, rules-based linguistic filter** coupled with **dynamic inductive thematic discovery**.
        """
    )
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 1. Inductive Thematic Synthesis")
        st.info("Analyzes the entire uploaded quote corpus dynamically to discover 4-6 systemic transformation pillars specific to what stakeholders shared.")

        st.markdown("#### 2. Subheading Semantic Scoping")
        st.info("Subheadings containing `GRANTEE` or `SURVEY` are isolated as **Grantee Voice** so external feedback is never misattributed to the interviewee.")

        st.markdown("#### 3. Thought Overflow Stitching")
        st.info("Connects multi-bullet sentence overflows starting with conjunctions (`but`, `because`, `so`) into unified executive utterances.")
    with c2:
        st.markdown("#### 4. The Finite Verb Test")
        st.success("Excludes pure notetaker noun lists (*'Training costs'*, *'Facilities'*); requires active finite predicates.")

        st.markdown("#### 5. Deictic Spoken Voice Grounding")
        st.success("Anchors quotes in first-person executive voice (`I`, `we`, `our`, `us`) and rhetorical leadership questions.")

        st.markdown("#### 6. Shorthand Prefix Stripping")
        st.success("Strips analytical notetaker prefixes (`PAIN POINTS —`) to isolate the speaker's true words, preserving context in an audit column.")
