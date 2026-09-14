"""
app.py
Streamlit web application for the GNP Evidence Pipeline.
Features quote vs. paraphrase separation, persistent keys, and grounded Q&A.
"""

import streamlit as st
import pandas as pd
from pipeline import run_extraction_pipeline, ask_evidence_query

st.set_page_config(
    page_title="GNP Foundation | Evidence Pipeline",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ GNP Foundation — Qualitative Evidence & Verification Engine")
st.caption("Deterministic quote extraction, linguistic authenticity classification, and quote verification.")

# Sidebar Configuration
st.sidebar.header("Configuration")

default_key = ""
if "GEMINI_API_KEY" in st.secrets:
    default_key = st.secrets["GEMINI_API_KEY"]

if "saved_gemini_key" not in st.session_state:
    st.session_state.saved_gemini_key = default_key

api_key = st.sidebar.text_input(
    "Gemini API Key",
    value=st.session_state.saved_gemini_key,
    type="password",
    help="Key is preserved across sessions. Set once or save in Streamlit Secrets."
)

if api_key != st.session_state.saved_gemini_key:
    st.session_state.saved_gemini_key = api_key

uploaded_files = st.sidebar.file_uploader(
    "Upload Interview Files (.txt)",
    type=["txt"],
    accept_multiple_files=True
)

if "evidence_data" not in st.session_state:
    st.session_state.evidence_data = []

if st.sidebar.button("Run Evidence Pipeline", type="primary"):
    effective_key = api_key.strip() or st.session_state.saved_gemini_key.strip()
    if not effective_key:
        st.sidebar.error("Please enter your Gemini API Key.")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one .txt interview file.")
    else:
        with st.spinner("Classifying spoken quotes vs. paraphrases and running audits..."):
            try:
                files_dict = {f.name: f.read().decode("utf-8") for f in uploaded_files}
                st.session_state.evidence_data = run_extraction_pipeline(files_dict, effective_key)
                quotes_count = sum(1 for e in st.session_state.evidence_data if e["is_quote"])
                st.sidebar.success(f"Extracted {len(st.session_state.evidence_data)} total items ({quotes_count} verified quotes, {len(st.session_state.evidence_data) - quotes_count} notetaker paraphrases)!")
            except Exception as ex:
                st.error(f"Pipeline Error: {str(ex)}")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["📊 Evidence Matrix", "🔍 Verification Report", "💬 Grounded Q&A"])

# Tab 1: Evidence Matrix
with tab1:
    if not st.session_state.evidence_data:
        st.info("Upload the 5 interview files in the sidebar and click **Run Evidence Pipeline**.")
    else:
        df = pd.DataFrame(st.session_state.evidence_data)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            theme_choice = st.selectbox("Filter Theme", ["All"] + sorted(df["theme"].unique().tolist()))
        with col2:
            speaker_choice = st.selectbox("Filter Speaker", ["All"] + sorted(df["speaker"].unique().tolist()))
        with col3:
            evidence_view = st.selectbox(
                "Filter Evidence Type",
                ["True Quotes Only (Spoken/Direct)", "Direct Quotes Only (\"...\")", "Spoken Verbatims Only (1st person)", "All Items (Incl. Paraphrases)"]
            )
        with col4:
            verified_filter = st.checkbox("Verified Word-for-Word Only", value=True)

        filtered = df.copy()
        if theme_choice != "All":
            filtered = filtered[filtered["theme"] == theme_choice]
        if speaker_choice != "All":
            filtered = filtered[filtered["speaker"] == speaker_choice]
        
        # View filter logic
        if evidence_view == "True Quotes Only (Spoken/Direct)":
            filtered = filtered[filtered["is_quote"] == True]
        elif evidence_view == "Direct Quotes Only (\"...\")":
            filtered = filtered[filtered["evidence_type"] == "Direct Quote"]
        elif evidence_view == "Spoken Verbatims Only (1st person)":
            filtered = filtered[filtered["evidence_type"] == "Spoken Verbatim"]

        if verified_filter:
            filtered = filtered[filtered["verified"] == True]

        st.dataframe(
            filtered[["id", "theme", "speaker", "evidence_type", "quote", "file", "verified", "match_type"]],
            use_container_width=True,
            height=450
        )

# Tab 2: Verification Report
with tab2:
    if not st.session_state.evidence_data:
        st.info("Run the pipeline to generate an integrity audit report.")
    else:
        df = pd.DataFrame(st.session_state.evidence_data)
        quotes_df = df[df["is_quote"] == True]
        paraphrase_df = df[df["is_quote"] == False]

        total_quotes = len(quotes_df)
        verified_quotes = int(quotes_df["verified"].sum())
        quote_pass_rate = round((verified_quotes / total_quotes) * 100, 1) if total_quotes > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Spoken Quotes Audited", total_quotes)
        m2.metric("Verified Word-for-Word", verified_quotes)
        m3.metric("Notetaker Paraphrases Separated", len(paraphrase_df))
        m4.metric("Quote Integrity Pass Rate", f"{quote_pass_rate}%")

        st.divider()
        st.subheader("Audited Quotes (Natural Voice & Explicit Quotes)")
        for _, row in quotes_df.iterrows():
            badge = "✅ VERIFIED" if row["verified"] else "❌ FAILED"
            with st.expander(f"{row['id']} | {row['speaker']} [{row['evidence_type']}] — {badge}"):
                st.write(f"**Extracted Spoken Quote:** \"{row['quote']}\"")
                st.write(f"**Source Document:** `{row['file']}` ({row['context']})")
                st.write(f"**Verification Match:** `{row['match_type']}` (Similarity: {row['similarity_score']})")

# Tab 3: Grounded Q&A
with tab3:
    st.subheader("Ask the Interview Evidence")
    query = st.text_input(
        "Enter your strategic question:",
        placeholder="e.g., What did leadership say about decision-making bottlenecks?"
    )

    if st.button("Submit Question"):
        effective_key = api_key.strip() or st.session_state.saved_gemini_key.strip()
        if not effective_key:
            st.error("Gemini API key is required.")
        elif not st.session_state.evidence_data:
            st.error("Please run the pipeline on your interview files first.")
        elif not query:
            st.warning("Please type a question.")
        else:
            with st.spinner("Synthesizing answer grounded exclusively in verified spoken quotes..."):
                try:
                    res = ask_evidence_query(query, st.session_state.evidence_data, effective_key)
                    st.markdown("### Synthesized Finding")
                    st.write(res["answer"])

                    if res["cited_evidence"]:
                        st.markdown("#### Direct Verified Citations")
                        for ev in res["cited_evidence"]:
                            st.success(f"**\"{ev['quote']}\"**\n\n— *{ev['speaker']} ({ev['file']})* [Type: {ev['evidence_type']}]")
                except Exception as ex:
                    st.error(f"Query Error: {str(ex)}")
