"""
app.py
Streamlit web application for the GNP Evidence Pipeline.
Features subheading-aware speaker attribution, quote filtering, and audit tracking.
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
st.caption("Subheading-aware quote extraction, authentic speaker attribution, and deterministic verification.")

api_key = st.secrets.get("GEMINI_API_KEY", "").strip()

st.sidebar.header("Evidence Ingestion")
uploaded_files = st.sidebar.file_uploader(
    "Upload Interview Files (.txt)",
    type=["txt"],
    accept_multiple_files=True
)

if "evidence_data" not in st.session_state:
    st.session_state.evidence_data = []

if st.sidebar.button("Run Evidence Pipeline", type="primary"):
    if not api_key:
        st.sidebar.error("System configuration error: GEMINI_API_KEY is missing from app secrets.")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one .txt interview file.")
    else:
        with st.spinner("Classifying subheadings, attributing speakers, and verifying text..."):
            try:
                files_dict = {f.name: f.read().decode("utf-8") for f in uploaded_files}
                st.session_state.evidence_data = run_extraction_pipeline(files_dict, api_key)
                quotes_count = sum(1 for e in st.session_state.evidence_data if e["is_interviewee_quote"])
                st.sidebar.success(
                    f"Extracted {len(st.session_state.evidence_data)} total items "
                    f"({quotes_count} verified interviewee quotes, "
                    f"{len(st.session_state.evidence_data) - quotes_count} proxy notes/summaries)!"
                )
            except Exception as ex:
                st.error(f"Pipeline Error: {str(ex)}")

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
            with st.expander(f"{row['id']} | {row['speaker']} [{row['evidence_type']}] — {badge}"):
                st.write(f"**Verbatim Quote:** \"{row['quote']}\"")
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
                                f"**\"{ev['quote']}\"**\n\n— *{ev['speaker']} ({ev['file']})* [Type: {ev['evidence_type']}]"
                            )
                except Exception as ex:
                    st.error(f"Query Error: {str(ex)}")
