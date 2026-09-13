"""
app.py
Streamlit web application for the GNP Evidence Pipeline.
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
st.caption("Deterministic quote verification and structured synthesis for organizational diagnosis.")

# Sidebar Controls
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

uploaded_files = st.sidebar.file_uploader(
    "Upload Interview Files (.txt)",
    type=["txt"],
    accept_multiple_files=True
)

if "evidence_data" not in st.session_state:
    st.session_state.evidence_data = []

if st.sidebar.button("Run Evidence Pipeline", type="primary"):
    if not api_key:
        st.sidebar.error("Please enter your Gemini API Key.")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one .txt interview file.")
    else:
        with st.spinner("Analyzing text and verifying quotes against raw files..."):
            try:
                files_dict = {f.name: f.read().decode("utf-8") for f in uploaded_files}
                st.session_state.evidence_data = run_extraction_pipeline(files_dict, api_key)
                st.sidebar.success(f"Successfully extracted {len(st.session_state.evidence_data)} evidence points!")
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

        col1, col2, col3 = st.columns(3)
        with col1:
            theme_choice = st.selectbox("Filter Theme", ["All"] + sorted(df["theme"].unique().tolist()))
        with col2:
            speaker_choice = st.selectbox("Filter Speaker", ["All"] + sorted(df["speaker"].unique().tolist()))
        with col3:
            verified_filter = st.checkbox("Show Only Verified Quotes", value=True)

        filtered = df.copy()
        if theme_choice != "All":
            filtered = filtered[filtered["theme"] == theme_choice]
        if speaker_choice != "All":
            filtered = filtered[filtered["speaker"] == speaker_choice]
        if verified_filter:
            filtered = filtered[filtered["verified"] == True]

        st.dataframe(
            filtered[["id", "theme", "speaker", "quote", "file", "verified", "match_type"]],
            use_container_width=True,
            height=450
        )

# Tab 2: Verification Report
with tab2:
    if not st.session_state.evidence_data:
        st.info("Run the pipeline to generate an integrity audit report.")
    else:
        df = pd.DataFrame(st.session_state.evidence_data)
        total = len(df)
        verified_count = int(df["verified"].sum())
        fails = total - verified_count
        pass_rate = round((verified_count / total) * 100, 1) if total > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Quotes Audited", total)
        m2.metric("Verified Word-for-Word", verified_count)
        m3.metric("Verification Discrepancies", fails)
        m4.metric("Integrity Pass Rate", f"{pass_rate}%")

        st.divider()
        st.subheader("Itemized Verification Audit Trail")
        for _, row in df.iterrows():
            badge = "✅ VERIFIED" if row["verified"] else "❌ FAILED"
            with st.expander(f"{row['id']} | {row['speaker']} — {badge} ({row['match_type']})"):
                st.write(f"**Extracted Quote:** \"{row['quote']}\"")
                st.write(f"**Source Document:** `{row['file']}`")
                st.write(f"**Audit Status:** {row['audit_details']}")
                st.write(f"**Similarity Score:** {row['similarity_score']}")

# Tab 3: Grounded Q&A
with tab3:
    st.subheader("Ask the Interview Evidence")
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the primary barriers to cross-functional collaboration?"
    )

    if st.button("Submit Question"):
        if not api_key:
            st.error("Gemini API key is required.")
        elif not st.session_state.evidence_data:
            st.error("Please run the pipeline on your interview files first.")
        elif not query:
            st.warning("Please type a question.")
        else:
            with st.spinner("Querying grounded evidence base..."):
                try:
                    res = ask_evidence_query(query, st.session_state.evidence_data, api_key)
                    st.markdown("### Synthesized Finding")
                    st.write(res["answer"])

                    if res["cited_evidence"]:
                        st.markdown("#### Direct Verified Citations")
                        for ev in res["cited_evidence"]:
                            st.success(f"**\"{ev['quote']}\"**\n\n— *{ev['speaker']} ({ev['file']})* [Status: {ev['match_type']}]")
                except Exception as ex:
                    st.error(f"Query Error: {str(ex)}")
