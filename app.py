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
st.caption("Auditable, grounded qualitative evidence extraction for organizational operating model redesign.")

st.sidebar.header("1. Ingestion Configuration")
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
        st.sidebar.error("Please enter a Gemini API Key.")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one .txt interview file.")
    else:
        with st.spinner("Ingesting files and running deterministic verification..."):
            files_dict = {f.name: f.read().decode("utf-8") for f in uploaded_files}
            st.session_state.evidence_data = run_extraction_pipeline(files_dict, api_key)
            st.sidebar.success(f"Extracted {len(st.session_state.evidence_data)} evidence points!")

tab1, tab2, tab3 = st.tabs(["📊 Evidence Matrix", "🔍 Verification Report", "💬 Grounded Q&A"])

with tab1:
    if not st.session_state.evidence_data:
        st.info("Upload interview notes in the sidebar and click **Run Evidence Pipeline** to view findings.")
    else:
        df = pd.DataFrame(st.session_state.evidence_data)

        col1, col2, col3 = st.columns(3)
        with col1:
            selected_theme = st.selectbox("Filter by Theme", ["All"] + sorted(df["theme"].unique().tolist()))
        with col2:
            selected_speaker = st.selectbox("Filter by Speaker", ["All"] + sorted(df["speaker"].unique().tolist()))
        with col3:
            verified_only = st.checkbox("Verified Quotes Only", value=True)

        filtered_df = df.copy()
        if selected_theme != "All":
            filtered_df = filtered_df[filtered_df["theme"] == selected_theme]
        if selected_speaker != "All":
            filtered_df = filtered_df[filtered_df["speaker"] == selected_speaker]
        if verified_only:
            filtered_df = filtered_df[filtered_df["verified"] == True]

        st.dataframe(
            filtered_df[["id", "theme", "speaker", "quote", "file", "verified", "match_type"]],
            use_container_width=True,
            height=450
        )

with tab2:
    if not st.session_state.evidence_data:
        st.info("No audit report available. Run pipeline first.")
    else:
        df = pd.DataFrame(st.session_state.evidence_data)
        total = len(df)
        verified_count = int(df["verified"].sum())
        fail_count = total - verified_count
        pass_rate = round((verified_count / total) * 100, 1) if total > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Quotes Audited", total)
        m2.metric("Verified Word-for-Word", verified_count)
        m3.metric("Verification Failures", fail_count)
        m4.metric("Evidence Integrity Rate", f"{pass_rate}%")

        st.divider()
        st.subheader("Deterministic Audit Log")
        for _, row in df.iterrows():
            status_icon = "✅ VERIFIED" if row["verified"] else "❌ FAILED"
            with st.expander(f"{row['id']} | {row['speaker']} — {status_icon} ({row['match_type']})"):
                st.write(f"**Quote:** \"{row['quote']}\"")
                st.write(f"**Source File:** `{row['file']}`")
                st.write(f"**Audit Detail:** {row['audit_details']}")
                st.write(f"**Similarity Score:** {row['similarity_score']}")

with tab3:
    st.subheader("Query the Evidence Base")
    query = st.text_input(
        "Ask a strategic question:",
        placeholder="e.g., What are the primary bottlenecks in the grant approval process?"
    )

    if st.button("Submit Query"):
        if not api_key:
            st.error("API key required.")
        elif not st.session_state.evidence_data:
            st.error("Please run the pipeline on interview files first.")
        elif not query:
            st.warning("Please type a question.")
        else:
            with st.spinner("Synthesizing answer against verified evidence..."):
                res = ask_evidence_query(query, st.session_state.evidence_data, api_key)
                st.markdown("### Answer")
                st.write(res["answer"])

                if res["cited_evidence"]:
                    st.markdown("#### Supporting Verified Quotes")
                    for ev in res["cited_evidence"]:
                        st.success(f"**\"{ev['quote']}\"**\n\n— *{ev['speaker']} ({ev['file']})* [Verified: {ev['match_type']}]")
