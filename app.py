"""
app.py
Streamlit web application for the GNP Evidence Pipeline.
Features:
1. Evidence Matrix (with Context preservation)
2. Deterministic Verification Report
3. Grounded Q&A (gemini-3.6-flash)
4. Linguistic Methodology & Filter Logic
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
        with st.spinner("Classifying subheadings, preserving context, and running audits..."):
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

# Four Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Evidence Matrix",
    "🔍 Verification Report",
    "💬 Grounded Q&A",
    "📖 Linguistic Methodology & Filter Logic"
])

# Tab 1: Evidence Matrix with Context Column
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
            filtered[["id", "theme", "speaker", "quote", "context", "file", "verified", "match_type"]],
            use_container_width=True,
            height=460,
            column_config={
                "quote": st.column_config.TextColumn("Verbatim Quote", width="medium"),
                "context": st.column_config.TextColumn("Context / Surrounding Notes", width="large"),
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
                st.write(f"**Context:** `{row['context']}`")
                st.write(f"**Source Document:** `{row['file']}`")
                st.write(f"**Verification Match:** `{row['match_type']}` (Similarity: {row['similarity_score']})")

# Tab 3: Grounded Q&A
with tab3:
    st.subheader("Ask the Interview Evidence")
    query = st.text_input(
        "Enter your strategic question:",
        placeholder="e.g., What are the biggest barriers to effective decision-making?"
    )

    if st.button("Submit Question"):
        if not api_key:
            st.error("System configuration error: GEMINI_API_KEY is missing from app secrets.")
        elif not st.session_state.evidence_data:
            st.error("Please run the pipeline on your interview files first.")
        elif not query:
            st.warning("Please type a question.")
        else:
            with st.spinner("Querying grounded evidence base using gemini-3.6-flash..."):
                try:
                    res = ask_evidence_query(query, st.session_state.evidence_data, api_key)
                    st.markdown("### Synthesized Finding")
                    st.write(res["answer"])

                    if res["cited_evidence"]:
                        st.markdown("#### Direct Verified Citations")
                        for ev in res["cited_evidence"]:
                            st.success(
                                f"**\"{ev['quote']}\"**\n\n— *{ev['speaker']} ({ev['file']})*\n\n"
                                f"*Context: {ev['context']}*"
                            )
                except Exception as ex:
                    st.error(f"Query Error: {str(ex)}")

# Tab 4: Linguistic Methodology & Filter Logic
with tab4:
    st.header("📖 Quote Qualification Methodology & Linguistic Rules")
    st.markdown(
        """
        Consulting interview notes are inherently heterogeneous: they blend **verbatim quotes**, 
        **relayed third-party feedback**, and **telegraphic notetaker summaries**. 
        
        To prevent subjective LLM extraction variances, this application executes a **deterministic, 
        rules-based linguistic filter** before any semantic processing occurs.
        """
    )
    
    st.divider()

    st.subheader("The 6-Stage Linguistic Qualification Framework")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 1. Subheading Semantic Scoping (Attribution Guard)")
        st.info(
            """
            **The Challenge:** Sections titled `REQUESTS FROM GRANTEES` or `GRANTEE MESSAGES` relay external complaints. 
            Treating first-person words here as executive quotes causes false attribution.
            
            **The Rule:** Subheadings containing `GRANTEE`, `SURVEY`, or `EXTERNAL` are tagged as **Grantee Voice**. 
            They are isolated from executive verbatims so they are never misattributed to the interviewee.
            """
        )

        st.markdown("#### 2. Multi-Bullet Thought Overflow Stitching")
        st.info(
            """
            **The Challenge:** Notetakers frequently break a single spoken sentence across multiple lines with a bullet.
            
            **The Rule:** If bullet $N+1$ begins with an adversative/causal connector (`but`, `because`, `so`, `and`, `e.g.`) 
            or a lowercase character, it is syntactically stitched to bullet $N$ as a single coherent utterance.
            """
        )

        st.markdown("#### 3. Shorthand Meta-Label & Prefix Stripping")
        st.info(
            """
            **The Challenge:** Bullets often begin with notetaker labels followed by a dash or colon 
            (e.g., *'PAIN POINTS — We just don't deliver well'*).
            
            **The Rule:** The prefix regex strips structural tags (`PAIN POINTS —`) to isolate the actual spoken words, 
            while retaining the original label in the **Context** column.
            """
        )

    with c2:
        st.markdown("#### 4. The Finite Verb Test (Excluding Shorthand)")
        st.success(
            """
            **The Challenge:** Notetakers write shorthand lists (e.g., *'Training costs'*, *'Facilities'*, *'Smart people'*). 
            These are observational notes, not spoken quotes.
            
            **The Rule:** An item must contain at least one conjugated **finite verb** (*is, are, have, want, struggle, deliver*). 
            Pure noun phrases are automatically routed to *Notetaker Notes*.
            """
        )

        st.markdown("#### 5. Deictic & Conversational Voice Grounding")
        st.success(
            """
            **The Challenge:** Distinguishing between third-person summary bullets and active spoken voice.
            
            **The Rule:** Candidate quotes must contain first-person pronouns (`I`, `we`, `our`, `us`, `I'm`, `we're`) 
            or take the form of executive rhetorical questions (*'How do we show up to our grantees...'*).
            """
        )

        st.markdown("#### 6. Disambiguation of Quotation Marks")
        st.success(
            """
            **The Challenge:** Quotation marks in notes are often used to name initiatives or buzzwords 
            (e.g., *'moving away from "command and control"'* or *'"My President\'s Discretionary Fund"'*).
            
            **The Rule:** Quoted substrings are only isolated as direct quotes if they constitute a complete grammatical clause 
            ($\ge 4$ words with a verb). Isolated nouns in quotes remain embedded within their full bullet context.
            """
        )

    st.divider()

    st.subheader("Comparison Table: How Content is Classified")
    
    classification_examples = [
        {
            "Raw Bullet from Notes": "- We just don't deliver well. The hierarchy causes leadership to struggle with how to get approvals",
            "Classification": "Interviewee Quote (Spoken Verbatim)",
            "Reasoning": "Active first-person voice ('We'), contains finite verbs ('deliver', 'struggle'), expresses speaker's direct experience."
        },
        {
            "Raw Bullet from Notes": "- \"Our current structure relies too much on 'one person at the top having the answer'\"",
            "Classification": "Interviewee Quote (Direct Quote)",
            "Reasoning": "Full grammatical clause enclosed in double quotes; contains finite verb ('relies')."
        },
        {
            "Raw Bullet from Notes": "- \"How does the Foundation show up in these communities?\" (under 'REQUESTS FROM GRANTEES')",
            "Classification": "Grantee Voice (Relayed Feedback)",
            "Reasoning": "Subheading identifies external stakeholder context; attributed to Grantee Feedback, not the Head of Learning."
        },
        {
            "Raw Bullet from Notes": "- Training costs / Facilities / Smart people",
            "Classification": "Notetaker Note (Paraphrase)",
            "Reasoning": "Lacks finite verbs and subject-predicate structure; identified as notetaker shorthand indexing."
        },
        {
            "Raw Bullet from Notes": "- We are committed to the success of CEO... / - BUT until we know what shifting roles looks like...",
            "Classification": "Stitched Interviewee Quote",
            "Reasoning": "Second bullet starts with 'BUT', indicating an adversative continuation; stitched into a unified statement."
        }
    ]
    st.table(pd.DataFrame(classification_examples))
