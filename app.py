import os
import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Designs Law Assistant — IP India",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import config
import rag

# Custom styling for a polished, academic, professional look
st.markdown("""
<style>
    .main-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #E0E7FF;
        color: #3730A3;
        margin-right: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .badge-doc {
        background-color: #DCFCE7;
        color: #166534;
    }
    .answer-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2563EB;
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 1.2rem;
        margin-bottom: 1rem;
    }
    .not-found-card {
        background-color: #FFF1F2;
        border: 1px solid #FECDD3;
        border-left: 5px solid #E11D48;
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 1.2rem;
        margin-bottom: 1rem;
    }
    .ref-card {
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 0.9rem;
        margin-top: 0.8rem;
        font-size: 0.92rem;
    }
    .ref-header {
        font-weight: 700;
        color: #334155;
        margin-bottom: 0.3rem;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def check_prerequisites():
    """Validates the availability of PDF and ChromaDB."""
    pdf_exists = config.PDF_PATH.exists()
    chroma_exists = config.CHROMA_DIR.exists() and any(config.CHROMA_DIR.iterdir())
    return pdf_exists, chroma_exists


def main():
    # Sidebar
    with st.sidebar:
        st.header("⚖️ System Controls")
        
        pdf_exists, chroma_exists = check_prerequisites()
        
        # Document Status
        st.subheader("Document Status")
        if pdf_exists:
            st.success("✅ `designs_manual.pdf` (87 Pages)")
        else:
            st.error("❌ PDF not found in `data/`")
            st.caption("Place `designs_manual.pdf` into the `data/` directory.")

        # Vector Database Status
        st.subheader("Vector Database")
        if chroma_exists:
            st.success("✅ ChromaDB Ready")
        else:
            st.warning("⚠️ ChromaDB not built")
            st.caption("Run `python ingest.py` in your terminal to index the PDF.")

        st.divider()

        # LLM Provider Configuration
        st.subheader("LLM Configuration")
        provider_choice = st.selectbox(
            "Select LLM Provider",
            options=["Auto-detect", "Groq", "Google Gemini", "OpenAI", "Ollama"],
            index=0
        )
        
        api_key_input = st.text_input(
            "API Key (Optional override)",
            type="password",
            help="If provided, overrides .env for this session"
        )
        if api_key_input:
            if provider_choice == "Groq":
                config.GROQ_API_KEY = api_key_input
                config.DEFAULT_PROVIDER = "groq"
            elif provider_choice == "Google Gemini":
                config.GOOGLE_API_KEY = api_key_input
                config.DEFAULT_PROVIDER = "gemini"
            elif provider_choice == "OpenAI":
                config.OPENAI_API_KEY = api_key_input
                config.DEFAULT_PROVIDER = "openai"

        st.divider()

        # Mandatory Questions Quick Launcher
        st.subheader("📋 Assignment Test Questions")
        test_questions = [
            "How long does a proprietor have copyright in a registered design, and can it be extended?",
            "Which classification system is used to register designs?",
            "What is the government fee to register a trademark?",
            "Who is the current Controller General of Patents?",
            "What is the fee to register a design?"
        ]

        for i, q in enumerate(test_questions, 1):
            if st.button(f"Q{i}: {q[:32]}...", key=f"q_{i}", use_container_width=True):
                st.session_state["main_input"] = q
                st.session_state["auto_ask"] = True

    # Main Area
    st.markdown('<div class="main-title">Designs Law Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Ask questions about industrial design registration in India based on the '
        '<b>Manual of Designs Practice & Procedure</b> (CGPDTM, IP India).</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<span class="badge">RAG System</span>'
        '<span class="badge badge-doc">Manual: 87 Pages</span>'
        '<span class="badge">Strict Grounding</span>',
        unsafe_allow_html=True
    )

    # Initial prerequisite error handling
    if not pdf_exists:
        st.error(
            "🚨 **PDF not found.** Please place `designs_manual.pdf` inside the `data/` folder "
            "and run `python ingest.py`."
        )
        return

    if not chroma_exists:
        st.error(
            "🚨 **ChromaDB vector database not initialized.** "
            "Please run `python ingest.py` in your terminal to index the document."
        )
        return

    # Question Input
    if "main_input" not in st.session_state:
        st.session_state["main_input"] = ""

    user_question = st.text_input(
        "Ask your question:",
        key="main_input",
        placeholder="e.g. How long does a proprietor have copyright in a registered design?"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

    # Trigger search either on Ask button click or auto_ask from sidebar
    should_search = ask_button or st.session_state.get("auto_ask", False)
    if st.session_state.get("auto_ask", False):
        st.session_state["auto_ask"] = False

    if should_search:
        if not user_question.strip():
            st.warning("⚠️ Please enter a question.")
            return

        with st.spinner("Searching Manual of Designs Practice & Procedure..."):
            try:
                result = rag.answer_question(user_question)
            except Exception as e:
                err_msg = str(e)
                if "API key" in err_msg or "API_KEY" in err_msg:
                    st.error(f"🔑 **API Key Error**: {err_msg}")
                    st.info("💡 Please set your `GROQ_API_KEY`, `GOOGLE_API_KEY`, or `OPENAI_API_KEY` in `.env` or in the sidebar.")
                else:
                    st.error(f"Execution Error: {err_msg}")
                return

        answer_text = result["answer"]
        ref_text = result["reference"]
        is_found = result["found"]
        chunks = result["raw_chunks"]

        # Display Answer
        if is_found:
            st.markdown("### Answer")
            st.markdown(
                f'<div class="answer-card">{answer_text}</div>',
                unsafe_allow_html=True
            )

            # Display Reference
            st.markdown("### Reference")
            st.markdown(
                f'<div class="ref-card"><div class="ref-header">Grounding Source:</div>'
                f'<pre style="white-space: pre-wrap; margin: 0; font-family: inherit;">{ref_text}</pre></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown("### Answer")
            st.markdown(
                f'<div class="not-found-card">'
                f'<h4 style="margin: 0; color: #9F1239;">Not found</h4>'
                f'<p style="margin-top: 0.5rem; margin-bottom: 0; color: #4C0519; font-size: 0.95rem;">'
                f'The requested information is not present or cannot be reliably supported by the '
                f'<b>Manual of Designs Practice & Procedure</b>. The system strictly adheres to document '
                f'grounding and never utilizes external/unverified sources.'
                f'</p></div>',
                unsafe_allow_html=True
            )

            st.markdown("### Reference")
            st.markdown(
                '<div class="ref-card"><div class="ref-header">Reference:</div>—</div>',
                unsafe_allow_html=True
            )

        # Expandable transparency section for viva and evaluation
        if chunks:
            with st.expander("🔍 View Retrieved PDF Context Chunks (Evaluation & Viva)"):
                for idx, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {idx}:**")
                    st.text(chunk)
                    st.divider()


if __name__ == "__main__":
    main()
