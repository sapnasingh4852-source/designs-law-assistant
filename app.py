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

# Premium Google Fonts & Tailored Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 60%, #EEF2F6 100%);
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 2rem 2.2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 6px;
        height: 100%;
        background: linear-gradient(180deg, #2563EB 0%, #7C3AED 100%);
    }

    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        line-height: 1.6;
        margin-bottom: 1.2rem;
        max-width: 900px;
    }

    /* Modern Pill Badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }
    
    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 600;
        border-radius: 9999px;
    }
    
    .pill-blue {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
    
    .pill-green {
        background: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }
    
    .pill-purple {
        background: #F5F3FF;
        color: #6D28D9;
        border: 1px solid #DDD6FE;
    }

    /* Result Cards */
    .answer-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 6px solid #2563EB;
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        margin-top: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
        color: #1E293B;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    
    .card-header-tag {
        font-family: 'Outfit', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .tag-blue {
        color: #2563EB;
    }

    .not-found-card {
        background: #FFF5F5;
        border: 1px solid #FED7D7;
        border-left: 6px solid #E11D48;
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        margin-top: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px -2px rgba(225, 29, 72, 0.06);
    }

    .not-found-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #9F1239;
        margin-bottom: 0.4rem;
    }

    .not-found-desc {
        color: #4C0519;
        font-size: 0.98rem;
        line-height: 1.6;
        margin: 0;
    }

    /* Citation Reference Card */
    .ref-container {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.3rem 1.6rem;
        margin-top: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
    }
    
    .ref-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 0.92rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #334155;
        margin-bottom: 0.6rem;
    }

    .ref-content {
        color: #0F172A;
        font-size: 0.96rem;
        line-height: 1.65;
        font-weight: 500;
    }

    /* Sidebar Styling */
    .sidebar-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
    }
    
    .sidebar-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 0.25rem;
    }
    
    .sidebar-value {
        font-size: 0.92rem;
        font-weight: 700;
        color: #0F172A;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
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
        st.markdown("### ⚖️ System Status")
        
        pdf_exists, chroma_exists = check_prerequisites()
        
        # Document Card
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-label">Document Source</div>
                <div class="sidebar-value">{"✅ designs_manual.pdf (87 Pages)" if pdf_exists else "❌ PDF Missing"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Vector Store Card
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-label">Vector Database</div>
                <div class="sidebar-value">{"✅ ChromaDB Indexed (163 Chunks)" if chroma_exists else "⚠️ Not Initialized"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Security Status Card
        st.markdown(
            """
            <div class="sidebar-card">
                <div class="sidebar-label">Security & Model</div>
                <div class="sidebar-value">🔒 Configured via .env</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # Example Questions Quick Launcher
        st.markdown("### 💡 Example Questions")
        st.caption("Click any question below to test document grounding:")

        example_questions = [
            ("Q1: Copyright Duration & Extension", "How long does a proprietor have copyright in a registered design, and can it be extended?"),
            ("Q2: Locarno Classification System", "Which classification system is used to register designs?"),
            ("Q3: Trademark Govt Fee (Out-of-Scope)", "What is the government fee to register a trademark?"),
            ("Q4: Controller General Name (Out-of-Scope)", "Who is the current Controller General of Patents?"),
            ("Q5: Design Registration Fee", "What is the fee to register a design?")
        ]

        for label, q_text in example_questions:
            if st.button(label, key=f"btn_{label[:2]}", use_container_width=True):
                st.session_state["main_input"] = q_text
                st.session_state["auto_ask"] = True

    # Main Hero Section
    st.markdown(
        """
        <div class="hero-container">
            <div class="main-title">Designs Law Assistant</div>
            <div class="sub-title">
                Authoritative legal question-answering assistant for industrial designs in India, 
                grounded strictly in the <b>Manual of Designs Practice & Procedure</b> (CGPDTM, IP India).
            </div>
            <div class="badge-container">
                <span class="pill-badge pill-blue">🏛️ IP India CGPDTM</span>
                <span class="pill-badge pill-green">📖 87-Page Official Manual</span>
                <span class="pill-badge pill-purple">🛡️ Strict Document Grounding</span>
            </div>
        </div>
        """,
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

    # Question Input Field
    if "main_input" not in st.session_state:
        st.session_state["main_input"] = ""

    user_question = st.text_input(
        "Ask a question about design registration in India:",
        key="main_input",
        placeholder="e.g. How long does a proprietor have copyright in a registered design, and can it be extended?"
    )

    col1, col2 = st.columns([1.2, 6])
    with col1:
        ask_button = st.button("🔍 Search Manual", type="primary", use_container_width=True)

    should_search = ask_button or st.session_state.get("auto_ask", False)
    if st.session_state.get("auto_ask", False):
        st.session_state["auto_ask"] = False

    if should_search:
        if not user_question.strip():
            st.warning("⚠️ Please enter a question.")
            return

        with st.spinner("Searching official manual & verifying document citations..."):
            try:
                result = rag.answer_question(user_question)
            except Exception as e:
                err_msg = str(e)
                if "API key" in err_msg or "API_KEY" in err_msg:
                    st.error(f"🔑 **API Key Error**: {err_msg}")
                    st.info("💡 Please ensure your `GOOGLE_API_KEY` or `GROQ_API_KEY` is set in your `.env` file.")
                else:
                    st.error(f"Execution Error: {err_msg}")
                return

        answer_text = result["answer"]
        ref_text = result["reference"]
        is_found = result["found"]

        # Display Answer
        st.markdown("### Answer")
        if is_found:
            st.markdown(
                f"""
                <div class="answer-card">
                    <div class="card-header-tag tag-blue">🏛️ Grounded Manual Evidence</div>
                    <div>{answer_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Display Grounding Source Reference
            st.markdown("### Reference")
            st.markdown(
                f"""
                <div class="ref-container">
                    <div class="ref-title">📖 Source Citation from Document Metadata:</div>
                    <div class="ref-content">{ref_text.replace('\n', '<br>')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="not-found-card">
                    <div class="not-found-title">Not found</div>
                    <p class="not-found-desc">
                        The requested information is not present or cannot be reliably supported by the 
                        <b>Manual of Designs Practice & Procedure</b>. The system strictly adheres to document 
                        grounding and never utilizes external or speculative sources.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### Reference")
            st.markdown(
                """
                <div class="ref-container">
                    <div class="ref-title">📖 Source Citation:</div>
                    <div class="ref-content">— (Information unsupported by the supplied 87-page manual)</div>
                </div>
                """,
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()
