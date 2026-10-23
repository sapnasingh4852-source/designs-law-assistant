import os
import sys
import re
from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config

def extract_chapter_and_section_headings(pages_text):
    """
    Extracts chapter and section mapping across all pages of the PDF.
    Chapters follow patterns like:
      - CHAPTER-1 / CHAPTER 1
      - Chapter 1 – Introduction
    Sections follow patterns like:
      - 01.01 ...
      - 03.06.02.06 Classification of designs
      - 06.04 Period of protection
      - 07.02 Extension of copyright (renewal)
    """
    chapter_map = {}
    current_chapter = "General / Preliminary"
    
    # Known chapter headings from table of contents as reference points for validation
    chapter_titles = {
        1: "Chapter 1 – Introduction",
        2: "Chapter 2 – Important definitions",
        3: "Chapter 3 – Filing of Design Application",
        4: "Chapter 4 – Examination and registration of designs",
        5: "Chapter 5 – Patent Office Journal",
        6: "Chapter 6 – Piracy of registered designs and consequences thereof",
        7: "Chapter 7 – Post registration procedures",
        8: "Chapter 8 – Transfer of rights",
        9: "Chapter 9 – General Services",
        10: "Chapter 10 – Use and acquisition of registered design by Government",
        11: "Chapter 11 – General powers of Controller",
        12: "Chapter 12 – Evidences etc.",
        13: "Chapter 13 – Restrictive conditions in contract etc.",
        14: "Chapter 14 – Appeals",
        15: "Chapter 15 – Requirements before delivery on sales",
        16: "Chapter 16 – Timelines",
    }

    page_data = []

    for page_idx, text in enumerate(pages_text):
        page_num = page_idx + 1

        # Check for chapter title in page text
        chap_match = re.search(r'(?:CHAPTER\s*[-–]?\s*(\d+)|Chapter\s+(\d+))', text, re.IGNORECASE)
        if chap_match:
            chap_num = int(chap_match.group(1) or chap_match.group(2))
            if chap_num in chapter_titles:
                current_chapter = chapter_titles[chap_num]
        elif "annexure" in text.lower():
            if "important case laws" in text.lower():
                current_chapter = "Annexure I – Important Case Laws"
            elif "specimen novelty" in text.lower():
                current_chapter = "Annexure II – Specimen Statements"

        # Check for section title in page text
        sec_match = re.search(r'(\b\d{2}\.\d{2}(?:\.\d{2})*(?:\.\d{2})*\s+[A-Za-z][A-Za-z0-9\s,\(\)\-–/]+)', text)
        page_section = sec_match.group(1).strip().split('\n')[0] if sec_match else ""

        page_data.append({
            "page_num": page_num,
            "chapter": current_chapter,
            "default_section": page_section,
            "text": text
        })

    return page_data


def load_pdf_with_metadata(pdf_path: Path):
    """
    Loads all pages of the PDF, extracts text, and assigns 
    document-grounded chapter, section, and page metadata.
    """
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        print("Please place designs_manual.pdf inside the data folder.")
        sys.exit(1)

    print(f"Loading PDF from: {pdf_path}")
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    print(f"Total pages detected: {total_pages}")

    raw_pages = [page.extract_text() or "" for page in reader.pages]
    page_data = extract_chapter_and_section_headings(raw_pages)

    documents = []
    for item in page_data:
        text = item["text"].strip()
        if not text:
            continue

        doc = Document(
            page_content=text,
            metadata={
                "source": pdf_path.name,
                "page": item["page_num"],
                "chapter": item["chapter"],
                "section": item["default_section"],
            }
        )
        documents.append(doc)

    return documents


def refine_chunk_metadata(chunks):
    """
    Refines section metadata for each chunk by checking if a specific
    section heading (e.g. 06.04 Period of protection) starts within the chunk text.
    """
    for chunk in chunks:
        sec_match = re.search(r'(\b\d{2}\.\d{2}(?:\.\d{2})*(?:\.\d{2})*\s+[A-Za-z][A-Za-z0-9\s,\(\)\-–/]+)', chunk.page_content)
        if sec_match:
            candidate = sec_match.group(1).strip().split('\n')[0]
            # Ensure reasonable length for a section title
            if len(candidate) <= 80:
                chunk.metadata["section"] = candidate


def ingest_documents():
    """
    Main ingestion pipeline:
    1. Loads PDF and extracts text from all pages
    2. Chunks documents using RecursiveCharacterTextSplitter
    3. Generates embeddings using sentence-transformers/all-MiniLM-L6-v2
    4. Stores and persists embeddings in ChromaDB
    """
    print("=" * 60)
    print("Starting Ingestion Pipeline for Designs Law Manual")
    print("=" * 60)

    # 1. Load PDF
    docs = load_pdf_with_metadata(config.PDF_PATH)
    print(f"Successfully loaded {len(docs)} pages with text.")

    # 2. Text Splitting
    print(f"Splitting text with chunk_size={config.CHUNK_SIZE}, chunk_overlap={config.CHUNK_OVERLAP}...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    chunks = splitter.split_documents(docs)
    refine_chunk_metadata(chunks)
    print(f"Generated {len(chunks)} text chunks.")

    # 3. Embedding model
    print(f"Initializing embedding model: {config.EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 4. ChromaDB persistence
    print(f"Creating persistent ChromaDB vector store at: {config.CHROMA_DIR}...")
    # Ensure directory exists
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(config.CHROMA_DIR),
        collection_name=config.COLLECTION_NAME
    )

    print("=" * 60)
    print(f"SUCCESS: {len(chunks)} chunks ingested into ChromaDB collection '{config.COLLECTION_NAME}'.")
    print(f"Persistent database saved at: {config.CHROMA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    ingest_documents()
