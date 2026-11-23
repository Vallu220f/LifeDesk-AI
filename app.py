
import json
import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables BEFORE importing project modules
load_dotenv()

# Project modules
from src.document_loader import load_document
from src.chunking import chunk_documents
from src.retrieval import HybridRetriever
from src.answering import generate_answer
from evaluation.ragas_evaluator import run_ragas_evaluation


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="LifeDesk AI",
    layout="wide"
)

st.title("LifeDesk AI")
st.caption(
    "Intelligent Personal Document & Information Assistant | Hybrid RAG + RAGAS"
)
st.info(
    "Upload personal documents and ask questions. "
    "LifeDesk AI combines keyword search (BM25) with semantic vector search."
)


# --------------------------------------------------
# API KEY CONFIGURATION
# --------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    st.sidebar.success("OpenAI API key loaded successfully!")
else:
    st.sidebar.error(
        "OPENAI_API_KEY not found. Check your .env file."
    )


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

for key, default in {
    "documents": [],
    "retriever": None,
    "history": [],
    "last_eval": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# --------------------------------------------------
# SIDEBAR: DOCUMENT UPLOAD
# --------------------------------------------------

with st.sidebar:
    st.header("Document library")

    uploads = st.file_uploader(
        "Upload PDF, DOCX or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if st.button(
        "Process documents",
        type="primary",
        use_container_width=True
    ):
        if not uploads:
            st.warning("Upload at least one document first.")

        else:
            all_docs, errors = [], []
            progress = st.progress(0)

            for i, upload in enumerate(uploads):
                try:
                    all_docs.extend(
                        load_document(upload.name, upload.getvalue())
                    )
                except Exception as exc:
                    errors.append(f"{upload.name}: {exc}")

                progress.progress((i + 1) / len(uploads))

            if all_docs:
                chunks = chunk_documents(all_docs)

                with st.spinner(
                    "Building keyword and semantic indexes "
                    "(first run downloads the embedding model)..."
                ):
                    st.session_state.documents = chunks
                    st.session_state.retriever = HybridRetriever(chunks)

                st.session_state.history = []
                st.session_state.last_eval = None

                st.success(
                    f"Indexed {len(chunks)} chunks from "
                    f"{len(uploads) - len(errors)} file(s)."
                )

            for error in errors:
                st.error(error)

    

        for name in sorted({
            d["metadata"]["source"]
            for d in st.session_state.documents
        }):
            st.caption(f"• {name}")

        if st.button(
            "Clear document library",
            use_container_width=True
        ):
            for key, default in {
                "documents": [],
                "retriever": None,
                "history": [],
                "last_eval": None
            }.items():
                st.session_state[key] = default

            st.rerun()

    st.divider()

   


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

if st.session_state.retriever is None:

    st.subheader("Get started")

    st.markdown("""
    1. Upload a bill, warranty, insurance policy, bank statement or rental agreement.
    2. Click **Process documents**.
    3. Ask questions and inspect the source passages.
    """)

else:

    tab_chat, tab_eval = st.tabs([
        "💬 Ask documents",
        "📊 RAGAS evaluation"
    ])

    # --------------------------------------------------
    # CHAT TAB
    # --------------------------------------------------

    with tab_chat:

        st.subheader("Ask your documents")

        for item in st.session_state.history:

            with st.chat_message(item["role"]):

                st.markdown(item["content"])

                if item.get("sources"):

                    with st.expander("Sources used"):

                        for source in item["sources"]:
                            st.caption(source)

        question = st.chat_input(
            "Ask a question about your uploaded documents..."
        )

        if question:

            st.session_state.history.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching your documents and preparing an answer..."
                ):

                    try:
                        results = st.session_state.retriever.search(
                            question,
                            top_k=5
                        )

                        answer = generate_answer(question, results)

                    except Exception as exc:
                        answer = (
                            f"Sorry, I couldn't generate an answer. "
                            f"Error: {exc}"
                        )

                st.markdown(answer)

                source_lines = [
                    (
                        f"{r['metadata']['source']} — "
                        f"page {r['metadata'].get('page', 'N/A')} "
                        f"(hybrid score: {r['score']:.3f})"
                    )
                    for r in results
                ] if "results" in locals() else []

                if source_lines:
                    with st.expander("Sources used"):
                        for line in source_lines:
                            st.caption(line)

            st.session_state.history.append({
                "role": "assistant",
                "content": answer,
                "sources": source_lines
            })


    # --------------------------------------------------
    # RAGAS EVALUATION TAB
    # --------------------------------------------------

    with tab_eval:

        st.subheader("Evaluate answer and retrieval quality")

        st.write(
            "RAGAS evaluates generated answers against retrieved context "
            "and reference answers. Evaluation makes additional LLM/API "
            "calls and may incur API costs."
        )

        st.markdown(
            "Enter a small test set as JSON. "
            "Each item needs a `question` and a `reference` "
            "(your expected answer)."
        )

        default_tests = [
            {
                "question": "When does the example laptop warranty expire?",
                "reference": "The example laptop warranty expires on 15 January 2027."
            },
            {
                "question": "What is the total amount due on the sample utility bill?",
                "reference": "The total amount due is INR 1,450."
            }
        ]

        test_json = st.text_area(
            "Evaluation test cases (JSON)",
            value=json.dumps(default_tests, indent=2),
            height=190
        )

        if st.button(
            "Run RAGAS evaluation",
            type="primary"
        ):

            try:
                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY is missing. "
                        "Please configure your .env file."
                    )

                cases = json.loads(test_json)

                if not isinstance(cases, list) or not cases:
                    raise ValueError(
                        "Provide a non-empty JSON list."
                    )

                for item in cases:
                    if (
                        not isinstance(item, dict)
                        or not item.get("question")
                        or not item.get("reference")
                    ):
                        raise ValueError(
                            "Every test case must include question and reference."
                        )

                with st.spinner(
                    "Running retrieval, answer generation and RAGAS metrics..."
                ):

                    rows = []

                    for item in cases:

                        retrieved = st.session_state.retriever.search(
                            item["question"],
                            top_k=5
                        )

                        response = generate_answer(
                            item["question"],
                            retrieved
                        )

                        rows.append({
                            "question": item["question"],
                            "reference": item["reference"],
                            "response": response,
                            "retrieved_contexts": [
                                r["text"] for r in retrieved
                            ]
                        })

                    scores = run_ragas_evaluation(rows)

                st.session_state.last_eval = scores

                st.success("Evaluation completed.")

            except Exception as exc:

                st.error(
                    f"Evaluation could not be completed: {exc}"
                )

                st.caption(
                    "Check that OPENAI_API_KEY is set, dependencies "
                    "installed, and the evaluation JSON is valid."
                )

        if st.session_state.last_eval is not None:

            st.markdown("#### RAGAS scores")

            st.dataframe(
                st.session_state.last_eval,
                use_container_width=True
            )

            st.caption(
                "Scores are dataset- and model-dependent; "
                "interpret them alongside manual review."
            )

            st.download_button(
                "Download evaluation results (CSV)",
                st.session_state.last_eval.to_csv(index=False),
                "lifedesk_ragas_results.csv",
                "text/csv"
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
