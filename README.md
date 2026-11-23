# LifeDesk AI – Intelligent Personal Document Assistant Using Hybrid RAG

LifeDesk AI is an AI-powered document assistant that enables users to upload, search, and retrieve information from personal documents using Retrieval-Augmented Generation (RAG).

## Features
- Supports PDF, DOCX, and TXT documents.
- Extracts and processes document content.
- Uses BM25 for keyword-based retrieval.
- Supports semantic search for context-aware retrieval.
- Combines keyword and semantic search.
- Integrates OpenAI GPT-4o-mini for answer generation.
- Provides document source references and page numbers.
- Includes evidence-based responses to reduce unsupported answers.
- Interactive Streamlit interface.

## Tech Stack
- Python
- Streamlit
- LangChain
- Retrieval-Augmented Generation (RAG)
- BM25
- Semantic Search
- OpenAI API
- GPT-4o-mini

## Installation

1. Clone the repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   ```

2. Navigate to the project folder:

   ```bash
   cd LifeDesk-AI
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure your OpenAI API key securely using an environment variable or Streamlit secrets.

5. Run the application:

   ```bash
   streamlit run app.py
   ```

## Security
Never upload API keys, private documents, or environment files to GitHub.

## Author
Developed as a Generative AI project to explore document intelligence and Retrieval-Augmented Generation.
