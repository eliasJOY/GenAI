# AI Resource Planning Assistant — Final Project (Advanced RAG)

## 1. Problem Statement

Large organizations maintain a pool of skilled resources (employees) that are not fully allocated to projects. When a new project opens, PMOs and delivery managers must scan spreadsheets or internal systems to identify the best-fit resources — a time-consuming, manual, and error-prone process. The goal of this project is to automate and improve resource selection by building an AI-powered assistant that:

- Accepts a job description (JD) as free text.
- Matches the JD against a CSV resource pool using semantic search + lexical search.
- Reranks candidates with a cross-encoder and domain-specific weighted scoring.
- Produces short, grounded explanations for the recommended resources.

**Outcome:** faster, more accurate resource assignment decisions and improved utilization.

---

## 2. Solution Overview

The **AI Resource Planning Assistant** is a Streamlit web app that implements an **Advanced Retrieval-Augmented Generation (RAG)** pipeline:

1. **Ingest**: Upload a CSV of pooled resources (one row per resource).
2. **Index**: Build sentence-transformer embeddings for each row (document) and a BM25 lexical index.
3. **Retrieve**: Hybrid retrieval combining dense vector similarity (FAISS or NumPy fallback) and BM25 keyword search.
4. **Rerank**: Use a Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to rerank the top candidates.
5. **Score**: Apply domain-specific, weighted scoring (skills overlap, experience, availability, rating, cost, certifications).
6. **Generate**: Use Google Gemini (via `google.generativeai`) to create a concise 1–2 sentence justification grounded in retrieved evidence.
7. **UX**: Streamlit UI to upload CSV, paste JD, tune weights, and download recommendations.

This is an **Advanced RAG** implementation because it uses:
- Hybrid retrieval (dense + lexical),
- Cross-encoder reranking (improves precision),
- LLM-based grounded generation for explanations.

---

## 3. Architecture & Design Rationale

### High-level architecture

[User JD & CSV] -> [Preprocess] -> [Embeddings + BM25] -> [Hybrid Retrieval] -> [CrossEncoder Rerank] -> [Weighted Scoring] -> [Gemini Justification] -> [Streamlit UI]


### Why RAG ?
- **Why Retrieval?** The candidate pool is structured, but each row acts like a small document. Retrieval allows precise grounding and prevents hallucination: LLM outputs are grounded in actual resource data.
- **Why Hybrid (Dense + BM25)?** Dense embeddings capture semantic matches (e.g., "ETL" ≈ "data ingestion") while BM25 ensures exact keyword matches (e.g., "Snowflake") are not missed. Hybrid improves recall and precision.
- **Why Cross-Encoder Reranker?** Embedding similarity is fast but sometimes noisy. A cross-encoder evaluates (query, doc) pairs more precisely and improves final ranking—especially important in enterprise data where small differences matter.
- **Why Weighted Scoring Layer?** Business preferences (cost, rating, availability) matter beyond pure semantic match. The scoring layer lets users tune trade-offs.
- **Why Generate (Gemini)?** Short justifications make decisions auditable and interpretable. Grounding the LLM on retrieved evidence reduces hallucinations.

### Key implementation choices
- **Sentence-Transformers (`all-MiniLM-L6-v2`)** for compact, fast embeddings (good balance of speed & quality).
- **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)** for re-ranking (CPU-friendly and effective).
- **FAISS (optional)** for fast vector search; fallback to NumPy-based search if FAISS install fails.
- **Gemini via `google.generativeai`**: used for generating short explanations grounded in evidence.
- **Streamlit UI**: quick to prototype and shares easily.

---

## 4. Setup & Usage Instructions





