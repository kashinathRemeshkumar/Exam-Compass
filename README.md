# Exam Compass

A local, citation-checked RAG (Retrieval-Augmented Generation) tutor for NEET/JEE exam prep, built on NCERT textbooks. Ask a question, get an answer grounded in the actual textbook pages — with the source page numbers verified against what was actually retrieved, not just trusted from the model.

> **Status: Work in progress.** Core pipeline (extract → embed → query) works end-to-end, but several pieces (see [Known Issues](#known-issues-and-planned-work) below) are unresolved. Expect rough edges.

## How it works

The pipeline runs in three stages:

1. **Extract** (`Main/extract_pdf_text.py`) — Parses NCERT PDFs from `dataset/`, strips boilerplate, and caches the text per-page as JSON under `extracted_text/`. Filenames are decoded using NCERT's naming convention to auto-tag each file with subject, grade, and chapter. Only needs to run once per new PDF.

2. **Embed** (`Main/embed_text.py`) — Chunks the cached text and embeds it with `all-MiniLM-L6-v2` (via `sentence-transformers`) into a local [ChromaDB](https://www.trychroma.com/) collection (`chroma_db/`). Safe to re-run any time you want to retune chunk size/overlap or rebuild from scratch.

3. **Query** (`Main/query.py`) — A CLI chat loop. Embeds your question, retrieves the top-K most relevant chunks from ChromaDB, and sends them + the question to an LLM for an answer. The LLM is required to cite `(Source: <book>, p.<page>)` for every claim, and the answer is cross-checked against what was actually retrieved — flagging any cited page number that doesn't show up in the retrieved context, as a lightweight hallucination guard.

The LLM itself is **not** run locally — it's hosted separately on a free Colab GPU (`Main/Ai_inference_server_V1.ipynb`) and exposed to `query.py` over a Cloudflare tunnel URL.

## Project layout

```
Main/
  extract_pdf_text.py       # Stage 1: PDF -> cached JSON text
  embed_text.py              # Stage 2: JSON -> ChromaDB embeddings
  query.py                   # Stage 3: ask questions, get cited answers
  Ai_inference_server_V1.ipynb  # Colab notebook hosting the LLM
  test.py                    # quick sanity check on chroma_db contents
dataset/                     # source NCERT PDFs (grade/subject folders)
extracted_text/              # cached per-page text, mirrors dataset/
chroma_db/                   # persistent Chroma vector store
identified_problems.txt      # running list of known issues / TODOs
```

## Setup

```bash
pip install chromadb sentence-transformers pymupdf requests
```

1. Drop your NCERT PDFs into `dataset/<grade>/...` following the standard NCERT filename convention (e.g. `kebo101.pdf`, `kemh1a1.pdf`).
2. Run the pipeline in order:
   ```bash
   python Main/extract_pdf_text.py
   python Main/embed_text.py
   ```
3. Open `Main/Ai_inference_server_V1.ipynb` in Colab, run it, and copy the Cloudflare tunnel URL it prints into `COLAB_ENDPOINT` at the top of `Main/query.py`.
4. Start asking questions:
   ```bash
   python Main/query.py
   ```
   Use `/subject Biology` to filter by subject, or `/clear` to reset filters.

## Known issues and planned work

Tracked in [`identified_problems.txt`](./identified_problems.txt):

- **No adaptive assessment loop** — the system doesn't yet track how a candidate is doing over time or adjust the study plan to prevent stagnation. Planned approach: surface different study-plan options and let the user pick.
- **No graceful "insufficient context" handling for weak retrievals** — needs work to make the model answer sensibly even when retrieved context doesn't fully cover the question, without it starting to hallucinate.
- **No skill leveling** — no mechanism yet to move a user from beginner to expert as they progress.

## License

GPL-3.0 — see [LICENSE](./LICENSE).
