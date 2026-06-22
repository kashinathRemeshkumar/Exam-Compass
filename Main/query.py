"""
query.py — Ask a question, get an answer with citations.

Pipeline:
  1. Embed the question locally (same embedding model used during ingest).
  2. Retrieve the top-K most relevant chunks from ChromaDB (local, free).
  3. Send those chunks + the question to the Colab-hosted LLM over HTTP.
  4. Verify the model's cited pages actually match what was retrieved.
  5. Print the answer + a verification flag.

Before running:
  - Make sure embed_text.py has already been run (chroma_db/ exists and is populated).
  - Make sure your Colab notebook is running and you've copied its
    Cloudflare URL into COLAB_ENDPOINT below.
"""

import re
import requests
import chromadb
from sentence_transformers import SentenceTransformer

# ---- CONFIG ----
COLAB_ENDPOINT = "https://explains-entry-rolling-carol.trycloudflare.com/generate"
COLLECTION_NAME = "exam_prep"
TOP_K = 4
REQUEST_TIMEOUT = 120  # seconds — generation on a free GPU can be slow

# ---- SETUP (runs once when script starts) ----
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to local ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name=COLLECTION_NAME)


def retrieve(question: str, top_k: int = TOP_K, filters: dict | None = None):
    """
    Embed the question and fetch the most similar chunks from ChromaDB.
    `filters` lets you narrow by metadata, e.g. {"subject": "Biology", "grade": "11"}
    """
    query_embedding = embedder.encode(question).tolist()

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }
    if filters:
        query_kwargs["where"] = filters

    return collection.query(**query_kwargs)


def build_context(results) -> str:
    """Format retrieved chunks into a labeled context block for the LLM."""
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        return ""

    blocks = []
    for doc, meta in zip(documents, metadatas):
        if not doc:
            continue

        meta = meta or {}  # guard against None metadata for a given chunk

        label = (
            f"[Subject: {meta.get('subject', '?')} | "
            f"Grade: {meta.get('grade', '?')} | "
            f"Book: {meta.get('book', '?')} | "
            f"Page: {meta.get('page', '?')}]"
        )
        blocks.append(f"{label}\n{doc}")

    return "\n\n---\n\n".join(blocks)


def get_retrieved_pages(results) -> set[tuple[str, str]]:
    """
    Build the set of (book, page) pairs that were actually retrieved,
    so we can later check whether the model's cited sources are real.
    """
    metadatas = results["metadatas"][0]
    pages = set()
    for meta in metadatas:
        if not meta:
            continue
        book = str(meta.get("book", "")).strip()
        page = str(meta.get("page", "")).strip()
        if book and page:
            pages.add((book, page))
    return pages


def extract_cited_pages(answer_text: str) -> list[tuple[str, str]]:
    """
    Parse citations out of the model's answer text.
    Expects the format: (Source: <book>, p.<page>) but is built to also
    handle multi-page citations like (Source: <book>, p.23, 27, 54-55),
    since the model doesn't always cite a single page per source.
    Returns a list of (book, page) tuples — one per individual page number,
    with page ranges expanded into individual pages.
    """
    # Step 1: find each "Source: <book>, p./page/pages <numbers>" block.
    # The model doesn't always phrase this the same way every time (e.g.
    # "p.13" vs "page 13" vs "pages 7, 17, 18"), so we accept all of them.
    block_pattern = r"Source:\s*([^,]+?),\s*pages?\.?\s*([\d,\-\s]+)"
    blocks = re.findall(block_pattern, answer_text, flags=re.IGNORECASE)

    results = []
    for book, page_blob in blocks:
        book = book.strip()
        # Step 2: split the page blob on commas, then expand ranges like "54-55"
        for part in page_blob.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_str, _, end_str = part.partition("-")
                start_str, end_str = start_str.strip(), end_str.strip()
                if start_str.isdigit() and end_str.isdigit():
                    for page_num in range(int(start_str), int(end_str) + 1):
                        results.append((book, str(page_num)))
                    continue
            if part.isdigit():
                results.append((book, part))

    return results


def verify_citations(answer_text: str, retrieved_pages: set[tuple[str, str]]) -> dict:
    """
    Cross-check every page number the model cited against what was
    actually retrieved. This does NOT verify the book name strictly
    (since the model may shorten/paraphrase book titles) — it checks
    whether the cited PAGE NUMBER appears anywhere in the retrieved set,
    which is the simplest reliable signal that the model isn't just
    inventing a plausible-sounding page.
    """
    cited = extract_cited_pages(answer_text)
    retrieved_page_numbers = {page for (_, page) in retrieved_pages}

    if not cited:
        return {"status": "NO_CITATIONS_FOUND", "cited": [], "unverified": []}

    unverified = [
        (book, page) for (book, page) in cited
        if page not in retrieved_page_numbers
    ]

    status = "VERIFIED" if not unverified else "WARNING_UNVERIFIED_CITATION"
    return {"status": status, "cited": cited, "unverified": unverified}


def call_colab_model(system_prompt: str, user_message: str) -> str:
    """Send a generation request to the Colab-hosted model over HTTP."""
    try:
        response = requests.post(
            COLAB_ENDPOINT,
            json={"system_prompt": system_prompt, "user_message": user_message},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return f"[Model error: {data['error']}]"

        return data.get("response", "[No response field in model output]")

    except requests.exceptions.ConnectionError:
        return ("[Could not connect to Colab endpoint. Is the notebook running, "
                 "and is COLAB_ENDPOINT set to the current ngrok URL?]")
    except requests.exceptions.Timeout:
        return "[Request to Colab timed out. The model may be overloaded or the GPU is slow.]"
    except Exception as e:
        return f"[Unexpected error calling Colab: {e}]"


SYSTEM_PROMPT = (
    "You are a NEET/JEE exam tutor. Answer the student's question using ONLY "
    "the information in the provided context below. "
    "For every claim you make, cite the source like this: (Source: <book>, p.<page>). "
    "If the context does not contain enough information to answer the question, "
    "say so clearly instead of guessing or using outside knowledge."
)


def ask(question: str, filters: dict | None = None) -> dict:
    """
    Returns a dict: {"answer": str, "verification": dict}
    so callers can show both the answer and whether its citations checked out.
    """
    results = retrieve(question, filters=filters)
    context = build_context(results)

    if not context:
        return {
            "answer": "No relevant content found in the database for this question.",
            "verification": {"status": "NO_CONTEXT", "cited": [], "unverified": []},
        }

    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    answer = call_colab_model(SYSTEM_PROMPT, user_message)

    retrieved_pages = get_retrieved_pages(results)
    verification = verify_citations(answer, retrieved_pages)

    return {"answer": answer, "verification": verification}


def print_answer(result: dict):
    """Print just the answer text."""
    print("\n--- Answer ---")
    print(result["answer"])
    print()


if __name__ == "__main__":
    print("\nNEET/JEE RAG Tutor — type 'quit' to exit.\n")
    print("Tip: you can filter by subject, e.g. type '/subject Biology' before asking.\n")

    active_filters = {}

    KNOWN_SUBJECTS = {"biology", "physics", "chemistry", "mathematics"}

    while True:
        q = input("Ask a question: ").strip()

        if q.lower() == "quit":
            break

        if q.startswith("/subject "):
            remainder = q[len("/subject "):].strip()
            # Try to split off a known subject name from the start of the
            # remainder, so "/subject Biology What is..." works on one line
            # as well as "/subject Biology" on its own.
            words = remainder.split(" ", 1)
            first_word = words[0].lower() if words else ""

            if first_word in KNOWN_SUBJECTS:
                active_filters["subject"] = words[0].capitalize()
                print(f"(Filter set: subject = {words[0].capitalize()})\n")
                # If there's a question attached on the same line, ask it now
                if len(words) > 1 and words[1].strip():
                    result = ask(words[1].strip(), filters=active_filters)
                    print_answer(result)
                continue
            else:
                print(f"(Unrecognized subject '{remainder}'. "
                      f"Known subjects: {', '.join(s.capitalize() for s in KNOWN_SUBJECTS)})\n")
                continue

        if q.startswith("/clear"):
            active_filters = {}
            print("(Filters cleared)\n")
            continue

        if not q:
            continue

        answer = ask(q, filters=active_filters or None)
        print_answer(answer)