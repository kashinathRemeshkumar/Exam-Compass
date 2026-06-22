"""
STAGE 2: Chunk the cached extracted text and embed it into ChromaDB.

Reads only from extracted_text/*.json (produced by extract_text.py).
Never touches the original PDFs. Safe to re-run any time you want to:
  - change CHUNK_SIZE / CHUNK_OVERLAP
  - switch embedding models
  - rebuild the DB from scratch (just delete ./chroma_db first)

This is the script you'll re-run often while tuning your RAG pipeline.
"""

import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

EXTRACTED_DIR = "extracted_text"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
COLLECTION_NAME = "exam_prep"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def main():
    chunk_id = 0
    json_count = 0
    failed_files = []

    for dirpath, _, filenames in os.walk(EXTRACTED_DIR):
        for fname in sorted(filenames):
            if not fname.lower().endswith(".json"):
                continue

            json_path = os.path.join(dirpath, fname)

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    record = json.load(f)

                json_count += 1
                print(f"Embedding: {json_path}  "
                      f"[grade={record['grade']}, subject={record['subject']}, "
                      f"type={record['doc_type']}, chapter={record['chapter']}]")

                # Batch embeddings per-document for speed
                texts_to_embed = []
                metadatas = []

                for page_entry in record["pages"]:
                    page_num = page_entry["page"]
                    page_text = page_entry["text"]

                    for chunk in chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP):
                        texts_to_embed.append(chunk)
                        metadatas.append({
                            "grade": record["grade"],
                            "subject": record["subject"],
                            "doc_type": record["doc_type"],
                            "chapter": record["chapter"] or "",
                            "book": record["book"],
                            "page": page_num,
                        })

                if not texts_to_embed:
                    print(f"  (no text found, skipping)")
                    continue

                # Embed all chunks for this document in one batch call (much faster)
                embeddings = embedder.encode(texts_to_embed, show_progress_bar=False).tolist()

                ids = [f"chunk_{chunk_id + i}" for i in range(len(texts_to_embed))]
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts_to_embed,
                    metadatas=metadatas,
                )
                chunk_id += len(texts_to_embed)

            except Exception as e:
                print(f"  FAILED: {json_path} -> {e}")
                failed_files.append((json_path, str(e)))
                continue  # move on to the next file instead of crashing the whole run

    print(f"\nDone. Processed {json_count} cached documents, {chunk_id} total chunks embedded.")
    if failed_files:
        print(f"\n{len(failed_files)} file(s) FAILED and were skipped:")
        for path, err in failed_files:
            print(f"  - {path}: {err}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()