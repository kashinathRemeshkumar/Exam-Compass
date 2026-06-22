"""
STAGE 1: Extract text from all PDFs and cache to disk as JSON.

Run this ONCE (or whenever you add new PDFs). It never needs to touch
the actual PDF files again after this — everything downstream (chunking,
embedding, re-embedding with a different model, changing chunk size, etc.)
reads from the cached JSON instead of re-parsing PDFs.

Output: extracted_text/<same_relative_path>.json
Each JSON file contains a list of {page, text} for that PDF.
"""

import os
import re
import json
import fitz  # pymupdf

ROOT_DIR = "dataset"            # your PDF root folder
OUTPUT_DIR = "extracted_text"   # cached text goes here, mirrors folder structure

SUBJECT_CODES = {
    "kebo": "Biology",    "lebo": "Biology",
    "kech": "Chemistry",  "lech": "Chemistry",
    "kemh": "Mathematics","lemh": "Mathematics",
    "keph": "Physics",    "leph": "Physics",
}


def clean_text(text):
    """Strip known NCERT boilerplate and tidy whitespace."""
    text = re.sub(r"Reprint \d{4}-\d{2}", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def parse_filename(fname):
    """
    Derive subject / doc_type / chapter number from NCERT filename convention.
    Examples:
      kebo101.pdf -> Biology, chapter, "01"
      kemh1a1.pdf -> Mathematics, appendix, None
      keph1ps.pdf -> Physics, problem_set, None
      keph1an.pdf -> Physics, answers, None
    """
    base = os.path.splitext(fname)[0]
    prefix = base[:4]
    rest = base[4:]

    subject = SUBJECT_CODES.get(prefix, "Unknown")

    if rest.endswith("ps"):
        doc_type, chapter = "problem_set", None
    elif rest.endswith("an"):
        doc_type, chapter = "answers", None
    elif re.search(r"a\d", rest):
        doc_type, chapter = "appendix", None
    else:
        doc_type = "chapter"
        match = re.search(r"(\d{1,2})$", rest)
        chapter = match.group(1) if match else None

    return subject, doc_type, chapter


def main():
    pdf_count = 0
    skipped = []

    for dirpath, _, filenames in os.walk(ROOT_DIR):
        for fname in sorted(filenames):
            if not fname.lower().endswith(".pdf"):
                continue

            full_path = os.path.join(dirpath, fname)
            rel_dir = os.path.relpath(dirpath, ROOT_DIR)
            parts = rel_dir.split(os.sep)
            grade = parts[0] if len(parts) > 0 else "unknown"

            subject, doc_type, chapter = parse_filename(fname)

            # Mirror the folder structure under OUTPUT_DIR, swap .pdf -> .json
            out_subdir = os.path.join(OUTPUT_DIR, rel_dir)
            os.makedirs(out_subdir, exist_ok=True)
            out_path = os.path.join(out_subdir, os.path.splitext(fname)[0] + ".json")

            if os.path.exists(out_path):
                print(f"SKIP (already extracted): {full_path}")
                continue

            print(f"Extracting: {full_path}")

            try:
                doc = fitz.open(full_path)
            except Exception as e:
                print(f"  FAILED to open: {e}")
                skipped.append(full_path)
                continue

            pages = []
            for page_num, page in enumerate(doc, start=1):
                raw_text = page.get_text()
                text = clean_text(raw_text)
                if text:
                    pages.append({"page": page_num, "text": text})
            doc.close()

            record = {
                "source_pdf": full_path,
                "grade": grade,
                "subject": subject,
                "doc_type": doc_type,
                "chapter": chapter,
                "book": fname,
                "pages": pages,
            }

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            pdf_count += 1

    print(f"\nDone. Extracted {pdf_count} new PDFs.")
    if skipped:
        print(f"Failed to open {len(skipped)} files:")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()