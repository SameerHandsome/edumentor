"""
EduMentor RAG Data Ingestion Script
====================================
Run this ONCE before starting the server (or whenever you add new docs).
It is NOT called at runtime — Qdrant stores vectors permanently.

Supported file types:  .txt  .pdf  .md

Usage examples:
  # See all available topic slugs from your DB
  python scripts/ingest.py --list-topics

  # Ingest a single file
  python scripts/ingest.py --file data/algebra_textbook.pdf --topic-slug algebra --difficulty intermediate

  # Ingest an entire folder at once
  python scripts/ingest.py --folder data/mathematics/ --topic-slug mathematics

  # See what is already stored in Qdrant
  python scripts/ingest.py --list

  # Delete a document by its doc_id
  python scripts/ingest.py --delete <doc_id_here>
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

import pymupdf4llm

# Project root on path so app.* imports work when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Text extraction ───────────────────────────────────────────────────────────


def extract_text(filepath: Path) -> str:
    """Return clean Markdown text from .txt, .md, or .pdf file."""
    suffix = filepath.suffix.lower()

    if suffix in (".txt", ".md"):
        return filepath.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        # PyMuPDF4LLM preserves math formulas, tables, and document layout
        try:
            md_text = pymupdf4llm.to_markdown(str(filepath))
            return md_text
        except Exception as e:
            print(f"Error parsing PDF {filepath.name}: {e}")
            return ""

    raise ValueError(f"Unsupported file type: {suffix}")


# ── DB helpers ────────────────────────────────────────────────────────────────


async def resolve_topic_id(slug: str) -> str:
    """Look up topic UUID from slug. Prints available slugs and exits if not found."""
    from sqlalchemy import select

    from app.core.database import AsyncSessionLocal
    from app.models.topic import Topic

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Topic).where(Topic.slug == slug))
        topic = result.scalar_one_or_none()
        if not topic:
            print(f"\nERROR: No topic with slug '{slug}'")
            print("Run:  python scripts/ingest.py --list-topics\n")
            sys.exit(1)
        return str(topic.id)


async def print_topics() -> None:
    """Print all topics and slugs from the database."""
    from sqlalchemy import select

    from app.core.database import AsyncSessionLocal
    from app.models.topic import Topic

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Topic).order_by(Topic.grade_level, Topic.parent_id.nullsfirst(), Topic.name)
        )
        topics = result.scalars().all()

    print("\n=== Available Topics (use the SLUG with --topic-slug) ===\n")
    print(f"  {'SLUG':35s}  {'NAME':25s}  GRADE  LEVEL")
    print("  " + "-" * 75)
    for t in topics:
        level = "subtopic" if t.parent_id else "PARENT"
        indent = "    " if t.parent_id else ""
        print(f"  {indent}{t.slug:31s}  {t.name:25s}  {t.grade_level:5d}  {level}")
    print()


# ── List / Delete ─────────────────────────────────────────────────────────────


async def cmd_list(client) -> None:
    """Print all documents currently stored in Qdrant curriculum_docs."""
    print("\n=== Documents currently in Qdrant (curriculum_docs) ===\n")
    seen: dict = {}
    offset = None

    while True:
        points, next_offset = await client.scroll(
            collection_name="curriculum_docs",
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            doc_id = p.payload.get("doc_id", "unknown")
            if doc_id not in seen:
                seen[doc_id] = {
                    "source": p.payload.get("source", ""),
                    "difficulty": p.payload.get("difficulty", ""),
                    "doc_type": p.payload.get("doc_type", ""),
                    "chunks": 0,
                }
            seen[doc_id]["chunks"] += 1

        if next_offset is None:
            break
        offset = next_offset

    if not seen:
        print("  Nothing ingested yet.\n")
        return

    print(f"  {'DOC_ID':38s}  {'SOURCE':30s}  {'DIFFICULTY':12s}  {'TYPE':10s}  CHUNKS")
    print("  " + "-" * 103)
    total_chunks = 0
    for doc_id, info in seen.items():
        print(
            f"  {doc_id:38s}  {info['source'][:30]:30s}  "
            f"{info['difficulty']:12s}  {info['doc_type']:10s}  {info['chunks']}"
        )
        total_chunks += info["chunks"]
    print(f"\n  Total: {len(seen)} documents · {total_chunks} chunks\n")


async def cmd_delete(client, doc_id: str) -> None:
    from app.rag.ingestion import delete_document

    print(f"Deleting doc_id={doc_id} ...")
    await delete_document(client, doc_id)
    print("✓ Deleted")


# ── Core ingestion ────────────────────────────────────────────────────────────


async def ingest_one(
    client,
    bm25,
    filepath: Path,
    *,
    topic_id: str,
    difficulty: str,
    grade_level: int,
    doc_type: str,
    language: str,
) -> int:
    from app.rag.ingestion import ingest_document

    text = extract_text(filepath)
    if not text.strip():
        print(f"  SKIP (empty): {filepath.name}")
        return 0

    doc_id = f"{filepath.stem}_{uuid.uuid4().hex[:8]}"
    chunks = await ingest_document(
        client,
        bm25,
        text=text,
        doc_id=doc_id,
        source=filepath.name,
        topic_id=topic_id,
        difficulty=difficulty,
        language=language,
        doc_type=doc_type,
        grade_level=grade_level,
    )
    print(f"  ✓  {filepath.name:45s}  doc_id={doc_id}  chunks={chunks}")
    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────


async def main(args: argparse.Namespace) -> None:
    from app.rag.bm25 import BM25Encoder
    from app.rag.collections import get_qdrant_client, init_collections

    print("\nConnecting to Qdrant...")
    client = await get_qdrant_client()
    await init_collections(client)
    print("✓ Connected\n")

    # ── Utility commands (no ingestion needed) ────────────────────────────────
    if args.list:
        await cmd_list(client)
        await client.close()
        return

    if args.list_topics:
        await print_topics()
        await client.close()
        return

    if args.delete:
        await cmd_delete(client, args.delete)
        await client.close()
        return

    # ── Validate topic slug ───────────────────────────────────────────────────
    if not args.topic_slug:
        print("ERROR: --topic-slug is required for ingestion.")
        print("Run:   python scripts/ingest.py --list-topics")
        sys.exit(1)

    topic_id = await resolve_topic_id(args.topic_slug)
    print(f"✓ Topic '{args.topic_slug}' → {topic_id}")

    # ── Collect files to ingest ───────────────────────────────────────────────
    files: list[Path] = []

    if args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"ERROR: File not found: {args.file}")
            sys.exit(1)
        files.append(p)

    elif args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"ERROR: Folder not found: {args.folder}")
            sys.exit(1)
        files = [
            f
            for f in sorted(folder.rglob("*"))
            if f.is_file() and f.suffix.lower() in (".txt", ".pdf", ".md")
        ]
        if not files:
            print(f"ERROR: No .txt / .pdf / .md files found in {args.folder}")
            sys.exit(1)
        print(f"Found {len(files)} file(s) in {args.folder}")

    else:
        print("ERROR: Provide --file <path>  or  --folder <path>")
        sys.exit(1)

    # ── Build BM25 vocabulary from full corpus before encoding ────────────────
    # BM25 needs to see all documents to compute IDF scores correctly.
    # We sample the first 2000 chars of each file for vocabulary building.
    print("\nBuilding BM25 vocabulary from corpus...")
    corpus_samples: list[str] = []
    for f in files:
        t = extract_text(f)
        if t.strip():
            corpus_samples.append(t[:2000])

    if not corpus_samples:
        print("ERROR: No readable text found in any of the files.")
        sys.exit(1)

    bm25 = BM25Encoder()
    bm25.fit(corpus_samples)
    print(f"✓ BM25 ready — vocab_size={bm25.vocab_size()} from {len(corpus_samples)} doc(s)\n")

    # ── Ingest each file ──────────────────────────────────────────────────────
    print(f"Ingesting {len(files)} file(s) into Qdrant...\n")
    total_chunks = 0
    failed = 0

    for filepath in files:
        try:
            n = await ingest_one(
                client,
                bm25,
                filepath,
                topic_id=topic_id,
                difficulty=args.difficulty,
                grade_level=args.grade_level,
                doc_type=args.doc_type,
                language=args.language,
            )
            total_chunks += n
        except Exception as exc:
            import traceback

            print(f"  ✗ FAILED {filepath.name}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*55}")
    print("  Done.")
    print(f"  Files ingested : {len(files) - failed} / {len(files)}")
    print(f"  Chunks stored  : {total_chunks}")
    if failed:
        print(f"  Failed         : {failed}")
    print(f"{'='*55}\n")

    await client.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest curriculum documents into EduMentor RAG (Qdrant)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest.py --list-topics
  python scripts/ingest.py --file data/algebra.pdf --topic-slug algebra
  python scripts/ingest.py --folder data/physics/ --topic-slug physics --difficulty advanced
  python scripts/ingest.py --list
  python scripts/ingest.py --delete algebra_a1b2c3d4
        """,
    )

    # Ingestion arguments
    parser.add_argument(
        "--file", type=str, metavar="PATH", help="Single file to ingest (.txt / .pdf / .md)"
    )
    parser.add_argument(
        "--folder",
        type=str,
        metavar="PATH",
        help="Folder — ingests every .txt/.pdf/.md file inside it",
    )
    parser.add_argument(
        "--topic-slug",
        type=str,
        metavar="SLUG",
        help="Topic slug from DB  (e.g. algebra, calculus, mechanics)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="intermediate",
        choices=["beginner", "intermediate", "advanced"],
        help="Difficulty level  (default: intermediate)",
    )
    parser.add_argument(
        "--grade-level", type=int, default=10, metavar="N", help="Grade level 1-12  (default: 10)"
    )
    parser.add_argument(
        "--doc-type",
        type=str,
        default="textbook",
        choices=["textbook", "lecture", "worksheet", "reference"],
        help="Document type  (default: textbook)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        metavar="CODE",
        help="ISO 639-1 language code  (default: en)",
    )

    # Utility arguments
    parser.add_argument("--list", action="store_true", help="List all documents already in Qdrant")
    parser.add_argument(
        "--list-topics", action="store_true", help="List all topic slugs available in the database"
    )
    parser.add_argument(
        "--delete", type=str, metavar="DOC_ID", help="Delete a document from Qdrant by its doc_id"
    )

    asyncio.run(main(parser.parse_args()))
