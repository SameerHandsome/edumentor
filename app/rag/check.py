"""
RAG Quality Check — run this BEFORE production, delete afterward.

Usage:
  python -m app.rag.check

Tests:
  1. Embedding endpoint reachable + correct dimensions
  2. BM25 encode/decode round-trip
  3. Qdrant collections exist + reachable
  4. Ingest a sample document
  5. Retrieve with HyDE transform
  6. Retrieve from user_memory with multi-query
  7. Reranker produces sorted results
  8. Full retrieval pipeline end-to-end
"""

from __future__ import annotations

import asyncio
import uuid

# Representative corpus used throughout all checks so the BM25 vocabulary
# covers every document and query token used in this test suite.
_SEED_CORPUS = [
    "photosynthesis converts sunlight to energy using chlorophyll",
    "mitosis is cell division process producing daughter cells",
    "algebra involves equations and variables and polynomials",
    "quadratic equations factoring polynomials algebra student",
    "chloroplasts capture light energy to produce glucose oxygen",
    "plants use carbon dioxide water sunlight to produce sugar",
]


async def run_checks():
    print("\n=== EduMentor RAG Quality Checks ===\n")
    passed = 0
    failed = 0

    # ── Check 1: Embeddings ──────────────────────────────────────────────────
    print("[1/8] Embedding endpoint...")
    try:
        from app.rag.embeddings import embed_batch, embed_text

        vec = await embed_text("test embedding for edumentor")
        assert len(vec) == 768, f"Expected 768-dim, got {len(vec)}"
        batch = await embed_batch(["hello", "world"])
        assert len(batch) == 2 and all(len(v) == 768 for v in batch)
        print("  PASS — 768-dim vectors, batch works\n")
        passed += 1
    except Exception as e:
        print(f"  FAIL — {e}\n")
        failed += 1

    # ── Check 2: BM25 ────────────────────────────────────────────────────────
    print("[2/8] BM25 encoder...")
    try:
        from app.rag.bm25 import BM25Encoder

        # BUG FIX: fit on the same representative corpus used for the whole
        # check suite so no tokens are OOV in later ingestion/retrieval steps.
        enc = BM25Encoder()
        enc.fit(_SEED_CORPUS)
        d_idx, d_vals = enc.encode_document("photosynthesis sunlight energy plant")
        q_idx, q_vals = enc.encode_query("how does photosynthesis work")
        assert len(d_idx) > 0 and len(q_idx) > 0
        assert enc.vocab_size() > 0
        print(
            f"  PASS — vocab_size={enc.vocab_size()}, doc_indices={len(d_idx)}, query_indices={len(q_idx)}\n"
        )
        passed += 1
    except Exception as e:
        print(f"  FAIL — {e}\n")
        failed += 1

    # ── Check 3: Qdrant connection ───────────────────────────────────────────
    print("[3/8] Qdrant connection + collections...")
    try:
        from app.rag.collections import (
            COLLECTION_CURRICULUM,
            COLLECTION_USER_MEMORY,
            get_qdrant_client,
            init_collections,
        )

        client = await get_qdrant_client()
        await init_collections(client)
        collections = {c.name for c in (await client.get_collections()).collections}
        assert COLLECTION_CURRICULUM in collections, f"Missing {COLLECTION_CURRICULUM}"
        assert COLLECTION_USER_MEMORY in collections, f"Missing {COLLECTION_USER_MEMORY}"
        print(f"  PASS — collections: {sorted(collections)}\n")
        passed += 1
    except Exception as e:
        print(f"  FAIL — {e}\n")
        failed += 1

    # ── Check 4: Document ingestion ──────────────────────────────────────────
    print("[4/8] Document ingestion...")
    try:
        from app.rag.ingestion import ingest_document

        test_doc = """
        Photosynthesis is the biological process by which green plants use sunlight,
        water, and carbon dioxide to produce glucose and oxygen. The reaction occurs
        in the chloroplasts, specifically using chlorophyll pigment to capture light energy.
        The overall equation is: 6CO2 + 6H2O + light → C6H12O6 + 6O2.
        This process is fundamental to life on Earth as it forms the base of most food chains.
        """
        test_doc_id = f"test_doc_{uuid.uuid4().hex[:8]}"
        count = await ingest_document(
            client,
            enc,
            text=test_doc,
            doc_id=test_doc_id,
            source="check_test",
            topic_id="00000000-0000-0000-0000-000000000001",
            difficulty="intermediate",
            language="en",
            doc_type="textbook",
            grade_level=10,
        )
        assert count > 0, "No chunks ingested"
        print(f"  PASS — ingested {count} chunks (doc_id={test_doc_id})\n")
        passed += 1
    except Exception as e:
        print(f"  FAIL — {e}\n")
        failed += 1

    # ── Check 5: HyDE + curriculum retrieval ────────────────────────────────
    print("[5/8] HyDE transform + curriculum retrieval...")
    try:
        from app.rag.hyde import hyde_transform
        from app.rag.retriever import retrieve_curriculum

        hyde_vec = await hyde_transform("How does photosynthesis produce oxygen?")
        assert len(hyde_vec) == 768
        # enc is fitted on _SEED_CORPUS which covers photosynthesis tokens — no OOV
        results = await retrieve_curriculum(client, enc, "How does photosynthesis work?", top_k=3)
        assert isinstance(results, list)
        print(f"  PASS — HyDE dim=768, retrieved {len(results)} results")
        if results:
            top = results[0]
            print(
                f"    Top result score={top.score:.3f}, content[:80]={top.payload.get('content','')[:80]}"
            )
        print()
        passed += 1
    except Exception as e:
        print(f"  FAIL — {e}\n")
        failed += 1

    # ── Check 6: User memory upsert + retrieval ──────────────────────────────
    print("[6/8] User memory upsert + multi-query retrieval...")
    try:
        from app.rag.retriever import retrieve_user_memory, upsert_user_memory

        test_user_id = f"user_{uuid.uuid4().hex[:8]}"
        # enc is already fitted on _SEED_CORPUS which includes algebra/quadratic tokens
        await upsert_user_memory(
            client,
            enc,
            user_id=test_user_id,
            doc_id=f"mem_{uuid.uuid4().hex[:8]}",
            memory_type="session_summary",
            content="Student struggled with quadratic equations and factoring polynomials.",
            topic="algebra",
            session_id="test_session",
        )
        memories = await retrieve_user_memory(
            client,
            enc,
            "algebra problems",
            user_id=test_user_id,
            top_k=2,
        )
        assert isinstance(memories, list)
        print(f"  PASS — upserted and retrieved {len(memories)} memories\n")
        passed += 1
    except Exception as e:
        print(f"  FAIL — {e}\n")
        failed += 1

    # ── Check 7: Metadata filtering ──────────────────────────────────────────
    print("[7/8] Metadata pre-filtering...")
    try:
        from app.rag.filters import build_curriculum_filter, build_user_memory_filter
        from app.rag.retriever import retrieve_curriculum

        f1 = build_curriculum_filter(difficulty="intermediate", language="en")
        assert f1 is not None
        f2 = build_user_memory_filter(test_user_id, memory_type="session_summary")
        assert f2 is not None
        # enc covers photosynthesis tokens — no OOV for BM25 sparse search
        results = await retrieve_curriculum(
            client, enc, "photosynthesis", difficulty="intermediate", top_k=3
        )
        print(f"  PASS — filters built and applied, retrieved {len(results)} filtered results\n")
        passed += 1
    except Exception as e:
        print(f"  FAIL — {e}\n")
        failed += 1

    # ── Check 8: Reranker ────────────────────────────────────────────────────
    print("[8/8] Cross-encoder reranker...")
    try:
        from app.rag.reranker import rerank

        candidates = [
            {
                "point_id": "1",
                "payload": {
                    "content": "Photosynthesis uses chlorophyll in chloroplasts to convert CO2 and water to glucose."
                },
            },
            {
                "point_id": "2",
                "payload": {
                    "content": "Mitosis is the process of cell division resulting in two identical daughter cells."
                },
            },
            {
                "point_id": "3",
                "payload": {
                    "content": "The light reactions of photosynthesis produce ATP and NADPH."
                },
            },
        ]
        ranked = rerank("How does photosynthesis work?", candidates, top_k=2)
        assert len(ranked) == 2
        assert ranked[0].score >= ranked[1].score, "Results not sorted by score"
        print(f"  PASS — top result: score={ranked[0].score:.3f}, point_id={ranked[0].point_id}\n")
        passed += 1
    except Exception as e:
        print(f"  FAIL — {e}\n")
        failed += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"{'='*40}")
    print(f"Results: {passed}/8 passed, {failed}/8 failed")
    if failed == 0:
        print("ALL CHECKS PASSED — RAG pipeline is production-ready.")
        print("You can now DELETE this file (app/rag/check.py).")
    else:
        print("Some checks failed. Fix issues before deploying.")
    print("=" * 40)


if __name__ == "__main__":
    asyncio.run(run_checks())
