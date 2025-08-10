# step3_rag_indexer.py
# Step 3 using your existing RAGManager (FAISS + Sentence Transformers)
#
# What it does:
# - Reads nodes.jsonl + edges.jsonl from a pack dir
# - Builds a compact text "card" per node and indexes it into RAGManager
# - Saves graph adjacency (from edges) and a node_id→vector_id map alongside the pack
#
# Usage:
#   python3 step3_rag_indexer.py \
#     --pack-dir packs/calc.v1 \
#     --model all-MiniLM-L6-v2 \
#     --index-path rag/knowledge.index \
#     --docs-path rag/documents.pkl \
#     --domain calculus \
#     --max-examples 2 \
#     --dry-run false
#
# Optional quick test:
#   python3 step3_rag_indexer.py ... --test-query "Differentiate e^{x^2} sin x" --topk 5
#
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from server.agent.rag_manager import RAGManager
import numpy as np

logger = logging.getLogger(__name__)

def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    items = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def compose_node_card(n: Dict[str, Any], max_examples: int) -> str:
    """Compact text that captures each node for embedding & retrieval."""
    lines = [f"{n.get('name','').strip()} ({n.get('type','concept')})"]
    summary = n.get("summary") or ""
    if summary:
        lines.append(summary.strip())
    formal = n.get("formal")
    if formal:
        lines.append(f"Formal: {formal}")
    # add up to N examples
    exs = [e for e in (n.get("examples") or []) if isinstance(e, dict)]
    for e in exs[:max_examples]:
        inp = (e.get("input") or "").strip()
        out = (e.get("output") or "").strip()
        if inp or out:
            lines.append(f"Example: {inp} -> {out}".strip())
    return "\n".join([s for s in lines if s])

def build_adjacency(edges: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns: { edge_type: { src_node_id: [dst_node_id, ...], ... }, ... }
    """
    adj: Dict[str, Dict[str, List[str]]] = {}
    for e in edges:
        et = (e.get("type") or "").strip().lower() or "depends_on"
        src = e.get("src")
        dst = e.get("dst")
        if not (src and dst):
            continue
        bucket = adj.setdefault(et, {})
        bucket.setdefault(src, []).append(dst)
    # dedupe lists
    for et in adj:
        for s in adj[et]:
            adj[et][s] = sorted(list(set(adj[et][s])))
    return adj

def infer_pack_name(pack_dir: Path) -> str:
    # e.g., packs/calc.v1 => calc.v1
    return pack_dir.name

def index_pack(
    pack_dir: Path,
    model_name: str,
    index_path: str,
    docs_path: str,
    domain_filter: str,
    max_examples: int,
    dry_run: bool,
    test_query: str,
    topk: int,
):
    nodes_path = pack_dir / "nodes.jsonl"
    edges_path = pack_dir / "edges.jsonl"
    if not nodes_path.exists() or not edges_path.exists():
        raise SystemExit(f"Missing {nodes_path} or {edges_path}")

    nodes = read_jsonl(nodes_path)
    edges = read_jsonl(edges_path)

    pack_name = infer_pack_name(pack_dir)

    # Initialize RAGManager with desired files/model (must happen before another import creates the singleton).
    rag = RAGManager(model_name=model_name, index_path=index_path, doc_store_path=docs_path)

    # Build adjacency and save alongside the pack
    adj = build_adjacency(edges)
    (pack_dir / "graph_adj.json").write_text(json.dumps(adj, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved adjacency: %s", pack_dir / "graph_adj.json")

    # Index nodes into RAG
    # We'll keep a local mapping from node_id -> vector_id for fast lookup later.
    node_to_vec: Dict[str, int] = {}

    indexed = 0
    for n in nodes:
        # Optional domain filter (uses node tags)
        tags = [t for t in (n.get("tags") or []) if isinstance(t, str)]
        if domain_filter and (domain_filter not in tags):
            continue

        text_card = compose_node_card(n, max_examples=max_examples)

        # stash useful metadata in source as JSON string
        meta = {
            "pack": pack_name,
            "node_id": n["id"],
            "name": n.get("name"),
            "type": n.get("type"),
            "tags": tags,
            "source": n.get("source"),
        }
        # Calculate the new vector ID: it's the current size before adding
        # (RAGManager.add_knowledge increments ntotal by 1).
        vec_id = rag.index.ntotal if rag.index is not None else 0

        if dry_run:
            logger.info("[dry-run] would index node %s as vec_id=%d", n["id"], vec_id)
        else:
            rag.add_knowledge(text=text_card, source=json.dumps(meta, ensure_ascii=False))
            node_to_vec[n["id"]] = vec_id
            indexed += 1

    if not dry_run:
        rag.save()
        # Persist the mapping so retrieval/expansion can be fast
        map_path = pack_dir / "nodeid_to_vecid.json"
        map_path.write_text(json.dumps(node_to_vec, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Indexed %d nodes; saved node→vector map: %s", indexed, map_path)

    # Optional quick test
    if test_query:
        if dry_run:
            logger.info("[dry-run] Skipping test query (no index writes)")
            return
        res = rag.retrieve_knowledge(test_query, top_k=topk)
        logger.info("TEST QUERY: %r", test_query)
        for i, r in enumerate(res, 1):
            src = r.get("source", "")
            try:
                meta = json.loads(src)
                nid = meta.get("node_id")
                name = meta.get("name")
                ntype = meta.get("type")
                logger.info("  %d) %s [%s]  (node_id=%s)", i, name, ntype, nid)
            except Exception:
                logger.info("  %d) %s", i, src[:120])

def parse_args():
    ap = argparse.ArgumentParser(description="Step 3: Index nodes into FAISS (RAGManager) + save graph adjacencies.")
    ap.add_argument("--pack-dir", required=True, help="Directory with nodes.jsonl and edges.jsonl")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    ap.add_argument("--index-path", default="../../knowledge.index", help="FAISS index path")
    ap.add_argument("--docs-path", default="../../documents.pkl", help="Doc-store path")
    ap.add_argument("--domain", default="", help="Optional tag filter (e.g., 'calculus')")
    ap.add_argument("--max-examples", type=int, default=2, help="Max examples to include per node card")
    ap.add_argument("--dry-run", type=lambda s: s.lower() in {"1","true","yes","y"}, default=False)
    ap.add_argument("--test-query", default="", help="Optional quick RAG search to validate")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    index_pack(
        pack_dir=Path(args.pack_dir),
        model_name=args.model,
        index_path=args.index_path,
        docs_path=args.docs_path,
        domain_filter=args.domain,
        max_examples=args.max_examples,
        dry_run=args.dry_run,
        test_query=args.test_query,
        topk=args.topk,
    )

if __name__ == "__main__":
    main()
