# server/agent/concept_graph.py
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from functools import lru_cache

def _dedupe_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

@lru_cache(maxsize=16)
def load_pack(pack_dir: str) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      adj: {edge_type: {src_id: [dst_id,...]}}
      nodes: {node_id: node_dict}
    """
    p = Path(pack_dir)
    adj_path = p / "graph_adj.json"
    nodes_path = p / "nodes.jsonl"
    if not adj_path.exists():
        raise FileNotFoundError(f"Missing {adj_path}")
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing {nodes_path}")

    adj = json.loads(adj_path.read_text(encoding="utf-8"))
    nodes: Dict[str, Dict[str, Any]] = {}
    with nodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n = json.loads(line)
                nodes[n["id"]] = n
    return adj, nodes

def rag_topk_nodes(rag_manager, query: str, pack_name: str, topk: int) -> List[str]:
    """
    Query FAISS via rag_manager and return a list of node_ids (filtered to this pack).
    Assumes the 'source' field is a JSON string with {"pack":..., "node_id":...}
    """
    hits = rag_manager.retrieve_knowledge(query, top_k=topk * 3)  # overfetch then filter
    node_ids: List[str] = []
    for h in hits:
        src = h.get("source") if isinstance(h, dict) else ""
        try:
            meta = json.loads(src) if isinstance(src, str) else {}
        except Exception:
            meta = {}
        if meta.get("pack") == pack_name and meta.get("node_id"):
            node_ids.append(meta["node_id"])
    return _dedupe_keep_order(node_ids)[:topk]

def expand_nodes(node_ids: List[str],
                 adj: Dict[str, Dict[str, List[str]]],
                 edge_types: Tuple[str, ...] = ("depends_on",),
                 radius: int = 1) -> List[str]:
    """
    Expand by following specified edge types up to 'radius' hops.
    """
    frontier = list(node_ids)
    out = list(node_ids)
    for _ in range(radius):
        nxt: List[str] = []
        for nid in frontier:
            for et in edge_types:
                for dst in adj.get(et, {}).get(nid, []):
                    nxt.append(dst)
        nxt = _dedupe_keep_order(nxt)
        # stop if no growth
        new = [x for x in nxt if x not in out]
        if not new:
            break
        out.extend(new)
        frontier = new
    return _dedupe_keep_order(out)

def format_context(nodes_by_id: Dict[str, Dict[str, Any]],
                   node_ids: List[str],
                   max_nodes: int = 8,
                   max_examples_per_node: int = 1) -> str:
    """
    Produce a compact, LLM-friendly knowledge bundle:
    - Name (Type)
    - Formal rule (when present)
    - 1–2 sentence summary
    - 0–1 example I/O
    """
    chunks: List[str] = []
    for nid in node_ids[:max_nodes]:
        n = nodes_by_id.get(nid)
        if not n:
            continue
        header = f"{n.get('name','').strip()} ({n.get('type','concept')})"
        formal = n.get("formal")
        summary = (n.get("summary") or "").strip()
        part = [f"### {header}"]
        if formal:
            part.append(f"Formal: {formal}")
        if summary:
            part.append(summary)
        exs = [e for e in (n.get("examples") or []) if isinstance(e, dict)]
        if exs:
            inp = (exs[0].get("input") or "").strip()
            out = (exs[0].get("output") or "").strip()
            if inp or out:
                part.append(f"Example: {inp} -> {out}".strip())
        chunks.append("\n".join(part))
    return "\n\n".join(chunks)
