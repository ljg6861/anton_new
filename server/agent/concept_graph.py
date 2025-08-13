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
                   max_nodes: int = 6,  # Reduced from 8
                   max_examples_per_node: int = 1,
                   max_chars_per_node: int = 400) -> str:  # New parameter
    """
    Produce a compact, LLM-friendly knowledge bundle:
    - Name (Type)
    - Formal rule (when present)
    - 1–2 sentence summary
    - 0–1 example I/O or example ID for expansion
    """
    chunks: List[str] = []
    seen_titles = set()  # De-duplicate by normalized title
    
    def norm_title(name: str) -> str:
        """Normalize title for deduplication"""
        return name.lower().strip().replace(' ', '').replace('-', '').replace('_', '')
    
    for nid in node_ids[:max_nodes]:
        n = nodes_by_id.get(nid)
        if not n:
            continue
            
        name = n.get('name', '').strip()
        norm_name = norm_title(name)
        
        # Skip duplicates by normalized title
        if norm_name in seen_titles:
            continue
        seen_titles.add(norm_name)
        
        header = f"{name} ({n.get('type','concept')})"
        formal = n.get("formal")
        summary = (n.get("summary") or "").strip()
        
        part = [f"### {header}"]
        
        if formal:
            # Keep formal concise
            if len(formal) > 150:
                formal = formal[:147] + "..."
            part.append(f"Formal: {formal}")
            
        if summary:
            # Prefer summary field, keep to 2 sentences max
            sentences = summary.split('. ')
            if len(sentences) > 2:
                summary = '. '.join(sentences[:2]) + '.'
            part.append(summary)
        
        # Handle examples - prefer IDs for large examples
        exs = [e for e in (n.get("examples") or []) if isinstance(e, dict)]
        if exs:
            inp = (exs[0].get("input") or "").strip()
            out = (exs[0].get("output") or "").strip()
            
            # If example is long, replace with ID
            example_text = f"{inp} -> {out}".strip()
            if len(example_text) > 100:
                example_id = f"{nid}#ex1"
                part.append(f"Example: ... (see example: {example_id})")
            elif inp or out:
                part.append(f"Example: {example_text}")
        
        node_content = "\n".join(part)
        
        # Enforce max_chars_per_node limit
        if len(node_content) > max_chars_per_node:
            # Truncate content intelligently
            lines = node_content.split('\n')
            truncated_lines = [lines[0]]  # Always keep header
            current_length = len(lines[0])
            
            for line in lines[1:]:
                if current_length + len(line) + 1 <= max_chars_per_node - 20:  # Leave room for "..."
                    truncated_lines.append(line)
                    current_length += len(line) + 1
                else:
                    truncated_lines.append("...")
                    break
            node_content = "\n".join(truncated_lines)
        
        chunks.append(node_content)
    
    return "\n\n".join(chunks)
