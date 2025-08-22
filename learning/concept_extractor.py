# concept_extractor.py
# Step 2: Concept & Dependency Extractor
#
# What it does
# - Reads blocks.jsonl from Step 1
# - Groups by section and chunks text
# - Calls your local LLM streaming endpoint
# - Produces nodes.jsonl (concepts/rules/etc.) and edges.jsonl (dependencies)
#
# Usage
#   pip install httpx==0.27.0
#   python concept_extractor.py packs/calc.v1/blocks.jsonl \
#     --outdir packs/calc.v1 \
#     --domain calculus \
#     --api http://localhost:8001 \
#     --section-concurrency 2 \
#     --chunk-concurrency 3 \
#     --request-timeout 120 \
#     --retries 2
#
# Notes
# - Uses logger = logging.getLogger(__name__)
# - Concurrency:
#     * --section-concurrency limits how many sections process in parallel
#     * --chunk-concurrency limits parallel LLM calls per section
# - We never keep chain-of-thought; we extract only structured JSON.
import argparse
import asyncio
import json
import logging
import random
import re
import time
import unicodedata
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import httpx
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from server.config import QWEN_30B_INSTRUCT, QWEN_30B_THINKING

logging.basicConfig(
    level=logging.INFO,  # Or logging.DEBUG for more detail
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console
        logging.FileHandler("../logs/concept_extractor.log", mode="w", encoding="utf-8"),  # File
    ]
)
logger = logging.getLogger(__name__)

# ----------------- Schema -----------------

@dataclass
class Node:
    id: str
    type: str          # "concept" | "definition" | "rule" | "algorithm" | "theorem"
    name: str
    summary: str
    formal: Optional[str]
    examples: List[Dict[str, str]] = field(default_factory=list)  # [{"input": "...", "output": "..."}]
    source: Dict[str, Any] = field(default_factory=dict)          # {"pages":[..], "section_path":[..]}
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.0
    synonyms: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class Edge:
    src: str  # node id
    dst: str  # node id
    type: str  # "depends_on" | "analogous_to" | "derived_from" | "applies_to"
    rationale: str
    confidence: float = 0.0
    source: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class NamedEdge:
    """Temporary edge representation that refers to concepts by NAME."""
    src: str
    dst: str
    type: str
    rationale: str
    confidence: float
    source: Dict[str, Any] = field(default_factory=dict)


# ----------------- Utilities -----------------

def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:96] or "unnamed"


def stable_node_id(domain: str, name: str) -> str:
    return f"{domain}.{slugify(name)}"


def norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def chunk_text(paragraphs: List[Dict[str, Any]], max_chars: int) -> List[List[Dict[str, Any]]]:
    chunks, cur, sz = [], [], 0
    for p in paragraphs:
        t = p["text"]
        if sz and (sz + len(t)) > max_chars:
            chunks.append(cur)
            cur, sz = [], 0
        cur.append(p)
        sz += len(t) + 1
    if cur:
        chunks.append(cur)
    return chunks


def coalesce_pages(blocks: List[Dict[str, Any]]) -> List[int]:
    return sorted(set([b["page"] for b in blocks]))


# ----------------- LLM Client -----------------

class LLMClient:
    def __init__(self, api_base: str, temperature: float = 0.6, request_timeout: float = 3600.0, retries: int = 2):
        self.api_base = api_base.rstrip("/")
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.retries = retries

    async def chat_json(self, system_prompt: str, user_prompt: str, model: Optional[str] = None) -> str:
        """
        Calls your streaming endpoint and returns the visible answer (no SSE metadata).
        If the model emits <think>…</think>, we discard it and keep only the content after </think>.
        """

        request_payload = {
            "model": "anton",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],            "temperature": 0.6,
            "stream": False,
            "max_tokens": 4096,
        }

        backoff = 1.0
        vllm_url = "http://localhost:8003"
        url = f"{vllm_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer anton-vllm-key"
        }
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(url, json=request_payload, headers=headers)
                    response.raise_for_status()
                    response_json = response.json()
                    response = response_json['choices'][0]['message']
                    return response['content']


            except Exception as e:
                dur = time.perf_counter() - t0
                logger.warning("LLM call failed in %.2fs (attempt %d/%d): %s", dur, attempt + 1, self.retries + 1, str(e))
                if attempt >= self.retries:
                    logger.exception("LLM request failed (final)")
                    raise
                await asyncio.sleep(backoff + random.random() * 0.25)
                backoff = min(backoff * 2, 8.0)


# ----------------- Prompts -----------------

SYSTEM_PROMPT = """You extract formal knowledge from mixed sources (code files, configs, docs, commit diffs, issues) into a compact graph.
Return STRICT JSON ONLY that matches the schema. Do NOT include notes, prose, explanations, or markdown.

Schema:
{
  "concepts": [
    {
      "type": "concept|definition|rule|algorithm|theorem",
      "name": "<short, canonical title>",
      "summary": "<2-4 sentences, plain text>",
      "formal": "<LaTeX OR plain formula OR precise signature/contract (e.g., `def func(a: int) -> str`, CLI syntax, regex, AST/CST pattern, complexity) OR null>",
      "examples": [{"input":"<minimal before/call/example>", "output":"<after/return/result>"}],
      "synonyms": ["...", "..."],
      "confidence": 0.0_to_1.0
    }
  ],
  "edges": [
    {
      "src": "<name of source concept>",
      "dst": "<name of target concept>",
      "type": "depends_on|analogous_to|derived_from|applies_to",
      "rationale": "<1 sentence why>",
      "confidence": 0.0_to_1.0
    }
  ]
}

Mapping guidance (important):
- Code APIs (functions/methods/classes) → usually "algorithm" (procedures) or "definition" (types/data structures).
- Conventions/policies (lint rules, typing rules, error-handling patterns) → "rule".
- Refactors/codemods (pattern→replacement, API migrations) → "algorithm" (procedure) or "rule" (policy).
- Config keys/schemas → "definition" (what it is) + "rule" for constraints/allowed values.
- Tests/specs → capture as "rule" (contract) and put minimal I/O as examples.

How to fill fields:
- name: concise + canonical (e.g., "Slugify", "LibCST Rename Call", "Retry Policy").
- formal: prefer exact signatures/contracts/patterns (type hints, CLI grammar, regex, AST patterns, Big-O).
- examples:
  - For pure functions: "input" = call/args; "output" = return.
  - For refactors/migrations: "input" = BEFORE snippet; "output" = AFTER snippet.
  - For CLI/config: "input" = command/config; "output" = expected effect/result.
- edges:
  - depends_on: uses/imports/calls/requires/precondition.
  - derived_from: transformation/migration based on or generalization of another rule/API.
  - applies_to: rule/policy applies to a target (e.g., linter rule → code style; test/spec → API).
  - analogous_to: alternative API/pattern with similar purpose.

Noise & scope rules:
- Ignore license banners, changelog boilerplate, autogenerated files, vendor deps, duplicated code.
- Only include high-signal, reusable rules, APIs, contracts, patterns. Skip trivial syntax explanations.
- If nothing meaningful is present, return {"concepts": [], "edges": []}.

Output rules:
- Output VALID JSON ONLY per schema.
- No code fences, no markdown, no commentary.
"""


USER_PROMPT_TEMPLATE = """Extract concepts and dependencies from this repository/documentation slice.
Follow the JSON schema in the system prompt exactly. ONLY output JSON — no extra text.

Domain: {domain}
Section path: {section_path}
Pages/Lines: {pages}

Focus:
- Capture APIs, contracts, invariants, typing/lint/security rules, refactor/codemod patterns, config schemas.
- Prefer minimal I/O pairs for examples (function calls, before→after diffs, command→result).
- Map code/test/config relationships to the allowed edge types (depends_on, derived_from, applies_to, analogous_to).

TEXT:
"""
# The chunk text will be appended after the blank line above.


# ----------------- Extraction orchestration -----------------

def load_blocks(blocks_path: Path) -> List[Dict[str, Any]]:
    out = []
    with blocks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def group_by_section(blocks: List[Dict[str, Any]]) -> Dict[Tuple[str, ...], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for b in blocks:
        key = tuple(b.get("section_path") or [])
        groups.setdefault(key, []).append(b)
    # Preserve order by page then block_index
    for k in groups:
        groups[k].sort(key=lambda x: (x["page"], x.get("block_index", 0)))
    return groups


def build_user_prompt(domain: str, section_path: List[str], pages: List[int], chunk_blocks: List[Dict[str, Any]]) -> str:
    header = USER_PROMPT_TEMPLATE.format(
        domain=domain,
        section_path=" > ".join(section_path),
        pages=f"{min(pages)}–{max(pages)}",
    )
    text = []
    for b in chunk_blocks:
        prefix = f"[{b.get('block_type','paragraph').upper()} p.{b['page']}] "
        text.append(prefix + b["text"])
    return header + "\n".join(text)


def try_json_load(s: str) -> Optional[Dict[str, Any]]:
    s = s.split("</think>")[1] if len(s.split("</think>")) > 1 else s
    s = s.strip()
    try:
        x= json.loads(s)
        logger.info('Successfully loaded initial JSON')
        return x
    except Exception:
        pass
    logger.info('Failed initial load json from: ' + s[:100] + ' ....')
    # Extract the largest plausible JSON object/array substring
    first_brace = s.find("{")
    last_brace = s.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = s[first_brace:last_brace + 1]
        try:
            x= json.loads(candidate)
            logger.info('Successfully loaded secondary JSON')
            return x
        except Exception:
            pass
    first_bracket = s.find("[")
    last_bracket = s.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        candidate = s[first_bracket:last_bracket + 1]
        try:
            x= json.loads(candidate)
            logger.info('Successfully loaded tertiary JSON')
            return x
        except Exception:
            pass
    logger.info('Failed loading tertiary JSON')
    return None


def dedupe_keep_best(nodes: List[Node]) -> List[Node]:
    by_key: Dict[str, Node] = {}
    def key(n: Node) -> str:
        return norm_title(n.name)
    for n in nodes:
        k = key(n)
        if k not in by_key:
            by_key[k] = n
        else:
            cur = by_key[k]
            if n.confidence > cur.confidence:
                cur.summary = n.summary or cur.summary
                cur.formal = n.formal or cur.formal
                cur.type = n.type or cur.type
                cur.confidence = n.confidence
            # merge examples
            seen_ex = set((e.get("input",""), e.get("output","")) for e in cur.examples)
            for e in n.examples:
                tup = (e.get("input",""), e.get("output",""))
                if tup not in seen_ex and any(tup):
                    cur.examples.append(e)
                    seen_ex.add(tup)
            cur.synonyms = sorted(set(cur.synonyms + n.synonyms))
            cur.tags = sorted(set(cur.tags + n.tags))
            # merge pages
            cur.source.setdefault("pages", [])
            for p in n.source.get("pages", []):
                if p not in cur.source["pages"]:
                    cur.source["pages"].append(p)
    return list(by_key.values())


def normalize_edges(named_edges: List[NamedEdge], name_to_id: Dict[str, str]) -> List[Edge]:
    out: List[Edge] = []
    seen = set()
    for e in named_edges:
        src_key = norm_title(e.src)
        dst_key = norm_title(e.dst)
        if src_key not in name_to_id or dst_key not in name_to_id:
            continue
        src_id = name_to_id[src_key]
        dst_id = name_to_id[dst_key]
        key = (src_id, dst_id, e.type)
        if key in seen:
            continue
        seen.add(key)
        out.append(Edge(src=src_id, dst=dst_id, type=e.type, rationale=e.rationale, confidence=e.confidence, source=e.source))
    return out


# ----------------- Main async pipeline -----------------

async def process_section(
    section_key: Tuple[str, ...],
    section_blocks: List[Dict[str, Any]],
    domain: str,
    api: str,
    max_chars: int,
    temperature: float,
    chunk_concurrency: int,
    request_timeout: float,
    retries: int,
    global_llm_sem: asyncio.Semaphore
) -> Tuple[List[Node], List[Edge]]:
    client = LLMClient(api_base=api, temperature=temperature, request_timeout=request_timeout, retries=retries)
    chunks = chunk_text(section_blocks, max_chars=max_chars)
    total = len(chunks)
    all_nodes: List[Node] = []
    all_named_edges: List[NamedEdge] = []

    chunk_sem = asyncio.Semaphore(max(1, min(chunk_concurrency, total)))

    async def process_chunk(i: int, chunk: List[Dict[str, Any]]) -> Tuple[List[Node], List[NamedEdge]]:
        async with chunk_sem:
            pages = coalesce_pages(chunk)
            prompt = build_user_prompt(domain, list(section_key), pages, chunk)
            t0 = time.perf_counter()
            try:
                # Global limiter here:
                async with global_llm_sem:
                    logger.info(
                        "Starting chunk %d/%d for section '%s'", i + 1, total, " > ".join(section_key)
                    )
                    # Pass 1: thinking model
                    raw_think = await client.chat_json(SYSTEM_PROMPT, prompt, model=QWEN_30B_THINKING)
                data = normalize_extractor_json(try_json_load(raw_think))
                if data is None:
                    # Pass 2: instruct model converts the thinking trace into strict JSON
                    if not raw_think or not raw_think.strip():
                        # If the thinking pass produced nothing, try instruct on the original prompt
                        async with global_llm_sem:
                            raw_instr = await client.chat_json(SYSTEM_PROMPT, prompt, model=QWEN_30B_INSTRUCT)
                    else:
                        fix_msg = (
                            "Return ONLY valid JSON (no code fences, no commentary) matching the schema. "
                            "Here is your previous output:\n" + raw_think
                        )
                        async with global_llm_sem:
                            raw_instr = await client.chat_json(SYSTEM_PROMPT, fix_msg, model=QWEN_30B_INSTRUCT)
                    data = normalize_extractor_json(try_json_load(raw_instr))

                # GUARD: if still None, skip this chunk safely
                if data is None:
                    logger.warning(
                        "JSON parse/normalize failed for section %s pages %s. Skipping chunk.",
                        " > ".join(section_key), pages
                    )
                    return [], []

                # Concepts
                concepts = data.get("concepts", []) or []
                nodes: List[Node] = []
                for c in concepts:
                    name = (c.get("name") or "").strip()
                    if not name:
                        continue
                    nodes.append(
                        Node(
                            id="",  # filled after dedupe
                            type=(c.get("type") or "concept").lower(),
                            name=name,
                            summary=(c.get("summary") or "").strip(),
                            formal=c.get("formal", None),
                            examples=[e for e in (c.get("examples") or []) if isinstance(e, dict)],
                            source={"section_path": list(section_key), "pages": pages},
                            tags=[domain],
                            confidence=float(c.get("confidence", 0.0) or 0.0),
                            synonyms=[s for s in (c.get("synonyms") or []) if isinstance(s, str)],
                        )
                    )

                # Edges (keep names for now; map after merging nodes across all chunks)
                edges_in = data.get("edges", []) or []
                named_edges: List[NamedEdge] = []
                for e in edges_in:
                    src = (e.get("src") or "").strip()
                    dst = (e.get("dst") or "").strip()
                    et = (e.get("type") or "").strip().lower()
                    if not (src and dst and et):
                        continue
                    named_edges.append(
                        NamedEdge(
                            src=src,
                            dst=dst,
                            type=et,
                            rationale=(e.get("rationale") or "").strip(),
                            confidence=float(e.get("confidence", 0.0) or 0.0),
                            source={"section_path": list(section_key), "pages": pages},
                        )
                    )

                dur = time.perf_counter() - t0
                logger.info(
                    "Chunk %d/%d for section '%s' done in %.2fs (+%d nodes, +%d edges)",
                    i + 1, total, " > ".join(section_key), dur, len(nodes), len(named_edges),
                )
                return nodes, named_edges
            except Exception:
                logger.exception("Chunk %d/%d failed for section '%s'", i + 1, total, " > ".join(section_key))
                return [], []

    tasks = [asyncio.create_task(process_chunk(i, ch)) for i, ch in enumerate(chunks)]
    for fut in asyncio.as_completed(tasks):
        nodes, n_edges = await fut
        all_nodes.extend(nodes)
        all_named_edges.extend(n_edges)

    # Merge nodes across chunks, assign stable IDs
    all_nodes = dedupe_keep_best(all_nodes)
    for n in all_nodes:
        n.id = stable_node_id(domain, n.name)

    name_to_id = {norm_title(n.name): n.id for n in all_nodes}
    all_edges = normalize_edges(all_named_edges, name_to_id)

    return all_nodes, all_edges


def normalize_extractor_json(obj):
    """
    Coerce model output into {"concepts":[...], "edges":[...]} or return None if impossible.
    Handles cases where the model returns a top-level list, or wraps inside 'data'/'result'.
    """
    if obj is None:
        return None

    # If wrapped in a common container key
    if isinstance(obj, dict):
        for k in ("data", "result", "output"):
            if isinstance(obj.get(k), dict):
                obj = obj[k]
                break

        concepts = obj.get("concepts", [])
        edges = obj.get("edges", [])

        # If concepts given as dict, convert to list
        if isinstance(concepts, dict):
            concepts = list(concepts.values())
        if not isinstance(concepts, list):
            concepts = []

        if isinstance(edges, dict):
            edges = list(edges.values())
        if not isinstance(edges, list):
            edges = []

        return {"concepts": concepts, "edges": edges}

    # If the model returned a bare list, assume it's the concepts list
    if isinstance(obj, list):
        # sanity check: looks like concept dicts?
        if all(isinstance(x, dict) for x in obj):
            return {"concepts": obj, "edges": []}
        return None

    return None



async def main_async(args):
    blocks = load_blocks(Path(args.blocks))
    groups = group_by_section(blocks)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nodes_path = outdir / "nodes.jsonl"
    edges_path = outdir / "edges.jsonl"
    progress_path = outdir / "progress.jsonl"

    t_start = time.perf_counter()

    # Resume support (optional merge)
    existing_nodes: Dict[str, Node] = {}
    existing_edges: set = set()
    completed_sections: set = set()

    if nodes_path.exists():
        with nodes_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    existing_nodes[d["id"]] = Node(**d)
    if edges_path.exists():
        with edges_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    existing_edges.add((d["src"], d["dst"], d["type"]))
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        sp = tuple(rec.get("section_path") or [])
                        if sp:
                            completed_sections.add(sp)
                    except Exception:
                        continue

    # Process sections with bounded concurrency
    section_sem = asyncio.Semaphore(max(1, args.section_concurrency))
    completed = 0
    # Filter out sections already completed
    pending_items = [(k, v) for k, v in groups.items() if k not in completed_sections]
    total_sections = len(pending_items)
    if completed_sections:
        logger.info("Resuming: %d/%d sections already completed; %d pending", len(completed_sections), len(groups), total_sections)

    global_llm_sem = asyncio.Semaphore(max(1, args.max_inflight))
    write_lock = asyncio.Semaphore(1)  # Async-safe single-writer for files

    async def run_one(section_key, section_blocks):
        async with section_sem:
            nodes, edges = await process_section(
                section_key=section_key,
                section_blocks=section_blocks,
                domain=args.domain,
                api=args.api,
                max_chars=args.max_chars,
                temperature=args.temperature,
                chunk_concurrency=args.chunk_concurrency,
                request_timeout=args.request_timeout,
                retries=args.retries,
                global_llm_sem=global_llm_sem,
            )
            return section_key, nodes, edges

    task_list = [asyncio.create_task(run_one(k, v)) for k, v in pending_items]

    all_nodes: Dict[str, Node] = existing_nodes.copy()
    all_edges: set = set(existing_edges)

    for fut in asyncio.as_completed(task_list):
        try:
            k, nodes, edges = await fut
        except Exception:
            logger.exception("Section failed (unknown key)")
            k, nodes, edges = (), [], []
        logger.info("Starting section: %s", " > ".join(k) or "(root)")  # <-- Add here
        # Merge nodes
        for n in nodes:
            if n.id in all_nodes:
                cur = all_nodes[n.id]
                if n.confidence > cur.confidence:
                    cur.summary = n.summary or cur.summary
                    cur.formal = n.formal or cur.formal
                    cur.type = n.type or cur.type
                    cur.confidence = n.confidence
                # lists
                seen_ex = set((e.get("input",""), e.get("output","")) for e in cur.examples)
                for e in n.examples:
                    tup = (e.get("input",""), e.get("output",""))
                    if tup not in seen_ex and any(tup):
                        cur.examples.append(e)
                        seen_ex.add(tup)
                cur.synonyms = sorted(set(cur.synonyms + n.synonyms))
                cur.tags = sorted(set(cur.tags + n.tags))
                # pages
                cur.source.setdefault("pages", [])
                for p in n.source.get("pages", []):
                    if p not in cur.source["pages"]:
                        cur.source["pages"].append(p)
            else:
                all_nodes[n.id] = n

        # Merge edges
        for e in edges:
            key = (e.src, e.dst, e.type)
            if key not in all_edges:
                all_edges.add(key)

        # Incremental saving: append only new or improved items and record progress
        nodes_written = 0
        edges_written = 0
        async with write_lock:
            if nodes:
                with nodes_path.open("a", encoding="utf-8") as fn:
                    for n in nodes:
                        prior = existing_nodes.get(n.id)
                        should_write = False
                        if prior is None:
                            should_write = True
                        elif n.confidence > prior.confidence:
                            # Write improved version as a new line; last-win on load
                            should_write = True
                        if should_write:
                            fn.write(n.to_json() + "\n")
                            existing_nodes[n.id] = n
                            nodes_written += 1
            if edges:
                with edges_path.open("a", encoding="utf-8") as fe:
                    for e in edges:
                        key = (e.src, e.dst, e.type)
                        if key not in existing_edges:
                            fe.write(e.to_json() + "\n")
                            existing_edges.add(key)
                            edges_written += 1
            # Mark section as completed
            with progress_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps({
                    "section_path": list(k),
                    "nodes_written": nodes_written,
                    "edges_written": edges_written,
                    "timestamp": time.time(),
                }) + "\n")
            completed_sections.add(k)

        completed += 1
        logger.info("Progress: %d/%d sections complete", completed, total_sections)
        logger.info("Section processed: %s (+%d nodes, +%d edges; wrote %d new nodes, %d new edges)", " > ".join(k) or "(root)", len(nodes), len(edges), nodes_written, edges_written)

    # Write outputs (deduplicated final snapshot)
    with nodes_path.open("w", encoding="utf-8") as fn:
        for n in sorted(all_nodes.values(), key=lambda x: x.id):
            fn.write(n.to_json() + "\n")

    with edges_path.open("w", encoding="utf-8") as fe:
        for (src, dst, et) in sorted(all_edges):
            e = Edge(src=src, dst=dst, type=et, rationale="", confidence=0.0, source={})
            fe.write(e.to_json() + "\n")

    logger.info("Total runtime: %.2fs", time.perf_counter() - t_start)
    logger.info("Wrote %s", nodes_path)
    logger.info("Wrote %s", edges_path)
    logger.info("Next: Step 3 will load nodes/edges and build your Concept Graph / pgvector index.")


# ----------------- CLI -----------------

def parse_args():
    ap = argparse.ArgumentParser(description="Step 2: Extract concepts and dependencies with your LLM.")
    ap.add_argument("blocks", type=str, help="Path to blocks.jsonl from Step 1")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory (same pack dir)")
    ap.add_argument("--domain", type=str, required=True, help="Domain tag (e.g., calculus)")
    ap.add_argument("--api", type=str, default="http://localhost:8001", help="Model server base URL")
    ap.add_argument("--max-chars", type=int, default=13986, help="Max characters per LLM chunk")
    ap.add_argument("--temperature", type=float, default=0.6, help="LLM temperature for extraction")
    ap.add_argument("--section-concurrency", type=int, default=4, help="Parallel sections (beware rate limits)")
    ap.add_argument("--chunk-concurrency", type=int, default=4, help="Parallel chunks per section")
    ap.add_argument("--request-timeout", type=float, default=3600.0, help="Per-call timeout seconds")
    ap.add_argument("--retries", type=int, default=2, help="LLM call retries")
    ap.add_argument("--max-inflight", type=int, default=5,
                help="Global cap on simultaneous LLM calls across all sections/chunks")
    return ap.parse_args()


def main():
    '''
    Example command for calculus: 
    python3 concept_extractor.py packs/calc.v1/blocks.jsonl --outdir packs/calc.v1 --domain calculus --api http://localhost:8000
    '''
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
