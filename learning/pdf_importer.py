# study_importer.py
# Usage:
#   pip install pymupdf==1.24.7
#   python study_importer.py INPUT.pdf --outdir packs/calc.v1 --domain calculus
#
# Outputs:
#   packs/calc.v1/outline.json
#   packs/calc.v1/blocks.jsonl

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

try:
    import fitz  # PyMuPDF
except Exception as e:
    print("PyMuPDF (fitz) is required. Install with: pip install pymupdf==1.24.7", file=sys.stderr)
    raise

# ---------- Text utilities ----------

WHITESPACE_RE = re.compile(r"[ \t]+")
MULTI_NL_RE = re.compile(r"\n{3,}")
SOFT_HYPHEN_RE = re.compile(r"\u00AD")
HARD_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
LINE_HYPHEN_END_RE = re.compile(r"(\w+)-\n")
JOIN_NL_RE = re.compile(r"(?<![\.!?;:])\n(?=\w)")  # join linebreaks not after end punctuation

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = SOFT_HYPHEN_RE.sub("", s)
    # fix hyphenation across line breaks (basic heuristic)
    s = HARD_HYPHEN_BREAK_RE.sub(r"\1\2", s)
    s = LINE_HYPHEN_END_RE.sub(r"\1", s)
    # collapse spaces
    s = WHITESPACE_RE.sub(" ", s)
    # join harmless single newlines
    s = JOIN_NL_RE.sub(" ", s)
    # collapse huge blank runs
    s = MULTI_NL_RE.sub("\n\n", s)
    return s.strip()

# ---------- Data structures ----------

@dataclass
class SectionNode:
    id: str
    title: str
    level: int
    start_page: int
    end_page: Optional[int] = None
    index: int = 0
    children: List["SectionNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "index": self.index,
            "children": [c.to_dict() for c in self.children],
        }

@dataclass
class BlockRecord:
    id: str
    domain: str
    section_path: List[str]
    page: int
    block_index: int
    block_type: str  # "paragraph" | "example" | "exercise" | "definition" | "theorem" | "remark" | "figure" | "unknown"
    text: str
    source: Dict[str, Any]
    tags: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

# ---------- Heuristics & helpers ----------

HEADING_NUMBER_RE = re.compile(r"^\s*(?:Chapter\s+\d+|Appendix\s+[A-Z]|[\dIVX]+(?:\.\d+){0,3})\b", re.IGNORECASE)
LABEL_CLASS_RE = [
    (re.compile(r"^\s*Example\b", re.IGNORECASE), "example"),
    (re.compile(r"^\s*Exercise\b", re.IGNORECASE), "exercise"),
    (re.compile(r"^\s*Definition\b", re.IGNORECASE), "definition"),
    (re.compile(r"^\s*Theorem\b", re.IGNORECASE), "theorem"),
    (re.compile(r"^\s*Remark\b", re.IGNORECASE), "remark"),
    (re.compile(r"^\s*Figure\b", re.IGNORECASE), "figure"),
]

def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:64] or "untitled"

def detect_headers_footers(doc: "fitz.Document", sample_pages: int = 30) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Identify repeated header/footer lines by sampling pages.
    Returns maps: page_index -> header_text/footer_text to strip if present.
    """
    total = len(doc)
    idxs = list(range(0, min(sample_pages, total)))
    top_counts: Dict[str, int] = {}
    bot_counts: Dict[str, int] = {}

    for i in idxs:
        page = doc[i]
        text_lines = page.get_text("text").splitlines()
        if not text_lines:
            continue
        # Take probable header/footer candidates (first and last non-blank lines)
        header = next((l.strip() for l in text_lines[:5] if l.strip()), "")
        footer = next((l.strip() for l in reversed(text_lines[-5:]) if l.strip()), "")
        if header:
            top_counts[header] = top_counts.get(header, 0) + 1
        if footer:
            bot_counts[footer] = bot_counts.get(footer, 0) + 1

    # Consider lines that appear on >= 40% of sampled pages as header/footer
    thresh = max(2, int(0.4 * len(idxs)))
    headers = {k: k for k, c in top_counts.items() if c >= thresh}
    footers = {k: k for k, c in bot_counts.items() if c >= thresh}
    return headers, footers

def extract_toc_sections(doc: "fitz.Document") -> List[SectionNode]:
    toc = doc.get_toc(simple=False) or doc.get_toc()
    nodes: List[SectionNode] = []
    stack: List[SectionNode] = []

    if not toc:
        return nodes

    for idx, entry in enumerate(toc, start=1):
        if isinstance(entry, dict):
            level = entry.get("level", 1)
            title = entry.get("title", "").strip() or f"Section {idx}"
            page = max(1, entry.get("page", 1))
        else:
            level, title, page = entry[0], entry[1].strip(), max(1, entry[2])

        node = SectionNode(
            id=f"s{idx:04d}-{slugify(title)}",
            title=title,
            level=level,
            start_page=page,
            index=idx,
        )

        # attach by level
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            nodes.append(node)
        stack.append(node)

    # fill end_page by next start
    flat = flatten_sections(nodes)
    for i, n in enumerate(flat):
        n.end_page = (flat[i + 1].start_page - 1) if i + 1 < len(flat) else len(doc)
    return nodes

def flatten_sections(nodes: List[SectionNode]) -> List[SectionNode]:
    out: List[SectionNode] = []
    def _walk(n: SectionNode):
        out.append(n)
        for c in n.children:
            _walk(c)
    for n in nodes:
        _walk(n)
    return out

def infer_sections_by_font(doc: "fitz.Document") -> List[SectionNode]:
    """
    Fallback if no ToC: detect headings by large font spans and numbering patterns.
    """
    candidates: List[Tuple[int, str, int, float]] = []  # (idx, title, page, size)
    idx = 0
    for pno in range(len(doc)):
        page = doc[pno]
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    text = s.get("text", "").strip()
                    if not text:
                        continue
                    size = float(s.get("size", 0.0))
                    if size >= 14.0 or HEADING_NUMBER_RE.search(text):
                        # Keep short-ish heading candidates
                        if len(text) <= 140:
                            idx += 1
                            candidates.append((idx, text, pno + 1, size))

    # keep top-k percentile by size per page, dedupe near-duplicates
    candidates.sort(key=lambda t: (t[2], -t[3]))  # by page, desc size
    filtered: List[Tuple[int, str, int, float]] = []
    seen_on_page: Dict[int, str] = {}
    for i, text, page, size in candidates:
        if page in seen_on_page:
            # keep the largest one per page
            continue
        seen_on_page[page] = text
        filtered.append((i, text, page, size))

    nodes: List[SectionNode] = []
    for i, text, page, size in filtered:
        title = text
        level = 1 if HEADING_NUMBER_RE.search(title) else 2
        nodes.append(SectionNode(
            id=f"s{i:04d}-{slugify(title)}",
            title=title,
            level=level,
            start_page=page,
            index=i,
        ))

    # sort by page then set end pages
    nodes.sort(key=lambda n: (n.start_page, n.level))
    flat = nodes
    for i, n in enumerate(flat):
        n.end_page = (flat[i + 1].start_page - 1) if i + 1 < len(flat) else len(doc)

    # nest simple: level 1 are parents, others children
    roots: List[SectionNode] = []
    last_root: Optional[SectionNode] = None
    for n in flat:
        if n.level <= 1 or last_root is None:
            roots.append(n)
            last_root = n
        else:
            last_root.children.append(n)
    return roots

def classify_block(text: str) -> str:
    head = text.strip().splitlines()[0] if text.strip() else ""
    for pattern, label in LABEL_CLASS_RE:
        if pattern.search(head):
            return label
    return "paragraph"

# ---------- Extraction ----------

def page_text_without_header_footer(page: "fitz.Page", headers: Dict[str, str], footers: Dict[str, str]) -> str:
    raw = page.get_text("text")
    lines = [ln.rstrip() for ln in raw.splitlines()]

    if lines:
        # strip header if exact match (best-effort)
        for h in headers:
            if lines[0].strip() == h:
                lines = lines[1:]
                break
    if lines:
        for f in footers:
            if lines[-1].strip() == f:
                lines = lines[:-1]
                break

    return "\n".join(lines)

def split_into_paragraphs(text: str) -> List[str]:
    # paragraphs separated by blank line or large indentation reset
    paras = re.split(r"\n\s*\n", text.strip())
    cleaned = []
    for p in paras:
        p = normalize_text(p)
        if len(p) >= 30:  # ignore too-short fragments
            cleaned.append(p)
    return cleaned

def build_section_path(section: SectionNode, roots: List[SectionNode]) -> List[str]:
    # reconstruct path by walking down from roots
    path = []
    def _walk(current: SectionNode, target: SectionNode, trail: List[str]) -> Optional[List[str]]:
        t = trail + [current.title]
        if current is target:
            return t
        for c in current.children:
            got = _walk(c, target, t)
            if got:
                return got
        return None
    for r in roots:
        got = _walk(r, section, [])
        if got:
            path = got
            break
    return path or [section.title]

def emit_blocks(doc: "fitz.Document", roots: List[SectionNode], domain: str, out_jsonl: Path, src_file: str):
    headers, footers = detect_headers_footers(doc)
    block_id_counter = 0
    with out_jsonl.open("w", encoding="utf-8") as fw:
        flat = flatten_sections(roots)
        for sec in flat:
            sec_path = build_section_path(sec, roots)
            start = max(1, sec.start_page)
            end = min(len(doc), sec.end_page or len(doc))
            for pno in range(start - 1, end):
                page = doc[pno]
                text = page_text_without_header_footer(page, headers, footers)
                paras = split_into_paragraphs(text)
                for idx, para in enumerate(paras, start=1):
                    block_id_counter += 1
                    btype = classify_block(para)
                    rec = BlockRecord(
                        id=f"b{block_id_counter:07d}",
                        domain=domain,
                        section_path=sec_path,
                        page=pno + 1,
                        block_index=idx,
                        block_type=btype,
                        text=para,
                        source={"file": os.path.basename(src_file), "page": pno + 1, "section_id": sec.id},
                        tags=[domain] if domain else [],
                    )
                    fw.write(rec.to_json() + "\n")

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Study Importer: PDF → outline.json + blocks.jsonl")
    ap.add_argument("pdf", type=str, help="Path to PDF textbook")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory (e.g., packs/calc.v1)")
    ap.add_argument("--domain", type=str, default="", help="Domain tag (e.g., calculus, chess)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(args.pdf)

    # 1) Sections: ToC if available, else heuristics
    roots = extract_toc_sections(doc)
    if not roots:
        print("[info] No ToC found; inferring sections by font/numbering...", file=sys.stderr)
        roots = infer_sections_by_font(doc)

    # 2) Persist outline
    outline_path = outdir / "outline.json"
    with outline_path.open("w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in roots], f, ensure_ascii=False, indent=2)

    # 3) Emit blocks
    blocks_path = outdir / "blocks.jsonl"
    emit_blocks(doc, roots, args.domain or "", blocks_path, args.pdf)

    print(f"[ok] Wrote {outline_path}")
    print(f"[ok] Wrote {blocks_path}")
    print("[next] Feed blocks.jsonl to your concept extractor (Step 2).")

if __name__ == "__main__":
    main()
