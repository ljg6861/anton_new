"""
Code Importer: Directory → blocks.jsonl

Usage:
  python learning/code_importer.py --root /path/to/repo --out packs/code.v1/blocks.jsonl --domain code

What it does
- Walks a code directory
- Reuses CodeIndexer filters and chunking to produce logical chunks
- Emits blocks.jsonl compatible with concept_extractor Step 2

Output schema (per line):
{
  id: str,
  domain: str,
  section_path: List[str],      # we use [rel_path] (file relative path) as the section
  page: int,                    # chunk index within the file, starting at 1
  block_index: int,             # always 1 per chunk (one block per page)
  block_type: "code",
  text: str,                    # the code chunk
  source: { file: rel_path, section: chunk_source },
  tags: [domain]
}
"""
import argparse
import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List
import fnmatch


@dataclass
class BlockRecord:
    id: str
    domain: str
    section_path: List[str]
    page: int
    block_index: int
    block_type: str
    text: str
    source: Dict[str, Any]
    tags: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _should_index_file(file_path: str, root: str) -> bool:
    """Lightweight filter logic adapted from CodeIndexer._should_index_file."""
    # Directories to exclude
    exclude_dirs = {
        '__pycache__', 'venv', 'env', '.venv', '.env', '.pytest_cache',
        'node_modules', 'dist', 'build', '.next',
        '.git', '.svn', '.hg',
        'data', 'datasets', 'chroma_db', 'packs',
        '.cache', '.chainlit',
        '.DS_Store', 'Thumbs.db'
    }
    # Specific extensions to exclude
    exclude_extensions = {
        '.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', '.bin', '.dat',
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp', '.svg', '.pdf',
        '.zip', '.tar', '.gz', '.7z', '.rar', '.jar', '.war',
        '.mp3', '.mp4', '.avi', '.mov', '.flv', '.wav',
        '.db', '.sqlite', '.mdb', '.ldb', '.npy', '.pkl', '.index',
        '.log', '.cache', '.tmp',
        '.idea', '.vscode', '.vs'
    }
    # Specific files to exclude by pattern
    exclude_file_patterns = [
        '.*', '*.lock', '*.min.*', 'package-lock.json', 'yarn.lock', 'poetry.lock', 'Pipfile.lock',
        'requirements*.txt', '.env*', '.flake8', '.gitignore', '.prettierrc', '.eslintrc', 'Dockerfile', 'LICENSE',
        '*.md5', '*.sum'
    ]
    # Only include these extensions
    include_extensions = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.scss',
        '.json', '.yaml', '.yml', '.toml', '.md', '.rst', '.txt'
    }
    max_file_size_kb = 500

    # Skip excluded directories
    parts = Path(file_path).parts
    for part in parts:
        if part in exclude_dirs or any(fnmatch.fnmatch(part, pattern) for pattern in ['.*']):
            return False

    file_name = os.path.basename(file_path)
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if any(fnmatch.fnmatch(file_name, pattern) for pattern in exclude_file_patterns):
        return False
    if ext in exclude_extensions:
        return False
    if include_extensions and ext not in include_extensions:
        return False
    try:
        if os.path.getsize(file_path) > max_file_size_kb * 1024:
            return False
    except OSError:
        return False

    # Simple binary detection
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = f.read(1024)
            if '\0' in sample:
                return False
    except UnicodeDecodeError:
        return False
    except Exception:
        return False

    return True


def _chunk_code_file(file_path: str, content: str, root: str) -> List[Dict[str, str]]:
    """Chunk code similar to CodeIndexer._chunk_code_file."""
    chunks: List[Dict[str, str]] = []
    _, ext = os.path.splitext(file_path)
    rel_path = os.path.relpath(file_path, root)

    if ext.lower() == '.py':
        import re
        pattern = r'(class\s+\w+\(.*?\)|def\s+\w+\(.*?\))'
        matches = list(re.finditer(pattern, content))
        if not matches:
            return [{"text": content, "source": f"{rel_path}:FULL"}]
        for i, match in enumerate(matches):
            start_pos = match.start()
            end_pos = matches[i+1].start() if i < len(matches) - 1 else len(content)
            chunk_content = content[start_pos:end_pos]
            definition_line = match.group(0)
            chunks.append({"text": chunk_content, "source": f"{rel_path}:{definition_line.strip()}"})
        if matches and matches[0].start() > 0:
            top_content = content[:matches[0].start()]
            chunks.append({"text": top_content, "source": f"{rel_path}:IMPORTS"})
        return chunks

    # For others: line-based
    lines = content.split('\n')
    chunk_size = 100
    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i:i + chunk_size]
        chunk_content = '\n'.join(chunk_lines)
        chunks.append({"text": chunk_content, "source": f"{rel_path}:{i+1}-{i+len(chunk_lines)}"})
    return chunks


def import_directory(root: str, out_path: str, domain: str = "") -> int:
    root = os.path.abspath(root)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    block_id = 0

    with out.open("w", encoding="utf-8") as fw:
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip dot directories quickly
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]

            for fname in filenames:
                file_path = os.path.join(dirpath, fname)
                if not _should_index_file(file_path, root):
                    continue

                rel_path = os.path.relpath(file_path, root)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception:
                    continue

                chunks = _chunk_code_file(file_path, content, root)
                if not chunks:
                    continue
                # Emit one block per chunk as a "page"
                for i, ch in enumerate(chunks, start=1):
                    block_id += 1
                    rec = BlockRecord(
                        id=f"b{block_id:07d}",
                        domain=domain,
                        section_path=[rel_path],
                        page=i,
                        block_index=1,
                        block_type="code",
                        text=ch.get("text", ""),
                        source={"file": rel_path, "section": ch.get("source", rel_path)},
                        tags=[domain] if domain else [],
                    )
                    fw.write(rec.to_json() + "\n")
                    count += 1

    return count


def main():
    ap = argparse.ArgumentParser(description="Import a code directory into blocks.jsonl")
    ap.add_argument("--root", required=True, help="Directory to import")
    ap.add_argument("--out", required=True, help="Output blocks.jsonl path")
    ap.add_argument("--domain", default="", help="Domain tag (e.g., code, calculus)")
    args = ap.parse_args()

    n = import_directory(args.root, args.out, args.domain)
    print(f"[ok] Wrote {args.out} with {n} code blocks")


if __name__ == "__main__":
    main()
