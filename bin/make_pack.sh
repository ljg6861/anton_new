#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bin/make_pack.sh --domain calculus --input ~/docs/calc.pdf --api http://localhost:8000 --pack calc.v1
#   bin/make_pack.sh --domain calculus --input packs/calc.v1/blocks.jsonl --api http://localhost:8000
#
# Notes:
# - If INPUT ends with .pdf, we convert it to blocks.jsonl first (step 1).
# - Then we run concept_extractor (step 2) and index to RAG (step 3).
# - Adjust the PDF→blocks command to your local tool if your path differs.

DOMAIN=""
INPUT=""
API="${API:-http://localhost:8000}"
PACK=""
SECTION_CONCURRENCY="${SECTION_CONCURRENCY:-4}"
CHUNK_CONCURRENCY="${CHUNK_CONCURRENCY:-4}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_CHARS="${MAX_CHARS:-13000}"
RETRIES="${RETRIES:-2}"
TIMEOUT="${TIMEOUT:-120}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2 ;;
    --input)  INPUT="$2"; shift 2 ;;
    --api)    API="$2"; shift 2 ;;
    --pack)   PACK="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$DOMAIN" || -z "$INPUT" ]]; then
  echo "Required: --domain <key> --input <pdf|blocks.jsonl>"
  exit 1
fi

# Project root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT"

# Derive PACK if not provided
if [[ -z "$PACK" ]]; then
  # default: <domain>.<yyyymmdd_HHMM>
  ts="$(date +%Y%m%d_%H%M)"
  PACK="packs/${DOMAIN}.${ts}"
fi
OUTDIR="$PACK"
mkdir -p "$OUTDIR"

# Step 1: PDF -> blocks.jsonl (only if INPUT is a PDF)
BLOCKS="$OUTDIR/blocks.jsonl"
if [[ "$INPUT" == *.pdf ]]; then
  echo "[1/3] Building blocks.jsonl from PDF..."
  # >>> Replace the line below with your actual PDF-to-blocks tool if different <<<
  # Example assumes you have a module or script that does this:
  # python3 -m server.agent.pdf_to_blocks "$INPUT" --out "$BLOCKS"
  # Fallback: if you already have blocks, you can do it elsewhere and just pass the .jsonl path.
  if [[ -x "server/agent/pdf_to_blocks.py" ]]; then
    python3 server/agent/pdf_to_blocks.py "$INPUT" --out "$BLOCKS"
  else
    echo "ERROR: Need a PDF→blocks tool. Put it at server/agent/pdf_to_blocks.py or edit this script."
    exit 1
  fi
else
  # INPUT is already a blocks.jsonl
  echo "[1/3] Using existing blocks.jsonl..."
  cp -f "$INPUT" "$BLOCKS"
fi

# Step 2: Concept extraction -> nodes.jsonl + graph_adj.json
echo "[2/3] Extracting concepts and graph..."
# concept_extractor.py is in the learning/ folder in your repo
python3 learning/concept_extractor.py "$BLOCKS" \
  --outdir "$OUTDIR" \
  --domain "$DOMAIN" \
  --api "$API" \
  --section-concurrency "$SECTION_CONCURRENCY" \
  --chunk-concurrency "$CHUNK_CONCURRENCY" \
  --temperature "$TEMPERATURE" \
  --max-chars "$MAX_CHARS" \
  --retries "$RETRIES" \
  --request-timeout "$TIMEOUT"

# Step 3: Index nodes into RAG
echo "[3/3] Indexing nodes into RAG..."
python3 -m server.agent.index_pack_to_rag --pack-dir "$OUTDIR" --domain "$DOMAIN"

echo "Done."
echo "Pack directory: $OUTDIR"
echo "Artifacts: $OUTDIR/blocks.jsonl, $OUTDIR/nodes.jsonl, $OUTDIR/graph_adj.json"
