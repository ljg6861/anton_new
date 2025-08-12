# server/agent/self_study.py
# Step 6: Self-study / Mastery tracker (domain-agnostic with plugins)
#
# What it does
# - Loads nodes.jsonl (+graph_adj.json) for a pack
# - Generates drills per concept node with the LLM (strict JSON)
# - Solves each drill using your domain bundle (RAG + graph expansion)
# - Verifies via pluggable verifier (e.g., calculus→SymPy)
# - Logs attempts and updates per-node mastery stats
#
# Usage
#   python -m server.agent.self_study \
#     --pack-dir packs/calc.v1 \
#     --api http://127.0.0.1:8001 \
#     --domain calculus \
#     --drills-per-node 2 \
#     --sample-nodes 8 \
#     --temperature 0.2 \
#     --save-good-examples
#
# Outputs (in pack dir)
#   self_study_attempts.jsonl   # one line per attempt
#   mastery.json                # per-node rolling stats
#   examples_aug.jsonl          # harvested good examples (optional)
#
import argparse
import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from server.agent.rag_manager import RAGManager, rag_manager  # existing singleton
from server.agent.concept_graph import load_pack, rag_topk_nodes, expand_nodes, format_context
from server.agent.verifiers.types import VerifyRequest, Verdict
from server.agent.verifiers.base import verify as run_verify
from server.config import QWEN_30B_INSTRUCT, QWEN_30B_THINKING
import server.agent.verifiers
logger = logging.getLogger(__name__)

# ---------------------------- Small utilities ----------------------------

def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    out = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def write_jsonl(p: Path, rows: List[Dict[str, Any]]):
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_json(p: Path, obj: Any):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def slug(s: str) -> str:
    return re.sub(r"[^\w\-]+", "-", s.strip().lower())

# ---------------------------- LLM client ----------------------------

class AgentClient:
    def __init__(self, base: str, path: str = "/react/stream", timeout: float = 180.0):
        self.base = base.rstrip("/")
        self.path = path
        self.timeout = httpx.Timeout(timeout)

    async def ask(self, user_prompt: str) -> str:
        # Sends ONE user message to the agent and streams back plaintext.
        payload = {"messages":[{"role":"user","content":user_prompt}]}
        buf = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", f"{self.base}{self.path}", json=payload) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_text():
                    for line in chunk.split("\n"):
                        if not line.strip(): 
                            continue
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                text = "".join(buf)
                                # keep only post-</think> region if present
                                tail = text.rsplit("</think>", 1)[-1]
                                return tail.strip()
                            buf.append(data)
                        else:
                            buf.append(line)
        text = "".join(buf)
        return text.rsplit("</think>", 1)[-1].strip()


class LLMClient:
    def __init__(self, api_base: str, temperature: float = 0.2, request_timeout: float = 120.0):
        self.api_base = api_base.rstrip("/")
        self.temperature = temperature
        self.request_timeout = request_timeout

    async def chat_once(self, system_prompt: str, user_prompt: str, model: Optional[str] = None) -> str:
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "complex": True,
            "model": model or QWEN_30B_THINKING,
        }
        timeout = httpx.Timeout(self.request_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self.api_base}/v1/chat/stream", json=payload) as response:
                response.raise_for_status()

                full_response_content = ""
                # Iterate over the streamed chunks
                async for chunk in response.aiter_text():
                    for line in chunk.split('\n'):
                        if line.startswith('data: '):
                            content = line[6:]  # Remove 'data: ' prefix
                            if content == '[DONE]':
                                # End of stream marker; continue to finalize below
                                continue
                            full_response_content += content
                        elif line.strip():
                            # Fallback for non-SSE format
                            full_response_content += line
        # Fallback
        text = "".join(full_response_content).strip()
        end = text.rfind("</think>")
        return text[end + len("</think>") :] if end != -1 else text

# ---------------------------- Drill schema ----------------------------

DRILL_GEN_SYSTEM = """You generate focused practice problems (drills) for a given concept.
Return STRICT JSON ONLY per the schema. No prose, no code fences.

Schema:
{{
  "drills": [
    {{
      "task_family": "derivative_at_point|derivative_expr|limit|integral_definite|integral_indefinite|evaluate",
      "problem": "<one clear problem statement in LaTeX-friendly text>",
      "answer_exact": "<exact LaTeX or plain math>",
      "difficulty": "easy|medium|hard",
      "uses_concepts": ["<primary concept name>", "..."]
    }}
  ]
}}

Rules:
- Choose task_family ONLY from the list above (no proofs/theory-only questions).
- derivative_at_point: define f(x)=... and include "at x = a" (small integer a).
- derivative_expr: define f(x)=... and ask for f'(x).
- limit: include "as x -> a" (or x→a) with a concrete a.
- integral_definite: use \\int ... dx with bounds "from a to b".
- integral_indefinite: use \\int ... dx and include + C in the exact answer.
- evaluate: define f(x)=... and ask to evaluate at x = a.
- If the target concept is a theorem/definition and not directly computable, generate drills for a closely-related prerequisite rule that demonstrates its use.
- Return at most {max_per} drills.
"""




DRILL_GEN_USER_TMPL = """Target concept: {name} ({ctype})
Formal: {formal}
Summary: {summary}

Return drills now as JSON matching the schema.
"""

ANSWER_STYLE_CONTRACT = """You must follow this output contract.

Answer Style (MANDATORY):
- Output exactly ONE line starting with: Final Answer:
- Give the simplest exact value in LaTeX (e.g., \\frac{1}{4}, \\sqrt{6}/12). For indefinite integrals: include + C.
- Optionally append a decimal approximation in parentheses with 3 sig figs.
- Do NOT restate the question, do NOT include steps, do NOT repeat the final line.
"""

SOLVE_SYSTEM_TMPL = """You are a precise problem solver. Use the provided domain rules.
Prefer formal rules over prose and obey the Answer Style.

{contract}

When ready, output only the final line as specified.
"""

SOLVE_USER_TMPL = """# Domain rules
{bundle}

# Problem
{problem}
"""

# ---------------------------- Data structures ----------------------------

@dataclass
class Attempt:
    ts: float
    node_id: str
    node_name: str
    problem: str
    candidate: str
    verdict: str
    expected: Optional[str]
    difficulty: str

@dataclass
class Mastery:
    attempts: int = 0
    correct: int = 0
    streak: int = 0
    last_ts: float = 0.0

    def update(self, ok: bool):
        self.attempts += 1
        self.correct += (1 if ok else 0)
        self.streak = self.streak + 1 if ok else 0
        self.last_ts = time.time()

# ---------------------------- Core pipeline ----------------------------

class SelfStudy:
    def __init__(self, pack_dir: str, api_base: str, domain: str, temperature: float, solver, solver_url, agent_path, evaluator: bool = False):
        self.pack_dir = Path(pack_dir)
        self.domain = domain
        self.solver = solver  # pass from args
        self.solver_url = solver_url or api_base  # default to same URL if not given
        self.agent_path = agent_path
        self.agent = AgentClient(self.solver_url, self.agent_path) if self.solver == "agent" else None
        self.client = LLL = LLMClient(api_base=api_base, temperature=temperature)
        self.adj, self.nodes = load_pack(pack_dir)
        self.pack_name = self.pack_dir.name
        # Attach to existing RAG singleton; ensure it's initialized (paths already set in your app)
        self.rag: RAGManager = rag_manager
        # Evaluator mode disables domain bundle (RAG) during solving for baseline comparison
        self.evaluator = evaluator

        # Files
        self.attempts_path = self.pack_dir / "self_study_attempts.jsonl"
        self.mastery_path = self.pack_dir / "mastery.json"
        self.examples_aug = self.pack_dir / "examples_aug.jsonl"

        # Load mastery
        self.mastery: Dict[str, Mastery] = {}
        if self.mastery_path.exists():
            try:
                raw = json.loads(self.mastery_path.read_text(encoding="utf-8"))
                for nid, st in raw.items():
                    self.mastery[nid] = Mastery(**st)
            except Exception:
                logger.warning("Could not load mastery.json; starting fresh.")

    def _format_bundle(self, query_text: str) -> str:
        # Use your existing bundle builder
        node_ids = rag_topk_nodes(self.rag, query_text, self.pack_name, topk=5)
        expanded = expand_nodes(node_ids, self.adj, edge_types=("depends_on",), radius=1)
        ordered = list(dict.fromkeys(node_ids + [x for x in expanded if x not in node_ids]))
        return format_context(self.nodes, ordered, max_nodes=8, max_examples_per_node=1)

    async def _gen_drills_for_node(self, node: Dict[str, Any], max_per: int) -> List[Dict[str, Any]]:
        sys = DRILL_GEN_SYSTEM.format(max_per=max_per)
        usr = DRILL_GEN_USER_TMPL.format(
            name=node.get("name",""),
            ctype=node.get("type","concept"),
            formal=node.get("formal") or "n/a",
            summary=(node.get("summary") or "")[:400],
        )
        # Pass 1: thinking model
        raw = await self.client.chat_once(sys, usr, model=QWEN_30B_THINKING)
        data = self._try_json(raw) or {}
        drills: List[Dict[str, Any]] = []
        if isinstance(data.get("drills"), list):
            for d in data["drills"]:
                prob = (d.get("problem") or "").strip()
                ans  = (d.get("answer_exact") or "").strip()
                diff = (d.get("difficulty") or "medium").lower()
                fam  = (d.get("task_family") or "").strip()
                if prob and ans and fam:
                    drills.append({"problem": prob, "answer_exact": ans, "difficulty": diff, "task_family": fam})

        if drills:
            return drills[:max_per]

        # Pass 2: instruct model to coerce previous output to strict JSON
        fix_msg = (
            "Return ONLY valid JSON (no code fences, no commentary) matching the schema. "
            "Here is your previous output:\n" + (raw or "")
        )
        raw2 = await self.client.chat_once(sys, fix_msg, model=QWEN_30B_INSTRUCT)
        data2 = self._try_json(raw2) or {}
        if isinstance(data2.get("drills"), list):
            for d in data2["drills"]:
                prob = (d.get("problem") or "").strip()
                ans  = (d.get("answer_exact") or "").strip()
                diff = (d.get("difficulty") or "medium").lower()
                fam  = (d.get("task_family") or "").strip()
                if prob and ans and fam:
                    drills.append({"problem": prob, "answer_exact": ans, "difficulty": diff, "task_family": fam})
        if drills:
            return drills[:max_per]

        # ---- Fallback: use a computable prerequisite ----
        nid = node["id"]
        prereqs = self.adj.get("depends_on", {}).get(nid, [])
        for pid in prereqs:
            pnode = self.nodes.get(pid)
            if not pnode:
                continue
            if pnode.get("type","concept") not in ("rule","algorithm","definition","concept"):
                continue
            usr2 = DRILL_GEN_USER_TMPL.format(
                name=pnode.get("name",""),
                ctype=pnode.get("type","concept"),
                formal=pnode.get("formal") or "n/a",
                summary=(pnode.get("summary") or "")[:400],
            )
            raw2 = await self.client.chat_once(sys, usr2, model=QWEN_30B_THINKING)
            data2 = self._try_json(raw2) or {}
            drills2: List[Dict[str, Any]] = []
            if isinstance(data2.get("drills"), list):
                for d in data2["drills"]:
                    prob = (d.get("problem") or "").strip()
                    ans  = (d.get("answer_exact") or "").strip()
                    diff = (d.get("difficulty") or "medium").lower()
                    fam  = (d.get("task_family") or "").strip()
                    if prob and ans and fam:
                        drills2.append({"problem": prob, "answer_exact": ans, "difficulty": diff, "task_family": fam})
            if not drills2 and raw2:
                # Try instruct coercion for prereq as well
                fix2 = (
                    "Return ONLY valid JSON (no code fences, no commentary) matching the schema. "
                    "Here is your previous output:\n" + raw2
                )
                raw2b = await self.client.chat_once(sys, fix2, model=QWEN_30B_INSTRUCT)
                data2b = self._try_json(raw2b) or {}
                if isinstance(data2b.get("drills"), list):
                    for d in data2b["drills"]:
                        prob = (d.get("problem") or "").strip()
                        ans  = (d.get("answer_exact") or "").strip()
                        diff = (d.get("difficulty") or "medium").lower()
                        fam  = (d.get("task_family") or "").strip()
                        if prob and ans and fam:
                            drills2.append({"problem": prob, "answer_exact": ans, "difficulty": diff, "task_family": fam})
            if drills2:
                logger.info("Fallback drills generated via prerequisite: %s", pnode.get("name"))
                return drills2[:max_per]

        logger.info("No drills generated for %s", node.get("name"))
        return []


    async def _solve(self, problem: str) -> str:
        if self.solver == "agent":
            # Let the AGENT fetch context. Just give it the wrapper + problem.
            user = f"""{ANSWER_STYLE_CONTRACT}

            Problem:
            {problem}
            """
            text = await self.agent.ask(user)
        else:
            # Default behavior (respects evaluator flag outside of evaluator compare loop)
            use_bundle = not self.evaluator
            text = await self._solve_variant(problem, use_bundle=use_bundle)
        # Extract single "Final Answer:" line or fall back to whole text
        m = re.search(r"(Final Answer:\s*.+)", text, flags=re.IGNORECASE)
        line = m.group(1).strip() if m else text.strip()
        # Normalize spacing
        logger.info(" ".join(line.split()))
        return " ".join(line.split())

    async def _solve_variant(self, problem: str, use_bundle: bool) -> str:
        bundle = self._format_bundle(problem) if use_bundle else ""
        if not use_bundle:
            logger.info("Evaluator (baseline): solving without domain bundle")
        sys = SOLVE_SYSTEM_TMPL.format(contract=ANSWER_STYLE_CONTRACT)
        usr = SOLVE_USER_TMPL.format(bundle=bundle, problem=problem)
        text = await self.client.chat_once(sys, usr)
        m = re.search(r"(Final Answer:\s*.+)", text, flags=re.IGNORECASE)
        line = m.group(1).strip() if m else text.strip()
        logger.info(" ".join(line.split()))
        return " ".join(line.split())

    def _verify(self, problem: str, candidate: str, expected_exact: Optional[str] = None, task_family: Optional[str] = None):
        clean = re.sub(r'(?is)^\s*final answer:\s*', '', candidate).strip()
        context: Dict[str, Any] = {"task_family": task_family} if task_family else {}
        if expected_exact:
            context["expected_exact"] = expected_exact
        req = VerifyRequest(domain=self.domain, problem=problem, candidate=clean, context=context or None)
        res = run_verify(req)
        if getattr(res, 'verdict', None) and res.verdict.name != 'UNKNOWN':
            return res.verdict, res.expected, res.to_dict()

        # Local lightweight fallback: try numeric equivalence when expected is known
        if expected_exact:
            try:
                from sympy import sympify
                def to_expr(s: str):
                    s = s.strip().strip('$')
                    s = s.replace('\\,', '')
                    # crude latex conversions
                    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
                    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
                    s = s.replace('^', '**')
                    return sympify(s)
                ce = to_expr(clean)
                ee = to_expr(expected_exact)
                if ce.equals(ee):
                    from server.agent.verifiers.types import Verdict as V
                    return V.CORRECT, str(ee), {"verdict": "correct", "fallback": "sympy.equals"}
            except Exception:
                pass
        return res.verdict, res.expected, res.to_dict()

    @staticmethod
    def _try_json(s: str) -> Optional[Dict[str, Any]]:
        s = s.strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(s[i:j+1])
            except Exception:
                return None
        return None

    def _update_mastery(self, nid: str, ok: bool):
        st = self.mastery.get(nid) or Mastery()
        st.update(ok)
        self.mastery[nid] = st

    def _persist_mastery(self):
        save_json(self.mastery_path, {k: asdict(v) for k, v in self.mastery.items()})

    async def study_epoch(
        self,
        drills_per_node: int,
        sample_nodes: int,
        save_good_examples: bool = False,
        seed: int = 0,
    ):
        random.seed(seed)
        # pick target nodes: prioritize lower mastery or unpracticed rules
        candidates = [n for n in self.nodes.values() if n.get("type","concept") in ("rule","theorem","algorithm","concept","definition")]
        random.shuffle(candidates)
        # sort by (low streak, low accuracy)
        def score(n):
            st = self.mastery.get(n["id"])
            if not st or st.attempts == 0:
                return (0.0, 0.0)
            acc = st.correct / max(1, st.attempts)
            return (acc, st.streak)
        candidates.sort(key=score)
        targets = candidates[:sample_nodes]

        attempts_batch: List[Dict[str, Any]] = []
        examples_batch: List[Dict[str, Any]] = []

        # accuracy counters
        accuracy = {"rag": {"total": 0, "correct": 0}, "baseline": {"total": 0, "correct": 0}, "agent": {"total": 0, "correct": 0}}

        for idx, node in enumerate(targets, 1):
            logger.info("Node %d/%d: %s (%s)", idx, len(targets), node.get("name"), node.get("id"))

            drills = await self._gen_drills_for_node(node, drills_per_node)
            if not drills:
                logger.info("No drills generated for %s", node.get("name"))
                continue

            for d in drills:
                prob = d["problem"]
                diff = d["difficulty"]
                family = d.get("task_family")
                expected_exact = d.get("answer_exact")

                # Decide which variants to run
                run_variants = []
                if self.solver == "agent":
                    run_variants = [("agent", None)]
                elif self.evaluator:
                    run_variants = [("baseline", False), ("rag", True)]
                else:
                    run_variants = [("rag", True)]

                for mode, use_bundle in run_variants:
                    try:
                        if mode == "agent":
                            cand = await self._solve(prob)
                        else:
                            cand = await self._solve_variant(prob, use_bundle=use_bundle)  # type: ignore[arg-type]
                    except Exception:
                        logger.exception("Solve failed; skipping drill variant (%s).", mode)
                        continue

                    verdict, expected, meta = self._verify(prob, cand, expected_exact=expected_exact, task_family=family)
                    logger.info("Verdict (%s) for '%s': %s", mode, node.get("name"), verdict.value)
                    if (verdict == Verdict.UNKNOWN):
                        logger.warning("Meta: \n" + str(meta))

                    # Log attempt
                    rec = asdict(Attempt(
                        ts=time.time(),
                        node_id=node["id"],
                        node_name=node.get("name",""),
                        problem=prob,
                        candidate=cand,
                        verdict=verdict.value,
                        expected=expected,
                        difficulty=diff,
                    ))
                    rec["mode"] = mode
                    attempts_batch.append(rec)

                    # Update counters
                    if mode in accuracy:
                        accuracy[mode]["total"] += 1
                        if verdict == Verdict.CORRECT:
                            accuracy[mode]["correct"] += 1

                    # Update mastery only if correct on any variant
                    if verdict == Verdict.CORRECT:
                        self._update_mastery(node["id"], True)

                    # Optionally harvest good examples back into nodes bank
                    if save_good_examples and verdict == Verdict.CORRECT:
                        try:
                            examples_batch.append({
                                "node_id": node["id"],
                                "name": node.get("name",""),
                                "problem": prob,
                                "answer": cand.replace("Final Answer:", "").strip(),
                                "mode": mode,
                            })
                        except Exception:
                            pass

        # Persist artifacts
        if attempts_batch:
            write_jsonl(self.attempts_path, attempts_batch)
            logger.info("Wrote %d attempts → %s", len(attempts_batch), self.attempts_path)
        if examples_batch:
            write_jsonl(self.examples_aug, examples_batch)
            logger.info("Wrote %d harvested examples → %s", len(examples_batch), self.examples_aug)
        self._persist_mastery()
        logger.info("Mastery updated → %s", self.mastery_path)

        # Print a compact accuracy summary
        try:
            def rate(ok, tot):
                return (ok / tot) if tot else 0.0
            rag_ok, rag_tot = accuracy["rag"]["correct"], accuracy["rag"]["total"]
            base_ok, base_tot = accuracy["baseline"]["correct"], accuracy["baseline"]["total"]
            agent_ok, agent_tot = accuracy["agent"]["correct"], accuracy["agent"]["total"]
            logger.info("--- EVALUATION SUMMARY ---")
            logger.info("RAG: %d/%d (%.1f%%)", rag_ok, rag_tot, 100*rate(rag_ok, rag_tot))
            logger.info("Baseline: %d/%d (%.1f%%)", base_ok, base_tot, 100*rate(base_ok, base_tot))
            if agent_tot:
                logger.info("Agent: %d/%d (%.1f%%)", agent_ok, agent_tot, 100*rate(agent_ok, agent_tot))
        except Exception:
            pass

# ---------------------------- CLI ----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Self-study / Mastery runner")
    ap.add_argument("--pack-dir", default="learning/packs/calc.v1", help="Path to pack dir (with nodes.jsonl, graph_adj.json)")
    ap.add_argument("--api", default="http://localhost:8000", help="LLM server base URL (same /v1/chat/stream endpoint)")
    ap.add_argument("--domain", default="calculus", help="Verifier domain key (e.g., calculus)")
    ap.add_argument("--drills-per-node", type=int, default=2)
    ap.add_argument("--sample-nodes", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--save-good-examples", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--solver", choices=["model","agent"], default="model")
    ap.add_argument("--solver-url", required=False, default="http://localhost:8001", help="Base URL for solver (model or agent)")
    ap.add_argument("--agent-path", default="/v1/agent/chat", help="Agent streaming path (override to match your server)")
    ap.add_argument("--evaluator", action="store_true", help="Run baseline without domain embeddings/bundle during solving")
    return ap.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ss = SelfStudy(
        pack_dir=args.pack_dir,
        api_base=args.api,          # still used for drill generation (model)
        domain=args.domain,
        temperature=args.temperature,
        solver=args.solver,
        solver_url=args.solver_url,
    agent_path=args.agent_path,
    evaluator=args.evaluator,
    )
    asyncio.run(ss.study_epoch(
            drills_per_node=args.drills_per_node,
            sample_nodes=args.sample_nodes,
            save_good_examples=args.save_good_examples,
        ))

if __name__ == "__main__":
    main()
