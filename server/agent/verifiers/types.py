from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Any, Dict

class Verdict(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNKNOWN = "unknown"

@dataclass
class VerifyRequest:
    domain: str            # e.g., "calculus", "sql", "units"
    problem: str           # the original user question (plain text)
    candidate: str         # model's answer (ideally starts with "Final Answer:")
    context: Optional[Dict[str, Any]] = None  # any extra structured hints

@dataclass
class VerifyResult:
    verdict: Verdict
    score: float                    # 0..1 confidence
    expected: Optional[str]         # canonical/correct answer (display string)
    normalized_candidate: Optional[str]  # parsed/normalized candidate form
    explanation: str
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d
