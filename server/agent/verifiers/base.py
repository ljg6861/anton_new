import logging
from typing import Callable, Dict, List, Optional
from .types import VerifyRequest, VerifyResult, Verdict

logger = logging.getLogger(__name__)

# Simple plugin registry keyed by domain name (lowercase)
_REGISTRY: Dict[str, List[Callable[[VerifyRequest], Optional[VerifyResult]]]] = {}

def register(domain: str):
    """
    Decorator to register a verifier function for a domain.
    The function should accept VerifyRequest and return VerifyResult or None (if it can't handle the request).
    """
    def deco(fn: Callable[[VerifyRequest], Optional[VerifyResult]]):
        _REGISTRY.setdefault(domain.lower(), []).append(fn)
        return fn
    return deco

def verify(req: VerifyRequest) -> VerifyResult:
    """
    Route to first plugin that can produce a result. If none returns a result, return UNKNOWN.
    """
    domain = (req.domain or "").lower()
    fns = _REGISTRY.get(domain, []) + _REGISTRY.get("*", [])
    for fn in fns:
        try:
            res = fn(req)
            if res is not None:
                return res
        except Exception as e:
            logger.exception("Verifier '%s' crashed; continuing to next plugin.", getattr(fn, "__name__", "unknown"))
    return VerifyResult(
        verdict=Verdict.UNKNOWN,
        score=0.0,
        expected=None,
        normalized_candidate=None,
        explanation=f"No verifier available for domain '{domain}' or request type.",
        meta={"domain": domain},
    )
