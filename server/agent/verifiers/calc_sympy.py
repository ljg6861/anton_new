import re
import json
import logging
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)
_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)

from typing import Optional, Dict, Any, Tuple

from sympy import symbols, diff, simplify, sympify, limit, integrate, Eq
import sympy
from sympy.core.relational import Relational

try:
    # optional; if not present we fall back to simple parsing
    from sympy.parsing.latex import parse_latex
    HAVE_LATEX = True
except Exception:
    HAVE_LATEX = False

from .types import VerifyRequest, VerifyResult, Verdict
from .base import register

logger = logging.getLogger(__name__)
x = symbols('x', real=True)

# ---- helpers ---------------------------------------------------------------

def _strip_prefix(s: str) -> str:
    s = s.strip()
    m = re.search(r"final answer:\s*(.*)$", s, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else s

def _maybe_parse_expr(expr_str: str):
    s = expr_str.strip()
    # strip $...$ and \( ... \)
    s = re.sub(r"^\s*\$\s*|\s*\$\s*$", "", s)
    s = re.sub(r"^\s*\\\(\s*|\s*\\\)\s*$", "", s)

    if HAVE_LATEX:
        try:
            return parse_latex(s)
        except Exception:
            pass

    # common LaTeX -> SymPy
    s = (s
         .replace("\\arcsin", "asin").replace("\\arccos", "acos").replace("\\arctan", "atan")
         .replace("\\sin", "sin").replace("\\cos", "cos").replace("\\tan", "tan")
         .replace("\\ln", "log").replace("\\log", "log")
         .replace("\\pi", "pi")
         .replace("\\cdot", "*").replace("\\times", "*")
         .replace("\\sqrt", "sqrt"))
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("^", "**")
    # kill trailing punctuation that sneaks in from problem text
    s = re.sub(r"[\.\;\,]\s*$", "", s)

    # robust fallback with implicit multiplication (handles "2x+4")
    try:
        return parse_expr(s, transformations=_TRANSFORMS, local_dict={"x": x, "pi": sympy.pi})
    except Exception:
        pass

    try:
        return sympify(s, {"x": x, "pi": sympy.pi})
    except Exception:
        return None




def _num_equal(a, b, tol=1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False

def _extract_fx(problem: str) -> Optional[str]:
    """
    Extract the RHS of f(x)=... but stop before separators like ' at ', ' for ', comma, or period.
    """
    m = re.search(
        r"f\s*\(\s*x\s*\)\s*=\s*([^\n]+?)\s*(?:,|;|\.|\bat\b|\bfor\b|$)",
        problem,
        flags=re.IGNORECASE,
    )
    return m.group(1).strip() if m else None



def _extract_at_point(problem: str) -> Optional[float]:
    m = re.search(r"x\s*=\s*([+-]?\d+(?:\.\d+)?)", problem)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

# ---- verifiers -------------------------------------------------------------

def _derivative_at_point(req: VerifyRequest) -> Optional[VerifyResult]:
    if not re.search(r"(derivative|differentiate|f'\(x\)|d/dx)", req.problem, re.IGNORECASE):
        return None
    a = _extract_at_point(req.problem)
    if a is None:
        return None  # not a point-evaluation task

    fx_txt = _extract_fx(req.problem)
    if not fx_txt:
        return None  # need f(x)=...

    f = _maybe_parse_expr(fx_txt)
    if f is None:
        return None

    truth = diff(f, x).subs(x, a)
    cand_expr_text = _strip_prefix(req.candidate)
    cand_expr = _maybe_parse_expr(cand_expr_text)
    if cand_expr is None:
        # try to parse numeric only
        cand_num = _maybe_parse_expr(cand_expr_text.replace("≈", "").split()[0])
        if cand_num is None:
            return None
        is_ok = _num_equal(truth.evalf(), cand_num.evalf())
        verdict = Verdict.CORRECT if is_ok else Verdict.INCORRECT
        return VerifyResult(
            verdict=verdict,
            score=1.0 if is_ok else 0.0,
            expected=str(simplify(truth)),
            normalized_candidate=str(cand_num),
            explanation="Compared numeric derivative at the evaluation point.",
            meta={"task": "derivative_at_point"},
        )

    # compare symbolically (or numerically if needed)
    try:
        is_ok = simplify(truth - cand_expr) == 0 or _num_equal(truth.evalf(), cand_expr.evalf())
    except Exception:
        is_ok = _num_equal(truth.evalf(), cand_expr.evalf())

    return VerifyResult(
        verdict=Verdict.CORRECT if is_ok else Verdict.INCORRECT,
        score=1.0 if is_ok else 0.0,
        expected=str(simplify(truth)),
        normalized_candidate=str(simplify(cand_expr)),
        explanation="Verified derivative at point via SymPy.",
        meta={"task": "derivative_at_point"},
    )

def _derivative_expr(req: VerifyRequest) -> Optional[VerifyResult]:
    if not re.search(r"(derivative|differentiate|d/dx)", req.problem, re.IGNORECASE):
        return None
    if _extract_at_point(req.problem) is not None:
        return None  # handled by point case

    fx_txt = _extract_fx(req.problem)
    if not fx_txt:
        return None

    f = _maybe_parse_expr(fx_txt)
    if f is None:
        return None

    truth = simplify(diff(f, x))
    cand_expr = _maybe_parse_expr(_strip_prefix(req.candidate))
    if cand_expr is None:
        return None

    same = simplify(truth - simplify(cand_expr)) == 0
    return VerifyResult(
        verdict=Verdict.CORRECT if same else Verdict.INCORRECT,
        score=1.0 if same else 0.0,
        expected=str(truth),
        normalized_candidate=str(simplify(cand_expr)),
        explanation="Compared symbolic derivatives.",
        meta={"task": "derivative_expr"},
    )

def _limit(req: VerifyRequest) -> Optional[VerifyResult]:
    if not re.search(r"(limit|\\lim|lim_{)", req.problem, re.IGNORECASE):
        return None
    # very simple extractor: "... as x -> a" or "x→a"
    m = re.search(r"x\s*(?:->|→)\s*([+-]?\d+(?:\.\d+)?)", req.problem)
    if not m:
        return None
    a = float(m.group(1))
    # try to find expression after 'of' or after 'lim'
    # e.g. "Evaluate the limit of (sin(5x))/x as x->0"
    ex = None
    m2 = re.search(r"of\s*(.+?)\s*as\s*x", req.problem, re.IGNORECASE)
    if m2:
        ex = m2.group(1)
    if ex is None:
        # fallback: ')' after lim, very rough
        m3 = re.search(r"lim[^\)]*\)\s*([^\s]+)", req.problem, re.IGNORECASE)
        ex = m3.group(1) if m3 else None
    if not ex:
        return None
    f = _maybe_parse_expr(ex)
    if f is None:
        return None
    truth = limit(f, x, a)
    cand_expr = _maybe_parse_expr(_strip_prefix(req.candidate))
    if cand_expr is None:
        # try numeric
        cand_expr = _maybe_parse_expr(_strip_prefix(req.candidate).replace("≈", "").split()[0])
    if cand_expr is None:
        return None
    ok = simplify(truth - cand_expr) == 0 or _num_equal(truth.evalf(), cand_expr.evalf())
    return VerifyResult(
        verdict=Verdict.CORRECT if ok else Verdict.INCORRECT,
        score=1.0 if ok else 0.0,
        expected=str(simplify(truth)),
        normalized_candidate=str(simplify(cand_expr)),
        explanation="Verified limit via SymPy.",
        meta={"task": "limit", "point": a},
    )

def _integral_definite(req: VerifyRequest) -> Optional[VerifyResult]:
    # \int_a^b ... dx
    m = re.search(
        r"(?:\\int|∫)\s*[_\{]\s*([^}\s]+)\s*[\}^\s]\s*[\^{]\s*([^}\s]+)\s*[\}]\s*(.+?)\s*d\s*x",
        req.problem, re.IGNORECASE
    )
    if m:
        a_txt, b_txt, integ_txt = m.group(1), m.group(2), m.group(3)
    else:
        # "... \int ... dx from a to b"
        m = re.search(
            r"(?:\\int|∫)\s*(.+?)\s*d\s*x.*?(?:from|between)\s*([+-]?\d+(?:\.\d+)?)\s*(?:to|-)\s*([+-]?\d+(?:\.\d+)?)",
            req.problem, re.IGNORECASE
        )
        if not m:
            return None
        integ_txt, a_txt, b_txt = m.group(1), m.group(2), m.group(3)

    integ_txt = re.sub(r"[\.\;\,]\s*$", "", integ_txt)
    integ = _maybe_parse_expr(integ_txt)
    a = _maybe_parse_expr(a_txt) or sympify(a_txt)
    b = _maybe_parse_expr(b_txt) or sympify(b_txt)
    if integ is None or a is None or b is None:
        return None

    truth = integrate(integ, (x, a, b))
    cand = _maybe_parse_expr(_strip_prefix(req.candidate)) or _maybe_parse_expr(_strip_prefix(req.candidate).replace("≈","").split()[0])
    if cand is None:
        return None
    ok = simplify(truth - cand) == 0 or _num_equal(truth.evalf(), cand.evalf())
    return VerifyResult(
        verdict=Verdict.CORRECT if ok else Verdict.INCORRECT,
        score=1.0 if ok else 0.0,
        expected=str(simplify(truth)),
        normalized_candidate=str(simplify(cand)),
        explanation="Verified definite integral via SymPy.",
        meta={"task": "integral_definite", "bounds": [str(a), str(b)]},
    )



def _integral_indefinite(req: VerifyRequest) -> Optional[VerifyResult]:
    # accept \int ... dx OR "indefinite integral of ..." / "antiderivative of ..."
    m = re.search(r"(?:\\int|∫)\s*(.+?)\s*d\s*x", req.problem, re.IGNORECASE)
    if not m:
        m = re.search(r"(?:indefinite integral of|antiderivative of)\s*(.+)", req.problem, re.IGNORECASE)
    if not m:
        return None

    expr_txt = m.group(1).strip()
    expr_txt = re.sub(r"[\.\;\,]\s*$", "", expr_txt)  # NEW: drop trailing punctuation
    integ = _maybe_parse_expr(expr_txt)
    if integ is None:
        return None

    cand_text = _strip_prefix(req.candidate)
    cand_text = re.sub(r"\+?\s*C\b", "", cand_text).strip()  # drop + C for comparison
    cand = _maybe_parse_expr(cand_text)
    if cand is None:
        return None

    same = simplify(diff(cand, x) - integ) == 0
    return VerifyResult(
        verdict=Verdict.CORRECT if same else Verdict.INCORRECT,
        score=1.0 if same else 0.0,
        expected=str(integrate(integ, x)),
        normalized_candidate=str(simplify(cand)),
        explanation="Verified indefinite integral by differentiating candidate.",
        meta={"task": "integral_indefinite"},
    )



def _evaluate_at_point(req: VerifyRequest):
    fx_txt = _extract_fx(req.problem)
    a = _extract_at_point(req.problem)
    if not fx_txt or a is None:
        return None
    f = _maybe_parse_expr(fx_txt)
    if f is None:
        return None
    truth = f.subs(x, a)
    cand = _maybe_parse_expr(_strip_prefix(req.candidate)) or _maybe_parse_expr(_strip_prefix(req.candidate).replace("≈","").split()[0])
    if cand is None:
        return None
    ok = simplify(truth - cand) == 0 or _num_equal(truth.evalf(), cand.evalf())
    return VerifyResult(
        verdict=Verdict.CORRECT if ok else Verdict.INCORRECT,
        score=1.0 if ok else 0.0,
        expected=str(simplify(truth)),
        normalized_candidate=str(simplify(cand)),
        explanation="Verified value of f(x) at a point.",
        meta={"task": "evaluate_at_point", "point": a},
    )


@register("calculus")
def calculus_router(req: VerifyRequest):
    fam = (req.context or {}).get("task_family")
    if fam == "derivative_at_point":   return _derivative_at_point(req)
    if fam == "derivative_expr":       return _derivative_expr(req)
    if fam == "limit":                 return _limit(req)
    if fam == "integral_definite":     return _integral_definite(req)
    if fam == "integral_indefinite":   return _integral_indefinite(req)
    if fam == "evaluate":              return _evaluate_at_point(req)
    # fallback guess
    for fn in (_derivative_at_point, _derivative_expr, _limit, _integral_definite, _integral_indefinite, _evaluate_at_point):
        r = fn(req)
        if r is not None:
            return r
    return None


