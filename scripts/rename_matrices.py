#!/usr/bin/env python3
"""Unify matrix notation in week1.md.

Changes (math-mode only unless noted):
  * system matrix / Hessian   A -> H   in Sections 1,2,3,4,6,7 and the
    deterministic part of Section 9 (lines < statistical-subsection start).
    `\\mathcal{A}` (the abstract algorithm) is protected.
  * data matrix               D -> X   document-wide in math mode.
    `\\mathcal{D}` (the data distribution) is protected.  The only math-mode
    `D` anywhere is the data matrix; prose tokens like "2D" / "SDE" live
    outside math spans and are untouched.
  * data-matrix rows          d_* -> x_*  (and bare `\\lVert d\\rVert` -> x)
    only inside the randomized-Kaczmarz block (Section 8).
  * Sobolev spaces            H^{..} -> W^{..,2}  (resolves the H collision).
  * Section-8 bridging prose: the deterministic-recap matrix and the
    "identification A=H" sentence are reconciled by hand (Section 8 keeps the
    LOCAL contraction A := I - gamma H untouched).
"""
import re
import bisect
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "week1.md"
text = SRC.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. Literal prose/recap fixes that sit in Section 8 (NOT an A->H region).
# ---------------------------------------------------------------------------
LITERAL = [
    # deterministic recap: the system matrix is now H, products v |-> Hv
    ("access the matrix $A$ only through matrix-vector products $v\\mapsto Av$",
     "access the matrix $H$ only through matrix-vector products $v\\mapsto Hv$", 1),
    # bridging sentence: both sides are now literally H
    ("of Section 1 under the identification $A=H$, the excess population risk",
     "of Section 1, the excess population risk", 1),
    # Sobolev characterization -> W^{,2}
    ("$f \\in H^m$", "$f \\in W^{m,2}$", 1),
    ("the Sobolev space $H^{2s}([0,1])$", "the Sobolev space $W^{2s,2}([0,1])$", 1),
    ("$h \\in H^{2+\\epsilon}$", "$h \\in W^{2+\\epsilon,2}$", 1),
    # summary-table recap: mirror Section 7's e_0 = H^s u (matrix H, vector u)
    ("$e_0 = A^s w$", "$e_0 = H^s u$", 1),
]
for old, new, n in LITERAL:
    got = text.count(old)
    assert got == n, "LITERAL expected %d of %r, found %d" % (n, old, got)
    text = text.replace(old, new)

# ---------------------------------------------------------------------------
# 2. Section boundaries.
# ---------------------------------------------------------------------------
lines = text.split("\n")
sec_start = {}
for i, l in enumerate(lines):
    m = re.match(r"^## (\d+)\.", l)
    if m:
        sec_start[int(m.group(1))] = i + 1  # 1-based
order = sorted(sec_start)
sec_end = {a: (sec_start[b] - 1 if b else len(lines))
           for a, b in zip(order, order[1:] + [None])}
stat_line = next(i + 1 for i, l in enumerate(lines)
                 if l.startswith("### A statistical lower bound"))

A_SECTIONS = {1, 2, 3, 4, 6, 7}


def in_A_region(L):
    if any(sec_start[s] <= L <= sec_end[s] for s in A_SECTIONS):
        return True
    if sec_start[9] <= L < stat_line:   # Section 9 deterministic part only
        return True
    return False


KACZ = (2005, 2090)  # randomized-Kaczmarz block (rows d_* live here)


def in_kaczmarz(L):
    return KACZ[0] <= L <= KACZ[1]

# ---------------------------------------------------------------------------
# 3. Token substitutions (applied to math-span contents).
# ---------------------------------------------------------------------------
# No trailing-letter guard: in math mode adjacent letters denote products, so
# Av, Ae_1, AE_m, Dw, Du, ... must all convert.  The leading guard still skips
# command tails (\Delta, \Lambda, ...) and letter-internal positions.
A_TOK = re.compile(r"(?<![A-Za-z\\])A")
D_TOK = re.compile(r"(?<![A-Za-z\\])D")
DROW = re.compile(r"(?<![A-Za-z\\])d_")            # row d_i, d_{i_t}, ...


def protect(s, token, ph):
    return s.replace(token, ph)


def A_to_H(s):
    s = s.replace("\\mathcal{A}", "\x00MCA\x00")
    s = A_TOK.sub("H", s)
    return s.replace("\x00MCA\x00", "\\mathcal{A}")


def D_to_X(s):
    s = s.replace("\\mathcal{D}", "\x00MCD\x00")
    s = D_TOK.sub("X", s)
    return s.replace("\x00MCD\x00", "\\mathcal{D}")


def rows_to_x(s):
    s = DROW.sub("x_", s)
    s = s.replace("\\lVert d\\rVert", "\\lVert x\\rVert")  # bare row in the average
    return s

# ---------------------------------------------------------------------------
# 4. Math-span scanner over the full text (returns (is_math, seg, offset)).
# ---------------------------------------------------------------------------
def scan(t):
    segs = []
    i, n, buf = 0, len(t), 0
    while i < n:
        if t[i] == "$":
            if t.startswith("$$", i):
                if i > buf:
                    segs.append((False, t[buf:i], buf))
                j = t.find("$$", i + 2)
                if j < 0:
                    raise ValueError("unbalanced $$ at %d" % i)
                segs.append((True, t[i + 2:j], i + 2))
                i = j + 2
                buf = i
            else:
                if i > buf:
                    segs.append((False, t[buf:i], buf))
                j = t.find("$", i + 1)
                if j < 0:
                    raise ValueError("unbalanced $ at %d" % i)
                segs.append((True, t[i + 1:j], i + 1))
                i = j + 1
                buf = i
        else:
            i += 1
    if buf < n:
        segs.append((False, t[buf:], buf))
    return segs


nl = [0]
for i, ch in enumerate(text):
    if ch == "\n":
        nl.append(i + 1)


def line_of(off):
    return bisect.bisect_right(nl, off)


pieces = []
n_A = n_D = n_rows = 0
for is_math, seg, off in scan(text):
    if not is_math:
        pieces.append(seg)
        continue
    L = line_of(off)
    new = seg
    if in_A_region(L):
        tmp = A_to_H(new)
        if tmp != new:
            n_A += 1
        new = tmp
    tmp = D_to_X(new)
    if tmp != new:
        n_D += 1
    new = tmp
    if in_kaczmarz(L):
        tmp = rows_to_x(new)
        if tmp != new:
            n_rows += 1
        new = tmp
    delim = "$$" if text[off - 2:off] == "$$" else "$"
    pieces.append(delim + new + delim)

result = "".join(pieces)

# ---------------------------------------------------------------------------
# 5. Sanity checks.
# ---------------------------------------------------------------------------
assert result.count("\\mathcal{A}") == text.count("\\mathcal{A}"), "mathcal A lost"
assert result.count("\\mathcal{D}") == text.count("\\mathcal{D}"), "mathcal D lost"
# the local contraction in Section 8 must survive
assert "A := I - \\gamma H" in result, "local A:=I-gamma H clobbered"

SRC.write_text(result, encoding="utf-8")
print("A->H spans:", n_A, "| D->X spans:", n_D, "| row spans:", n_rows)
