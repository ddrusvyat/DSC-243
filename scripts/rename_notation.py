#!/usr/bin/env python3
"""Unify decision-variable notation in week1.md.

Rules (apply only inside math spans):
  * decision variables and polynomial/auxiliary arguments  x -> w
  * the optimum                                            x^\\ast -> w_\\ast
  * source-condition vector (Section 7)                    w, w_i -> u, u_i
  * Jacobi weight (Section 7)                              w_{p,q} -> \\omega_{p,q}
  * data points x_i, x_1,...,x_n, k(x,x'), ... are KEPT (they live in the
    kernel-example boxes and in Sections 8/10/9-statistical, none of which
    are touched).

The local rotation vector  w := gamma Q^T x_k  in Section 9 collides with the
new decision variable; it is renamed to v by a separate targeted edit (it
shares its subsection with an existing dummy u), so this script leaves it alone
and only the two-line StrReplace handles it.
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "week1.md"

text = SRC.read_text(encoding="utf-8")
lines = text.split("\n")

# ----- section boundaries (## N.) -------------------------------------------
sec_start = {}
for i, l in enumerate(lines):
    m = re.match(r"^## (\d+)\.", l)
    if m:
        sec_start[int(m.group(1))] = i + 1  # 1-based line no
order = sorted(sec_start)
sec_end = {}
for a, b in zip(order, order[1:] + [None]):
    sec_end[a] = (sec_start[b] - 1) if b else len(lines)

# Section 9 statistical subsection start (no-touch from here on within sec 9)
stat_line = next(i + 1 for i, l in enumerate(lines)
                 if l.startswith("### A statistical lower bound"))

def in_rename_region(L):
    for s in (1, 2, 3, 4, 6, 7):
        if sec_start[s] <= L <= sec_end[s]:
            return s
    if sec_start[9] <= L < stat_line:
        return 9
    return None

# ----- data example boxes (keep x) ------------------------------------------
DATA_BOXES = [(524, 584), (807, 878)]
def in_data_box(L):
    return any(a <= L <= b for a, b in DATA_BOXES)

# isolated decision-x living in otherwise-untouched sections (Sec 8 objective,
# Summary complexity table); treat these single lines as rename targets.
EXTRA_X_LINES = {1482, 3709, 3710}

# ----- token substitutions --------------------------------------------------
# Exclude only a *lowercase* letter (or backslash) before x/w: command-internal
# letters (\max, \exp, \boxed, \approx) and English words are lowercase, whereas
# matrix juxtaposition (Ax, Dx, Cx) uses uppercase letters that we DO rename.
X_TOK = re.compile(r"(?<![a-z\\])x(?![A-Za-z])")
W_TOK = re.compile(r"(?<![a-z\\])w(?![A-Za-z])")
XAST = re.compile(r"x\^(?:\{\\ast\}|\\ast|\*)")

DX = re.compile(r"(?<![A-Za-z])dx(?![A-Za-z])")  # integration differential

def x_to_w(s):
    s = XAST.sub(r"w_\\ast", s)
    s = X_TOK.sub("w", s)
    return DX.sub("dw", s)

def w_to_u(s):
    s = s.replace("w_{p,q}", "\x00OMEGA\x00")  # protect weight
    s = W_TOK.sub("u", s)
    return s.replace("\x00OMEGA\x00", "\\omega_{p,q}")

# ----- math-span scanner over the full text ---------------------------------
# returns list of (is_math, segment_text, start_offset)
def scan(t):
    segs = []
    i, n = 0, len(t)
    buf_start = 0
    while i < n:
        if t[i] == "$":
            if t.startswith("$$", i):
                if i > buf_start:
                    segs.append((False, t[buf_start:i], buf_start))
                j = t.find("$$", i + 2)
                if j < 0:
                    raise ValueError("unbalanced $$ at offset %d" % i)
                segs.append((True, t[i + 2:j], i + 2))
                i = j + 2
                buf_start = i
            else:
                if i > buf_start:
                    segs.append((False, t[buf_start:i], buf_start))
                j = t.find("$", i + 1)
                if j < 0:
                    raise ValueError("unbalanced $ at offset %d" % i)
                segs.append((True, t[i + 1:j], i + 1))
                i = j + 1
                buf_start = i
        else:
            i += 1
    if buf_start < n:
        segs.append((False, t[buf_start:], buf_start))
    return segs

# offset -> 1-based line number
nl_positions = [0]
for i, ch in enumerate(text):
    if ch == "\n":
        nl_positions.append(i + 1)
import bisect
def line_of(off):
    return bisect.bisect_right(nl_positions, off)

pieces = []
changed = 0
for is_math, seg, off in scan(text):
    if not is_math:
        pieces.append(seg)
        continue
    L = line_of(off)
    reg = in_rename_region(L)
    new = seg
    if reg is not None:
        if reg == 7:
            new = w_to_u(new)
        if not in_data_box(L):
            new = x_to_w(new)
    elif L in EXTRA_X_LINES:
        new = x_to_w(new)
    if new != seg:
        changed += 1
    delim = "$$" if text[off - 2:off] == "$$" else "$"
    pieces.append(delim + new + delim)

result = "".join(pieces)

# --- Section 9 cleanup -------------------------------------------------------
# (a) local rotation vector w := gamma Q^T w_k  clashes with the iterate w_k;
#     rename it to v (u is already the dummy in the same display).
# (b) atomic-measure weights w_i / w_j clash with the decision variable; rename
#     them to rho.  (The query-point w_i on the rotation-lemma line is a vector
#     and is left untouched -- it never co-occurs with a measure weight.)
SEC9_FIXES = [
    ("Setting $w := \\gamma\\,Q^\\top w_k \\in E_{2k+1}$",
     "Setting $v := \\gamma\\,Q^\\top w_k \\in E_{2k+1}$", 1),
    ("\\alpha\\,\\bigl(\\bar f(w) - \\bar f^\\ast\\bigr)",
     "\\alpha\\,\\bigl(\\bar f(v) - \\bar f^\\ast\\bigr)", 1),
    ("\\sum_{i=1}^d w_i\\,\\delta_{\\theta_i}",
     "\\sum_{i=1}^d \\rho_i\\,\\delta_{\\theta_i}", 1),
    ("and weights $w_i > 0$", "and weights $\\rho_i > 0$", 1),
    ("\\sum_{i=1}^d w_i\\,\\delta_{\\lambda_i}",
     "\\sum_{i=1}^d \\rho_i\\,\\delta_{\\lambda_i}", 3),
    ("w_i \\asymp d^{-a}", "\\rho_i \\asymp d^{-a}", 1),
    ("w_i \\asymp d^{-1}", "\\rho_i \\asymp d^{-1}", 1),
    ("\\sum_{j=1}^N w_j\\,\\delta_{\\theta_j}",
     "\\sum_{j=1}^N \\rho_j\\,\\delta_{\\theta_j}", 2),
    ("w_j:=\\int \\ell_j(\\lambda)\\,d\\mu(\\lambda)",
     "\\rho_j:=\\int \\ell_j(\\lambda)\\,d\\mu(\\lambda)", 1),
    ("r(\\theta_j)w_j", "r(\\theta_j)\\rho_j", 1),
    ("w_j=\\int \\ell_j\\,d\\mu=\\int \\ell_j^2\\,d\\mu>0",
     "\\rho_j=\\int \\ell_j\\,d\\mu=\\int \\ell_j^2\\,d\\mu>0", 1),
]
# A target-function definition h(x)=... lives OUTSIDE the kernel example box
# (in the "Effect of kernel smoothness" paragraph); its argument is a data
# point and must stay x, even though the surrounding f(x_k) are decision.
DATA_FIXES = [
    ("$h(w) = \\sin(2\\pi w) + \\tfrac{1}{2}\\cos(4\\pi w)$",
     "$h(x) = \\sin(2\\pi x) + \\tfrac{1}{2}\\cos(4\\pi x)$", 1),
]
for old, new, n in SEC9_FIXES + DATA_FIXES:
    got = result.count(old)
    assert got == n, "FIXES expected %d of %r, found %d" % (n, old, got)
    result = result.replace(old, new)

out_path = SRC.parent / "week1.new.md"
out_path.write_text(result, encoding="utf-8")
print("wrote", out_path, "changed math segments:", changed)
