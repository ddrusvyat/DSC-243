#!/usr/bin/env python3
"""Split week1.md into four standalone lecture-note documents.

    part1.md : sections 1-6   (gradient descent, Chebyshev, CG)
    part2.md : sections 7-8   (source/spectral conditions, SGD)
    part3.md : section 9      (lower bounds)
    part4.md : section 10     (high-dimensional limits of SGD)

Each part receives the Related Literature bullets and the reference entries
relevant to it, anchors for every numbered statement and tagged equation, and
hyperlinks for every reference to a result that lives in a different part.

Run from the repo root:  python3 scripts/split_notes.py
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "week1.md"

PART_OF_SEC = {1: 1, 2: 1, 3: 1, 4: 1, 6: 1, 7: 2, 8: 2, 9: 3, 10: 4}

PARTS = {
    1: dict(file="part1.md", secs=[1, 2, 3, 4, 6], label="1\u20136",
            title="Convex Quadratics I: Gradient Descent, Chebyshev "
                  "Acceleration, and Conjugate Gradient"),
    2: dict(file="part2.md", secs=[7, 8], label="7\u20138",
            title="Convex Quadratics II: Source Conditions, Spectral "
                  "Structure, and Stochastic Gradient Descent"),
    3: dict(file="part3.md", secs=[9], label="9",
            title="Convex Quadratics III: Lower Bounds for First-Order "
                  "and Stochastic Algorithms"),
    4: dict(file="part4.md", secs=[10], label="10",
            title="Convex Quadratics IV: High-Dimensional Limits of "
                  "Streaming SGD"),
}

# Related Literature bullet (bold prefix) -> part
BULLET_PART = {
    "Gradient descent and first-order complexity.": 1,
    "Chebyshev acceleration and semi-iterative methods.": 1,
    "Conjugate gradient and Krylov optimality.": 1,
    "Source conditions and spectral-decay rates.": 2,
    "Marchenko--Pastur asymptotics.": 2,
    "Average-case optimization complexity.": 2,
    "Stochastic gradient descent for least squares.": 2,
    "Interpolation and randomized Kaczmarz.": 2,
    "Lower bounds for first-order methods.": 3,
    "Lower bounds for stochastic algorithms on least squares.": 3,
    "High-dimensional limits of streaming SGD.": 4,
}

# part-specific opening paragraph for the Related Literature section, so the
# four documents are not repetitive
REL_INTRO = {
    1: "The material in this part---linear convergence of gradient descent, "
       "Chebyshev/semi-iterative acceleration, and the conjugate gradient "
       "method---is classical in numerical optimization and Krylov subspace "
       "theory. The references below collect the standard sources; our "
       "presentation emphasizes the unifying role of minimax polynomials on "
       "the spectrum of $H$.",
    2: "This part draws on several distinct bodies of work: the regularization "
       "theory of inverse problems (source conditions and spectral-decay "
       "rates), random matrix theory (Marchenko--Pastur asymptotics), the "
       "average-case analysis of optimization complexity, and the study of "
       "stochastic gradient and randomized Kaczmarz methods for least squares. "
       "The references below are organized by these themes.",
    3: "The lower bounds developed in this part build on the classical "
       "complexity theory of first-order methods and its more recent "
       "extension to stochastic algorithms on least-squares problems. The "
       "references below collect the relevant sources.",
    4: "The high-dimensional scaling limits of streaming SGD developed in this "
       "part connect to recent work on diffusion (SDE) approximations of "
       "stochastic algorithms and to average-case analysis in high dimension. "
       "The references below collect the relevant sources.",
}

# references listed in the notes but never cited in the visible text;
# they are further reading for the Kaczmarz/sketching and orthogonal
# polynomial material of part 2
UNCITED_EXTRAS = {2: ["Sze39", "Mah11", "Woo14"]}

KIND_PFX = {"Theorem": "thm", "Lemma": "lem", "Corollary": "cor",
            "Definition": "def", "Assumption": "asm", "Algorithm": "alg"}

STMT_HEADER = re.compile(
    r"\*\*(Theorem|Lemma|Corollary|Definition|Assumption)\s+(\d+\.\d+)")
ALGO_HEADER = re.compile(r"\*\*Algorithm (\d+)\*\*")


# ---------------------------------------------------------------- parsing

def section_slices(lines):
    """Return {key: (start, end)} line ranges for ## headings."""
    heads = []
    for i, l in enumerate(lines):
        m = re.match(r"^## (\d+)\.", l)
        if m:
            heads.append((int(m.group(1)), i))
        elif re.match(r"^## Summary", l):
            heads.append(("summary", i))
    slices = {}
    for (key, start), (_, nxt) in zip(heads, heads[1:] + [(None, len(lines))]):
        end = nxt
        # drop the trailing --- separator and blank lines
        while end > start and lines[end - 1].strip() in ("", "---"):
            end -= 1
        slices[key] = (start, end)
    return slices


def parse_related(lines, start, end):
    """Section 11 -> (intro paragraph, [(part, bullet_text)], comment_block)."""
    body = lines[start:end]
    intro = next(l for l in body[1:] if l.strip())
    bullets = []
    for l in body:
        m = re.match(r"^- \*\*(.+?)\*\*", l)
        if m:
            part = BULLET_PART[m.group(1)]
            bullets.append((part, l.rstrip()))
    assert len(bullets) == len(BULLET_PART)
    txt = "\n".join(body)
    m = re.search(r"<!--.*?-->", txt, flags=re.S)
    return intro.rstrip(), bullets, m.group(0)


def parse_references(lines):
    """-> ordered [(label, line)] from the ### References list."""
    i = next(i for i, l in enumerate(lines) if l.startswith("### References"))
    refs = []
    for l in lines[i + 1:]:
        if l.strip() == "---" or l.startswith("##"):
            break
        m = re.match(r"^- \[([^\]]+)\]", l)
        if m:
            refs.append((m.group(1), l.rstrip()))
    return refs


def parse_summary(lines, start, end):
    """-> {1: worst-case block, 2: spectral block} (text)."""
    body = [l for l in lines[start + 1:end]
            if not l.startswith("[\u2190") and l.strip() != "---"]
    txt = "\n".join(body).strip("\n")
    parts = re.split(r"\n(?=\*\*)", txt)
    blocks = {}
    for p in parts:
        if p.startswith("**Worst-case"):
            blocks[1] = p.strip("\n")
        elif p.startswith("**Spectral structure"):
            blocks[2] = p.strip("\n")
    assert set(blocks) == {1, 2}
    return blocks


# ---------------------------------------------------------------- anchors

def add_statement_anchors(body_lines):
    """Insert <a id="thm-2-1"></a> before each numbered statement box."""
    out = []
    seen = set()
    i, n = 0, len(body_lines)
    while i < n:
        line = body_lines[i]
        if line.strip().startswith("<div style="):
            j = i + 1
            anchor = None
            while j < n and body_lines[j].strip() != "</div>":
                m = STMT_HEADER.search(body_lines[j])
                if m:
                    kind, num = m.group(1), m.group(2)
                    anchor = "%s-%s" % (KIND_PFX[kind], num.replace(".", "-"))
                    break
                m = ALGO_HEADER.search(body_lines[j])
                if m:
                    anchor = "alg-%s" % m.group(1)
                    break
                j += 1
            if anchor:
                if anchor in seen:   # duplicate numbering in the source
                    anchor += "-bis"
                seen.add(anchor)
                # blank line after the anchor so kramdown keeps the box
                # (and any display math) a separate block element
                out += ['<a id="%s"></a>' % anchor, ""]
        out.append(line)
        i += 1
    return out


def add_equation_anchors(text):
    """Insert <a id="eq-N"></a> before each display block carrying \\tag{N}."""
    def repl(m):
        t = re.search(r"\\tag\{(\d+[ab]?)\}", m.group(0))
        if t:
            return '<a id="eq-%s"></a>\n\n%s' % (t.group(1), m.group(0))
        return m.group(0)
    return re.sub(r"\$\$.*?\$\$", repl, text, flags=re.S)


# ---------------------------------------------------------------- linking

def link_cross_references(text, cur_part, tag_part, stmt_part):
    """Turn references to results in other parts into markdown links."""
    href = lambda p: PARTS[p]["file"].replace(".md", ".html")

    def on_visible(seg):
        # tagged-equation references $(N)$
        def eq(m):
            t = m.group(1)
            p = tag_part.get(t)
            if p and p != cur_part:
                return "[%s](%s#eq-%s)" % (m.group(0), href(p), t)
            return m.group(0)
        seg = re.sub(r"\$\((\d+[ab]?)\)\$", eq, seg)

        # Sections A--B (link to the first section of the range)
        def secrange(m):
            a = int(m.group(1))
            p = PART_OF_SEC.get(a)
            if p and p != cur_part:
                return "[%s](%s#sec-%d)" % (m.group(0), href(p), a)
            return m.group(0)
        seg = re.sub(r"Sections (\d+)--(\d+)", secrange, seg)

        # Section N
        def sec(m):
            s = int(m.group(1))
            p = PART_OF_SEC.get(s)
            if p and p != cur_part:
                return "[%s](%s#sec-%d)" % (m.group(0), href(p), s)
            return m.group(0)
        seg = re.sub(r"Section (\d+)", sec, seg)

        # Theorem/Lemma/Corollary/Definition/Assumption N.M and Algorithm N
        def stmt(m):
            kind, num = m.group(1), m.group(2)
            key = (kind, num)
            p = stmt_part.get(key)
            if p and p != cur_part:
                a = "%s-%s" % (KIND_PFX[kind], num.replace(".", "-"))
                return "[%s](%s#%s)" % (m.group(0), href(p), a)
            return m.group(0)
        seg = re.sub(
            r"(Theorem|Lemma|Corollary|Definition|Assumption)\s(\d+\.\d+)",
            stmt, seg)
        seg = re.sub(r"(Algorithm)\s(\d+)\b", stmt, seg)
        return seg

    # never rewrite inside html comments
    pieces = re.split(r"(<!--.*?-->)", text, flags=re.S)
    return "".join(p if k % 2 else on_visible(p) for k, p in enumerate(pieces))


# ---------------------------------------------------------------- assembly

def nav_line(cur):
    items = []
    for p, meta in PARTS.items():
        label = "Part %s (\u00a7%s)" % ("I II III IV".split()[p - 1], meta["label"])
        if p == cur:
            items.append("**%s**" % label)
        else:
            items.append("[%s](%s)" % (label, meta["file"].replace(".md", ".html")))
    return "**Lecture notes on convex quadratics:** " + " \u00b7 ".join(items)


def contents_list(lines, secs, has_summary):
    entries = []
    for s in secs:
        m = next(re.match(r"^## (\d+\.\s+.*?)\s*\{#(sec-\d+)\}", l)
                 for l in lines if re.match(r"^## %d\." % s, l))
        entries.append("- [%s](#%s)" % (m.group(1), m.group(2)))
    entries.append("- [Related Literature](#related)")
    entries.append("- [References](#references)")
    if has_summary:
        entries.append("- [Summary](#summary)")
    return "\n".join(entries)


def main():
    lines = SRC.read_text(encoding="utf-8").split("\n")
    slices = section_slices(lines)

    overview = "\n".join(lines[
        next(i for i, l in enumerate(lines) if l.startswith("## Overview")) + 1:
        next(i for i, l in enumerate(lines) if l.startswith("### Contents"))
    ]).strip("\n")

    intro, bullets, comment = parse_related(lines, *slices[11])
    references = parse_references(lines)
    ref_labels = {lab for lab, _ in references}
    summary_blocks = parse_summary(lines, *slices["summary"])

    # home part of every tagged equation and numbered statement
    tag_part, stmt_part = {}, {}
    for s, p in PART_OF_SEC.items():
        a, b = slices[s]
        seg = "\n".join(lines[a:b])
        for t in re.findall(r"\\tag\{(\d+[ab]?)\}", seg):
            tag_part[t] = p
        for kind, num in STMT_HEADER.findall(seg):
            stmt_part[(kind, num)] = p
        for num in ALGO_HEADER.findall(seg):
            stmt_part[("Algorithm", num)] = p

    cite_re = re.compile(r"\[([A-Za-z0-9+]+(?:,\s*[A-Za-z0-9+]+)*)\](?!\()")

    for part, meta in PARTS.items():
        # ---- body: the part's sections, with anchors and cross-links
        body_lines = []
        for k, s in enumerate(meta["secs"]):
            a, b = slices[s]
            if k:
                body_lines += ["", "---", ""]
            body_lines += lines[a:b]
        body_lines = add_statement_anchors(body_lines)
        body = add_equation_anchors("\n".join(body_lines))

        # ---- related literature + references + summary for this part
        rel = [REL_INTRO[part], ""]
        rel += [b for p, b in bullets if p == part]
        if part == 2:
            rel += ["", comment]
        rel = "\n".join(rel)

        tail = body + "\n\n" + rel
        cited = set()
        for k, p in enumerate(re.split(r"(<!--.*?-->)", tail, flags=re.S)):
            if k % 2:
                continue
            for grp in cite_re.findall(p):
                toks = [t.strip() for t in grp.split(",")]
                if all(t in ref_labels for t in toks):
                    cited.update(toks)
        cited.update(UNCITED_EXTRAS.get(part, []))
        refs = "\n".join(l for lab, l in references if lab in cited)

        body = link_cross_references(body, part, tag_part, stmt_part)
        rel = link_cross_references(rel, part, tag_part, stmt_part)

        # ---- assemble
        doc = ["---", "layout: default", 'title: "%s"' % meta["title"],
               "math:", "  engine: mathjax", "---", "",
               "# %s" % meta["title"], "",
               "[\u2190 Back to course page](./)", "",
               nav_line(part), "", "---", ""]
        if part == 1:
            doc += ["## Overview", "", overview, ""]
        else:
            doc += ["This is Part %s of the lecture notes on optimization "
                    "algorithms for convex quadratics. The problem setup and "
                    "notation are introduced in [Part I](part1.html#sec-1)."
                    % "I II III IV".split()[part - 1], ""]
        doc += ["### Contents", "",
                contents_list(lines, meta["secs"], part in summary_blocks), "",
                "---", "", body, "", "---", "",
                "## Related Literature {#related}", "", rel, "",
                "### References {#references}", "", refs, ""]
        if part in summary_blocks:
            doc += ["---", "", "## Summary {#summary}", "",
                    summary_blocks[part], ""]
        doc += ["---", "", "[\u2190 Back to course page](./)", ""]

        out = HERE / meta["file"]
        out.write_text("\n".join(doc), encoding="utf-8")
        print("wrote %s (%d lines)" % (out.name, len(doc)))


if __name__ == "__main__":
    main()
