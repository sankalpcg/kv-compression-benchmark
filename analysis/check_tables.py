r"""Structural check on generated LaTeX tables, before anything is pasted.

    python check_tables.py ci_table_8k.tex ruler_tables_8k.tex [more ...]
    python check_tables.py                      # every *.tex in this directory

Checks, per `tabular`:
  1. every body row has as many columns as the preamble declares, counting a
     \multicolumn{n} as n;
  2. no \label is used twice across the files given, in either a \label or a
     \ref, since LaTeX resolves a duplicate silently to whichever came last and
     points the prose at the wrong table;
  3. \toprule / \botrule / \begin{table} / \end{table} are balanced.

Why this exists. `bootstrap_ci.py` shipped a `tabular` preamble hardcoded to
three budget columns while its header row was built from the data, so a
single-budget file rendered two empty columns -- a defect invisible in the
plain-text report and only visible after a LaTeX build. The preamble is now
derived from the data, and this script is the check that keeps it honest: it
reads the emitted LaTeX rather than trusting the generator, so a future change
to either generator is caught here rather than in a proof.

Exit status is 1 if anything failed, so it can gate a paste.
"""
import sys, os, re, glob


def brace_group(t, i):
    """t[i] must be '{'. Returns (contents, index just past the matching '}')."""
    if i >= len(t) or t[i] != "{":
        raise ValueError("expected '{' at %d" % i)
    d = 0
    for j in range(i, len(t)):
        if t[j] == "{":
            d += 1
        elif t[j] == "}":
            d -= 1
            if d == 0:
                return t[i + 1:j], j + 1
    raise ValueError("unbalanced brace from %d" % i)


def columns_declared(spec):
    """Count alignment columns, ignoring @{...} inserts and p{...} widths."""
    spec = re.sub(r"@\{[^{}]*\}", "", spec)
    spec = re.sub(r"[pmb]\{[^{}]*\}", "c", spec)
    return sum(1 for ch in spec if ch in "lcr")


def columns_used(row):
    r"""Count columns a body row occupies, a \multicolumn{n} counting as n."""
    n = 0
    for c in row.split("&"):
        m = re.search(r"\\multicolumn\{(\d+)\}", c)
        n += int(m.group(1)) if m else 1
    return n


def check_file(path, labels, refs):
    t = open(path, encoding="utf-8").read()
    name = os.path.basename(path)
    fails = []

    for m in re.finditer(r"\\label\{([^}]*)\}", t):
        labels.setdefault(m.group(1), []).append(name)
    for m in re.finditer(r"\\ref\{([^}]*)\}", t):
        refs.setdefault(m.group(1), []).append(name)

    for m in re.finditer(r"\\begin\{tabular\}", t):
        spec, k = brace_group(t, m.end())
        try:
            end = t.index(r"\end{tabular}", k)
        except ValueError:
            fails.append("%s: \\begin{tabular} with no \\end{tabular}" % name)
            break
        want = columns_declared(spec)
        for ln, line in enumerate(t[k:end].splitlines(), 1):
            s = line.strip()
            # A row is a line ending in the alignment terminator. Rules and
            # \cmidrule lines do not, so they are skipped without a special case.
            if not s.endswith("\\\\"):
                continue
            got = columns_used(s[:-2])
            if got != want:
                fails.append("%s: preamble {%s} declares %d columns, row has %d"
                             "\n    %s" % (name, spec, want, got, s[:90]))

    # A longtable repeats \toprule in its continuation head against a single
    # \botrule, so equality is the wrong test there. Require at least one
    # \botrule per longtable instead.
    stripped, i = [], 0
    n_long = 0
    while True:
        j = t.find(r"\begin{longtable}", i)
        if j < 0:
            stripped.append(t[i:]); break
        k = t.find(r"\end{longtable}", j)
        k = len(t) if k < 0 else k + len(r"\end{longtable}")
        seg = t[j:k]
        n_long += 1
        if seg.count(r"\botrule") < 1:
            fails.append("%s: a longtable has no \\botrule" % name)
        stripped.append(t[i:j]); i = k
    outside = "".join(stripped)
    checks = [(r"\toprule", r"\botrule", outside),
              (r"\begin{table}", r"\end{table}", t)]
    for a, b, src in checks:
        if src.count(a) != src.count(b):
            fails.append("%s: %d %s against %d %s%s"
                         % (name, src.count(a), a, src.count(b), b,
                            " (outside longtable)" if src is outside else ""))
    return fails


def main():
    paths = sys.argv[1:] or sorted(
        glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.tex")))
    if not paths:
        sys.exit("no .tex files given or found")
    labels, refs, fails = {}, {}, []
    for p in paths:
        fails += check_file(p, labels, refs)

    for lab, where in sorted(labels.items()):
        if len(where) > 1:
            fails.append("duplicate \\label{%s} in %s -- LaTeX keeps the last "
                         "one and every \\ref to it resolves to that table"
                         % (lab, ", ".join(where)))
    dangling = [l for l in refs if l not in labels]
    for lab in sorted(dangling):
        print("# note: \\ref{%s} (%s) has no \\label in the files checked -- "
              "fine if the target lives in a file not passed in"
              % (lab, ", ".join(sorted(set(refs[lab])))), file=sys.stderr)

    print("checked %d file(s), %d label(s)" % (len(paths), len(labels)))
    for f in fails:
        print("FAIL " + f)
    print("OK" if not fails else "%d problem(s)" % len(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
