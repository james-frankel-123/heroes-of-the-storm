#!/bin/bash
# Build overleaf/draft_revision_markup.tex: changes of draft_revision.tex
# against the submitted draft.tex, via latexdiff (CTAN latexdiff-so, any
# recent version). Tables are diffed as whole blocks (old struck, new added);
# one spot where latexdiff splits a \textbf{} across the change is patched.
#   LATEXDIFF=/path/to/latexdiff-so bash "paper/paper 1/scripts/make_markup.sh"
set -euo pipefail
cd "$(dirname "$0")/../overleaf"
LD=${LATEXDIFF:-latexdiff}
perl "$LD" --type=CFONT --graphics-markup=none --math-markup=whole \
  --config="PICTUREENV=(?:picture|DIFnomarkup|tabular|tabular\*)[\w\d*@]*" \
  draft.tex draft_revision.tex > draft_revision_markup.tex
python3 - <<'EOF'
import re
p = "draft_revision_markup.tex"
s = open(p).read()
pat = re.compile(r"\\textbf\{GD\\DIFdelbegin.*?\\DIFaddend of a conservatively learned Q-function", re.S)
fix = (r"\DIFdelbegin \DIFdel{GD behavioral cloning: Argmax of the GD model at each step; the control for "
       r"pure human imitation. CQL: Argmax }\DIFdelend \DIFaddbegin \DIFadd{\textbf{GD:} the behavioral-cloning "
       r"argmax. \textbf{CQL:} argmax }\DIFaddend of a conservatively learned Q-function")
s, n = pat.subn(lambda m: fix, s)
if n != 1:
    raise SystemExit(f"markup patch matched {n} times; check the GD/CQL sentence")
open(p, "w").write(s)
EOF
echo "wrote overleaf/draft_revision_markup.tex"
