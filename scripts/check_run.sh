#!/bin/bash
# Check a run directory: summary, minimum number of ok units, exports present.
#   scripts/check_run.sh runs/chest_ct_findings/gpt-oss-20b/smoke [MIN_OK]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

RUN_DIR=${1:?usage: check_run.sh RUN_DIR [MIN_OK]}
MIN_OK=${2:-1}
"$PYTHON" - "$RUN_DIR" "$MIN_OK" <<'EOF'
import json, sys
from pathlib import Path
from rsx.results import latest_records
from rsx.stats import summarize

run_dir, min_ok = Path(sys.argv[1]), int(sys.argv[2])
s = summarize(latest_records(run_dir).values())
print(json.dumps(s, indent=2))
ok = s["status"].get("ok", 0)
problems = []
if ok < min_ok:
    problems.append(f"only {ok} ok units (< {min_ok})")
if s["reasoning_chars"]["zero"] == s["units"] and s["units"]:
    problems.append("no record has reasoning text (reasoning parser / harmony problem?)")
summary = run_dir / "exports" / "summary.json"
if not summary.exists():
    problems.append("no exports (run `rsx export RUN_DIR`)")
else:
    ex = json.loads(summary.read_text())
    print("tables:", ex["tables"])
    print("evidence:", json.dumps(ex["evidence"], indent=1))
print("CHECK", "FAILED: " + "; ".join(problems) if problems else "OK")
sys.exit(1 if problems else 0)
EOF
