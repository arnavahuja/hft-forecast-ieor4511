#!/bin/bash
# Full-year 2024 5 s TAQ bars for the symbols named on the command line. One process per
# symbol, and one WRDS session per process now that DataService.source reuses the plugin.
# Months already in the cache are served from disk, so this is resumable. Run ONE of these:
# WRDS allows only a few sessions per role, and tripping the cap makes the library fall
# back to prompting for a password. stdin is closed so a prompt fails instead of waiting.
PY=/opt/anaconda3/envs/backbone/bin/python
for s in "$@"; do
  if $PY - "$s" <<'PYEOF'
import sys, yaml, pathlib
p = pathlib.Path("data/manifests/sp_2024_5s.yaml")
m = yaml.safe_load(p.read_text()) if p.exists() else []
e = m if isinstance(m, list) else m.get("entries", [])
have = {x["instrument"] for x in e if x["dataset"] == "taq_trade_bars"}
sys.exit(0 if sys.argv[1] in have else 1)
PYEOF
  then echo "== $s already pulled, skipping"; continue; fi
  echo "== $s start $(date +%H:%M)"
  $PY -m hft_forecast.cli data pull-taq --manifest sp_2024_5s --start 2024-01-02 --end 2024-12-31 "$s" < /dev/null || echo "== $s FAILED"
done
echo "== worker done $(date +%H:%M)"
