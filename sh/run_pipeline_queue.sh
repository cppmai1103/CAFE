#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "Working directory: $(pwd)"

# Each sub-script below already does its own env setup / logs/<timestamp>.out+.err
# logging (see baseline_phase2base.sh etc.) -- this wrapper just sequences them and
# waits out any already-running job first, so it's safe to kick off even while
# something else (started outside this wrapper, e.g. by hand) is mid-run.

wait_for_pipeline_idle() {
  if pgrep -f "python src/(phase1/modeling|phase2/(base|simple|expert))/(train|evaluate)\.py" > /dev/null; then
    echo "Another CAFE pipeline job is running -- waiting for it to finish before starting the next queued step..."
    while pgrep -f "python src/(phase1/modeling|phase2/(base|simple|expert))/(train|evaluate)\.py" > /dev/null; do
      sleep 30
    done
    echo "Previous job finished -- proceeding."
  fi
}

echo "=== [1/4] historical_ner_span_level_fuzzy.sh ==="
wait_for_pipeline_idle
bash sh/sh_fr/historical_ner_span_level_fuzzy.sh

echo "=== [2/4] gliner_hipe2020_de_span_level_fuzzy.sh ==="
wait_for_pipeline_idle
bash sh/sh_de/gliner_hipe2020_de_span_level_fuzzy.sh

echo "=== [3/4] historical_ner_hipe2020_de_span_level_fuzzy.sh ==="
wait_for_pipeline_idle
bash sh/sh_de/historical_ner_hipe2020_de_span_level_fuzzy.sh

echo "=== [4/4] backup_to_hf.py ==="
wait_for_pipeline_idle
python src/utils/backup_to_hf.py

echo ""
echo "=== Queue done: all 3 runs + HF backup complete. ==="
echo "Not running 'git push' automatically -- review the results, then push yourself"
echo "(or ask me to do it once you've looked things over)."
