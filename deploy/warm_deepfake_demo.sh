#!/usr/bin/env bash
# Pre-compute the deepfake page's per-clip panels and detector report for the
# demo, on ASVspoof 2019 LA (the default dataset, shortest clips).
#
#   * "How good is <model>?" (Testing on every clip): every model, both
#     evaluation conditions. The as-distributed pass also caches every clip's
#     verdict, so all 200 verdicts become instant.
#   * Demo clips: verdict, "Voice or silence?" and "Where did it listen?" for
#     all six detectors.
#
# Results are cached for 90 days (deepfake cache TTL) in Redis on the cache
# volume.
#
#   bash deploy/warm_deepfake_demo.sh                # report + demo clips
#   SKIP_REPORT=1 bash deploy/warm_deepfake_demo.sh  # demo clips only
set -uo pipefail

SITE="${SITE:-https://chanugx--voxlit-web.modal.run}"
API="$SITE/api/tasks/deepfake"
BUILTIN="asvspoof2019-la"
# Demo clips (row = position in the page's library, sorted by name A-Z):
#   row 22  LA_E_1887210.flac  real
#   row 59  LA_E_3432530.flac  real
#   row 95  LA_E_4443804.flac  synthetic (A07)
#   row 107 LA_E_4920861.flac  synthetic (A11)
#   row 128 LA_E_5995426.flac  synthetic (A10)
#   row 143 LA_E_6759806.flac  real
DEMO_CLIPS="LA_E_1887210.flac LA_E_3432530.flac LA_E_4443804.flac LA_E_4920861.flac LA_E_5995426.flac LA_E_6759806.flac"
JAR="$(mktemp)"
trap 'rm -f "$JAR"' EXIT
failed=0

post() {  # post <endpoint> <json>  -> "<status> <seconds>"; follows Modal's 303s
  curl -sL -b "$JAR" -c "$JAR" --max-time 3600 --max-redirs 60 \
    "$API/$1" -H 'content-type: application/json' -d "$2" \
    -o /dev/null -w '%{http_code} %{time_total}'
}
report() {  # report <label> "<status> <seconds>"
  local status="${2% *}" secs="${2#* }"
  if [[ "$status" == 200 ]]; then printf '   %-44s ok (%.0fs)\n' "$1" "$secs"
  else printf '   %-44s HTTP %s\n' "$1" "$status"; failed=$((failed + 1)); fi
}

curl -s -c "$JAR" "$SITE/api/health" >/dev/null
models=$(curl -s "$API/models" | python3 -c 'import json,sys; print(" ".join(m["key"] for m in json.load(sys.stdin)["models"]))')
listing=$(curl -s "$API/dataset/recordings?dataset=$BUILTIN")

if [[ -n "${SKIP_REPORT:-}" ]]; then
  echo "== Detector report: skipped (SKIP_REPORT set)"
  models_for_report=""
else
  echo "== Detector report: every clip, every model ($BUILTIN)"
  models_for_report="$models"
fi
for model in $models_for_report; do
  for condition in as_distributed silence_trimmed; do
    body="{\"model\":\"$model\",\"condition\":\"$condition\",\"builtin\":\"$BUILTIN\"}"
    result=$(post scores "$body")
    report "$model $condition" "$result"
  done
done

echo "== Demo clips: verdict, silence test, heatmap"
for name in $DEMO_CLIPS; do
  rec=$(python3 -c '
import json, sys
print(next((r["recording_id"] for r in json.loads(sys.argv[1])["recordings"] if r["display_filename"] == sys.argv[2]), ""))' "$listing" "$name")
  if [[ -z "$rec" ]]; then echo "   $name not found"; failed=$((failed + 1)); continue; fi
  echo "   $name"
  for model in $models; do
    body="{\"model\":\"$model\",\"recording_id\":\"$rec\"}"
    for endpoint in run silence-probe saliency; do
      result=$(post "$endpoint" "$body")
      report "  $model $endpoint" "$result"
    done
  done
done

echo
if (( failed )); then
  echo "$failed request(s) failed. Run this script again; finished results are already cached."
  exit 1
fi
echo "Deepfake demo results cached."
