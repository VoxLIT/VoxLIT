#!/usr/bin/env bash
# Pre-compute every deepfake Voice map on the live site so visitors get it
# instantly. The first map per model/dataset scores all 200 clips, which takes
# longer than Modal's 150 s request limit; curl -L follows Modal's 303 "still
# working" redirects until the result is ready. Results are cached in Redis on
# the cache volume for 7 days, so run this after a deploy, before a demo.
#
#   bash deploy/warm_voice_maps.sh
set -uo pipefail

SITE="${SITE:-https://chanugx--voxlit-web.modal.run}"
API="$SITE/api/tasks/deepfake"
JAR="$(mktemp)"
trap 'rm -f "$JAR"' EXIT

curl -s -c "$JAR" "$SITE/api/health" >/dev/null

models=$(curl -s "$API/models" | python3 -c 'import json,sys; print(" ".join(m["key"] for m in json.load(sys.stdin)["models"]))')
datasets=$(curl -s "$API/builtin-datasets" | python3 -c 'import json,sys; print(" ".join(d["dataset_id"] for d in json.load(sys.stdin)["datasets"] if d.get("available")))')

failed=0
for dataset in $datasets; do
  for model in $models; do
    printf '%-16s %-14s ' "$dataset" "$model"
    # No -X POST: -d already makes this a POST, and leaving the method implicit
    # lets curl switch to GET when following Modal's 303 to the result URL.
    result=$(curl -sL -b "$JAR" -c "$JAR" --max-time 1800 --max-redirs 30 \
      "$API/embeddings" -H 'content-type: application/json' \
      -d "{\"model\":\"$model\",\"reduction_method\":\"pca\",\"n_components\":2,\"builtin\":\"$dataset\"}" \
      -w '\n%{http_code} %{time_total}')
    status=$(tail -1 <<<"$result")
    echo "HTTP ${status% *} in ${status#* }s"
    [[ "${status% *}" == 200 ]] || failed=$((failed + 1))
  done
done

echo
if (( failed )); then
  echo "$failed map(s) did not finish. Run this script again; finished clips are already cached."
  exit 1
fi
echo "All Voice maps cached."
