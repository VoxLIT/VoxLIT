#!/usr/bin/env bash
# Pre-compute the hosted site's model results so demo visitors never wait,
# then keep every cached result for 90 days.
#
#   1. Deepfake Voice maps          (deploy/warm_voice_maps.sh)
#   2. Transcription (common-voice) whisper-base
#   3. Emotion (ravdess)            wav2vec2
#   4. Extend every cached result in Redis to 90 days
#
# Requests mirror what the transcription/emotion pages send, so they fill the
# same cache entries (keyed by model + resolved file path). Redis lives on the
# Modal cache volume, so results survive the app sleeping or being stopped.
#
#   bash deploy/warm_cache.sh              # everything
set -uo pipefail

SITE="${SITE:-https://chanugx--voxlit-web.modal.run}"
API="$SITE/api"
MODAL="${MODAL:-$HOME/modal-cli/bin/modal}"
TTL_SECONDS=$((90 * 24 * 60 * 60))
HERE="$(cd "$(dirname "$0")" && pwd)"
JAR="$(mktemp)"
trap 'rm -f "$JAR"' EXIT
failed=0

# POST JSON, following Modal's 303 "still working" redirects past 150 s.
post() {
  curl -sL -b "$JAR" -c "$JAR" --max-time 3600 --max-redirs 60 \
    "$API/$1" -H 'content-type: application/json' -d "$2" \
    -o /dev/null -w '%{http_code}'
}

check() {  # check <label> <http status>
  if [[ "$2" == 200 ]]; then echo "ok"; else echo "HTTP $2"; failed=$((failed + 1)); fi
  : "$1"
}

files_of() {  # basenames of a dataset's clips, one JSON array
  curl -s "$API/$1/metadata" | python3 -c '
import json, sys, os
print(json.dumps([os.path.basename(r["filename"]) for r in json.load(sys.stdin)]))'
}

curl -s -c "$JAR" "$API/health" >/dev/null

echo "== 1/4 Deepfake Voice maps"
bash "$HERE/warm_voice_maps.sh" || failed=$((failed + 1))

warm_dataset() {  # warm_dataset <dataset> <model> <per-file endpoints...>
  local dataset="$1" model="$2"; shift 2
  local files; files=$(files_of "$dataset")
  local names; names=$(python3 -c 'import json,sys; print("\n".join(json.loads(sys.argv[1])))' "$files")
  local total; total=$(wc -l <<<"$names" | tr -d ' ')

  printf '   %-14s embeddings (%s clips) ... ' "$model" "$total"
  check emb "$(post inferences/embeddings "{\"model\":\"$model\",\"dataset\":\"$dataset\",\"files\":$files,\"reduction_method\":\"pca\",\"n_components\":3}")"

  printf '   %-14s frequency features ... ' "$model"
  check freq "$(post inferences/audio-frequency-batch "{\"filenames\":$files,\"dataset\":\"$dataset\",\"model\":\"$model\"}")"

  local i=0
  while IFS= read -r name; do
    i=$((i + 1))
    for endpoint in "$@"; do
      body="{\"model\":\"$model\",\"dataset\":\"$dataset\",\"dataset_file\":\"$name\",\"include_attention\":false}"
      status=$(post "$endpoint" "$body")
      if [[ "$status" != 200 ]]; then
        echo "   $model $endpoint $name -> HTTP $status"; failed=$((failed + 1))
      fi
    done
    (( i % 20 == 0 || i == total )) && echo "   $model clips: $i/$total"
  done <<<"$names"
}

echo "== 2/4 Transcription (common-voice)"
warm_dataset common-voice whisper-base inferences/run inferences/whisper-accuracy

echo "== 3/4 Emotion (ravdess)"
warm_dataset ravdess wav2vec2 inferences/run inferences/wav2vec2-detailed
printf '   wav2vec2 batch predictions ... '
check batch "$(post inferences/wav2vec2-batch "{\"filenames\":$(files_of ravdess),\"dataset\":\"ravdess\"}")"

echo "== 4/4 Keep every cached result for 90 days"
container=$("$MODAL" container list --json 2>/dev/null | python3 -c '
import json, sys
print(next((c["container_id"] for c in json.load(sys.stdin) if c.get("app_name") == "voxlit"), ""))')
if [[ -z "$container" ]]; then
  echo "   no running voxlit container found"; failed=$((failed + 1))
else
  # Give Redis a moment to land the last writes, then extend and snapshot.
  "$MODAL" container exec "$container" -- sh -c "
    n=0
    for k in \$(redis-cli --scan --pattern 'result:*'); do
      redis-cli expire \"\$k\" $TTL_SECONDS >/dev/null; n=\$((n+1))
    done
    redis-cli bgsave >/dev/null
    echo \"   extended \$n cached results to 90 days\""
  echo "   wait 2-3 minutes before stopping the app so the snapshot is saved"
fi

echo
if (( failed )); then
  echo "$failed step(s) failed. Run this script again; finished results are already cached."
  exit 1
fi
echo "All caches warmed and kept for 90 days."
