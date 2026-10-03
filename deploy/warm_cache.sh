#!/usr/bin/env bash
# Pre-compute the hosted site's model results so demo visitors never wait,
# then keep every cached result for 90 days.
#
#   1. Deepfake Voice maps          (deploy/warm_voice_maps.sh)
#   2. Transcription (common-voice) whisper-base
#   3. Emotion (ravdess)            wav2vec2
#   4. Speaker verification + diarization (deploy/warm_speaker_tasks.sh)
#   5. Extend every cached result in Redis to 90 days
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

# Run a shell snippet in the live container; retry, since `modal container
# exec` sometimes fails to connect. Fails (non-zero) if every attempt does.
exec_in_container() {  # exec_in_container <snippet> <success marker>
  local attempt container out
  for attempt in 1 2 3 4 5; do
    container=$("$MODAL" container list --json 2>/dev/null | python3 -c '
import json, sys
print(next((c["container_id"] for c in json.load(sys.stdin) if c.get("app_name") == "voxlit"), ""))')
    if [[ -n "$container" ]]; then
      out=$("$MODAL" container exec "$container" -- sh -c "$1" 2>&1)
      if grep -q "$2" <<<"$out"; then grep "$2" <<<"$out"; return 0; fi
    fi
    echo "   attempt $attempt could not reach the container, retrying..."
    sleep 15
  done
  return 1
}

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

echo "== 1/5 Deepfake Voice maps"
bash "$HERE/warm_voice_maps.sh" || failed=$((failed + 1))

chunks_of() {  # split a JSON array into JSON arrays of at most 50, one per line
  python3 -c '
import json, sys
items = json.loads(sys.argv[1])
for i in range(0, len(items), 50):
    print(json.dumps(items[i:i + 50]))' "$1"
}
count_of() { python3 -c 'import json,sys; print(len(json.loads(sys.argv[1])))' "$1"; }

# warm_dataset <dataset> <model> <batch endpoint or ""> <per-file endpoints...>
warm_dataset() {
  local dataset="$1" model="$2" batch_endpoint="$3"; shift 3
  local files; files=$(files_of "$dataset")
  local names; names=$(python3 -c 'import json,sys; print("\n".join(json.loads(sys.argv[1])))' "$files")
  local total; total=$(wc -l <<<"$names" | tr -d ' ')

  printf '   %-14s embeddings (%s clips) ... ' "$model" "$total"
  # Body built first: macOS bash 3.2 mis-parses quotes nested in "$(...)".
  body="{\"model\":\"$model\",\"dataset\":\"$dataset\",\"files\":$files,\"reduction_method\":\"pca\",\"n_components\":3}"
  check emb "$(post inferences/embeddings "$body")"

  # The batch endpoints accept at most 50 files per request.
  local chunk
  while IFS= read -r chunk; do
    printf '   %-14s frequency features (%s clips) ... ' "$model" "$(count_of "$chunk")"
    # Body built first: macOS bash 3.2 mis-parses quotes nested in "$(...)".
    body="{\"filenames\":$chunk,\"dataset\":\"$dataset\",\"model\":\"$model\"}"
    check freq "$(post inferences/audio-frequency-batch "$body")"
    if [[ -n "$batch_endpoint" ]]; then
      printf '   %-14s batch predictions (%s clips) ... ' "$model" "$(count_of "$chunk")"
      # Body built first: macOS bash 3.2 mis-parses quotes nested in "$(...)".
      body="{\"filenames\":$chunk,\"dataset\":\"$dataset\",\"model\":\"$model\"}"
      check batch "$(post "$batch_endpoint" "$body")"
    fi
  done < <(chunks_of "$files")

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

echo "== 2/5 Transcription (common-voice)"
warm_dataset common-voice whisper-base inferences/whisper-batch inferences/run inferences/whisper-accuracy

echo "== 3/5 Emotion (ravdess)"
warm_dataset ravdess wav2vec2 inferences/wav2vec2-batch inferences/run inferences/wav2vec2-detailed

echo "== 4/5 Speaker verification + diarization"
bash "$HERE/warm_speaker_tasks.sh" || failed=$((failed + 1))

echo "== 5/5 Keep every cached result for 90 days"
# Extend every matching key, then snapshot Redis to the cache volume.
snippet="n=0
for pattern in 'result:*' 'verify:emb:*' 'verify:batch:*'; do
  for k in \$(redis-cli --scan --pattern \"\$pattern\"); do
    redis-cli expire \"\$k\" $TTL_SECONDS >/dev/null; n=\$((n+1))
  done
done
redis-cli bgsave >/dev/null
echo \"   extended \$n cached results to 90 days\""
if ! exec_in_container "$snippet" "cached results to 90 days"; then
  echo "   could not extend cached results to 90 days"; failed=$((failed + 1))
else
  echo "   wait 2-3 minutes before stopping the app so the snapshot is saved"
fi

echo
if (( failed )); then
  echo "$failed step(s) failed. Run this script again; finished results are already cached."
  exit 1
fi
echo "All caches warmed and kept for 90 days."
