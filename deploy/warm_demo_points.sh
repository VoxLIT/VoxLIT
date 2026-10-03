#!/usr/bin/env bash
# Pre-compute the explainability tabs for a few demo clips on the transcription
# and emotion pages, then keep them for 90 days.
#
#   Transcription (common-voice, whisper-base): Saliency (GradCAM) + Attention
#     (layer 6, head 0: the page's defaults)
#   Emotion (ravdess, wav2vec2): Saliency (GradCAM)
#
# Short clips are tried first; a clip whose saliency fails (e.g. the shared
# saliency code's NoneType error on some Whisper timestamps) is skipped, so the
# chosen clips are ones that work. Perturbation is not cached by the backend,
# so it always computes live.
#
#   bash deploy/warm_demo_points.sh          # 3 clips per page, prints them
#   WANT=all bash deploy/warm_demo_points.sh # every clip on both pages
set -uo pipefail

SITE="${SITE:-https://chanugx--voxlit-web.modal.run}"
API="$SITE/api"
MODAL="${MODAL:-$HOME/modal-cli/bin/modal}"
TTL_SECONDS=$((90 * 24 * 60 * 60))
WANT="${WANT:-3}"   # clips per page; WANT=all warms every clip
[[ "$WANT" == all ]] && WANT=100000
PAGE_SIZE=20   # rows per page in the Audio Dataset table
JAR="$(mktemp)"
trap 'rm -f "$JAR"' EXIT
failed=0

exec_in_container() {  # exec_in_container <snippet> <success marker>  (retries)
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

post() {  # post <endpoint> <json> -> "<status> <seconds>"; follows Modal's 303s
  curl -sL -b "$JAR" -c "$JAR" --max-time 3600 --max-redirs 60 \
    "$API/$1" -H 'content-type: application/json' -d "$2" \
    -o /dev/null -w '%{http_code} %{time_total}'
}

# candidates <dataset>: "<row> <seconds> <filename>" for every clip, shortest
# first; row is the clip's position in the page's table (metadata order).
candidates() {
  curl -s "$API/$1/metadata" | python3 -c '
import json, os, sys
rows = json.load(sys.stdin)
out = []
for i, r in enumerate(rows, 1):
    try: secs = float(r.get("duration") or 0)
    except ValueError: secs = 0
    out.append((secs, i, os.path.basename(r["filename"])))
min_secs = float(sys.argv[1])
for secs, i, name in sorted(out):
    if secs >= min_secs:
        print(i, secs, name)' "$2"
}

warm_points() {  # warm_points <dataset> <model> <with attention: yes|no>
  local dataset="$1" model="$2" attention="$3" found=0 row secs name body result
  while read -r row secs name; do
    (( found >= WANT )) && break
    body="{\"model\":\"$model\",\"method\":\"gradcam\",\"dataset\":\"$dataset\",\"dataset_file\":\"$name\"}"
    result=$(post saliency/generate "$body")
    if [[ "${result% *}" != 200 ]]; then
      echo "   skip $name (saliency HTTP ${result% *})"; continue
    fi
    if [[ "$attention" == yes ]]; then
      # The page's defaults. The endpoint answers 200 even when processing
      # failed (with an "error" field), so check for real attention pairs.
      body="{\"model\":\"$model\",\"dataset\":\"$dataset\",\"dataset_file\":\"$name\",\"layer\":6,\"head\":0}"
      pairs=$(curl -sL -b "$JAR" -c "$JAR" --max-time 3600 --max-redirs 60 \
        "$API/inferences/attention-pairs" -H 'content-type: application/json' -d "$body" |
        python3 -c 'import json,sys
try: d = json.load(sys.stdin)
except Exception: print(0); sys.exit()
print(0 if d.get("error") else len(d.get("attention_pairs") or []))')
      if [[ "$pairs" == 0 ]]; then
        echo "   skip $name (attention returned no pairs)"; continue
      fi
    fi
    found=$((found + 1))
    printf '   CHOSEN  page %d, row %2d  %-28s %.2fs\n' \
      $(( (row - 1) / PAGE_SIZE + 1 )) $(( (row - 1) % PAGE_SIZE + 1 )) "$name" "$secs"
  done < <(candidates "$dataset" "$( (( WANT >= 100000 )) && echo 0 || echo 1.0)")
  if (( WANT >= 100000 )); then echo "   $found clips cached"
  elif (( found < WANT )); then echo "   only $found of $WANT clips worked"; failed=$((failed + 1)); fi
}

curl -s -c "$JAR" "$API/health" >/dev/null

echo "== Transcription (common-voice, whisper-base): saliency + attention"
warm_points common-voice whisper-base yes

echo "== Emotion (ravdess, wav2vec2): saliency"
warm_points ravdess wav2vec2 no

echo "== Keep every cached result for 90 days"
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
fi

echo
if (( failed )); then
  echo "$failed step(s) failed. Run this script again; finished results are already cached."
  exit 1
fi
echo "Demo points cached for 90 days."
