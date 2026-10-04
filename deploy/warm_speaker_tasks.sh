#!/usr/bin/env bash
# Pre-compute speaker verification and diarization results on the hosted site,
# then keep them for 90 days. Called by deploy/warm_cache.sh; can run alone.
#
#   Speaker verification: every demo clip's embedding, for each model, via one
#     /batch/dataset request (the page's batch view sends the same request;
#     /verify and the saliency views reuse the per-clip embedding cache), plus
#     the Perturbation sweep (noise, pitch shift, time stretch) of
#     rec_0154d898b28d04f8.wav, for each model.
#   Speaker diarization: audio.wav and ES2004a (8-min excerpt) in the AMI demo
#     folder, for each model. The other full meetings are left to compute live.
#
#   bash deploy/warm_speaker_tasks.sh
set -uo pipefail

SITE="${SITE:-https://chanugx--voxlit-web.modal.run}"
API="$SITE/api/tasks"
MODAL="${MODAL:-$HOME/modal-cli/bin/modal}"
TTL_SECONDS=$((90 * 24 * 60 * 60))
# Clips are matched by size (the listing hides real names): audio.wav, ES2004a.
DIARIZATION_FILE_BYTES="${DIARIZATION_FILE_BYTES:-272030 15343616}"
# Demo clips whose Perturbation sweep is pre-computed (ids are fixed hashes of the filename).
SWEEP_RECORDINGS="${SWEEP_RECORDINGS:-rec_0154d898b28d04f8}"
SWEEP_TYPES="noise pitch_shift time_stretch"
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

post() {  # post <path> <json>  -> prints HTTP status; follows Modal's 303s
  curl -sL -b "$JAR" -c "$JAR" --max-time 3600 --max-redirs 60 \
    "$API/$1" -H 'content-type: application/json' -d "$2" \
    -o /dev/null -w '%{http_code}'
}
get_json() { curl -sL -b "$JAR" -c "$JAR" --max-time 600 "$API/$1"; }
models_of() { get_json "$1/models" | python3 -c 'import json,sys; print(" ".join(m["key"] for m in json.load(sys.stdin)["models"]))'; }
report() {  # report <label> <status> <seconds>
  if [[ "$2" == 200 ]]; then echo "   $1 ok (${3}s)"; else echo "   $1 HTTP $2"; failed=$((failed + 1)); fi
}

curl -s -c "$JAR" "$SITE/api/health" >/dev/null

echo "== Speaker verification (voxceleb1-indian-demo)"
ids=$(get_json verification/dataset/recordings | python3 -c '
import json, sys
print(json.dumps([r["recording_id"] for r in json.load(sys.stdin)["recordings"]]))')
echo "   $(python3 -c 'import json,sys; print(len(json.loads(sys.argv[1])))' "$ids") clips"
for model in $(models_of verification); do
  start=$SECONDS
  status=$(post verification/batch/dataset "{\"model\":\"$model\",\"recording_ids\":$ids}")
  report "$model" "$status" $((SECONDS - start))
done

echo "== Speaker verification perturbation sweeps ($SWEEP_RECORDINGS)"
for rec in $SWEEP_RECORDINGS; do
  for model in $(models_of verification); do
    for type in $SWEEP_TYPES; do
      start=$SECONDS
      status=$(post verification/perturbation/sweep \
        "{\"model\":\"$model\",\"recording_id\":\"$rec\",\"perturbation_type\":\"$type\"}")
      report "$rec $model $type" "$status" $((SECONDS - start))
    done
  done
done

echo "== Speaker diarization (audio.wav, ES2004a)"
listing=$(get_json task-b/dataset/recordings)
diarization_models=$(models_of task-b)
for size in $DIARIZATION_FILE_BYTES; do
  rec=$(python3 -c '
import json, sys
size = int(sys.argv[1])
print(next((r["recording_id"] for r in json.loads(sys.argv[2])["recordings"] if r.get("size_bytes") == size), ""))' "$size" "$listing")
  if [[ -z "$rec" ]]; then
    echo "   clip of $size bytes not found in the live diarization dataset"
    failed=$((failed + 1)); continue
  fi
  echo "   clip $rec ($size bytes)"
  for model in $diarization_models; do
    start=$SECONDS
    status=$(post task-b/run "{\"model\":\"$model\",\"recording_id\":\"$rec\"}")
    report "$model" "$status" $((SECONDS - start))
  done
done

echo "== Keep speaker results for 90 days"
# Extend every matching key, then snapshot Redis to the cache volume.
snippet="n=0
for pattern in 'verify:emb:*' 'verify:batch:*' 'verify:pair:*' 'verify:sweep:*' 'result:diar:*'; do
  for k in \$(redis-cli --scan --pattern \"\$pattern\"); do
    redis-cli expire \"\$k\" $TTL_SECONDS >/dev/null; n=\$((n+1))
  done
done
redis-cli bgsave >/dev/null
echo \"   extended \$n speaker results to 90 days\""
if ! exec_in_container "$snippet" "speaker results to 90 days"; then
  echo "   could not extend cached results to 90 days"; failed=$((failed + 1))
fi

echo
if (( failed )); then
  echo "$failed step(s) failed. Run this script again; finished results are already cached."
  exit 1
fi
echo "Speaker verification and diarization cached for 90 days."
