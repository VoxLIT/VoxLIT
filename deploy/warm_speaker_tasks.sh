#!/usr/bin/env bash
# Pre-compute speaker verification and diarization results on the hosted site,
# then keep them for 90 days. Called by deploy/warm_cache.sh; can run alone.
#
#   Speaker verification: every demo clip's embedding, for each model, via one
#     /batch/dataset request (the page's batch view sends the same request;
#     /verify and the saliency views reuse the per-clip embedding cache), plus,
#     for each model:
#       - Pair Verification of id11100_01-05 (enrolment) vs id11100_13 (probe),
#         uploaded as the page does: /verify and the time, frequency and
#         Integrated Gradients saliency maps;
#       - for rec_0154d898b28d04f8.wav: the Perturbation sweep (noise, pitch
#         shift, time stretch), a +2 semitone pitch shift ("Run Perturbation"),
#         and the time, frequency and IG cluster saliency maps against its
#         cluster in the all-clip batch.
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
# Pair Verification demo: enrolment clips then the probe, from the demo dataset.
PAIR_DIR="${PAIR_DIR:-$(cd "$(dirname "$0")/.." && pwd)/Backend/data/speaker_verification/vox_indian_demo_92}"
PAIR_ENROLLMENT="id11100_01.wav id11100_02.wav id11100_03.wav id11100_04.wav id11100_05.wav"
PAIR_PROBE="id11100_13.wav"
# Single "Run Perturbation" requests: <recording id> <perturbation JSON>.
PERTURBATIONS=(
  'rec_0154d898b28d04f8 {"type":"pitch_shift","params":{"pitch_shift_semitones":2}}'
)
# Saliency views as the page requests them (segment_count is always sent).
SALIENCY_VIEWS=(
  "time:-F segment_count=8"
  "frequency:-F segment_count=8 -F occlusion_axis=frequency -F band_count=8"
  "integrated_gradients:-F segment_count=8 -F saliency_method=integrated_gradients"
)
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
post_form() {  # post_form <path> <curl -F args...>  -> prints HTTP status
  local path="$1"; shift
  curl -sL -b "$JAR" -c "$JAR" --max-time 3600 --max-redirs 60 \
    "$API/$path" "$@" -o /dev/null -w '%{http_code}'
}
post_body() {  # post_body <path> <json>  -> prints the response body
  curl -sL -b "$JAR" -c "$JAR" --max-time 3600 --max-redirs 60 \
    "$API/$1" -H 'content-type: application/json' -d "$2"
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

echo "== Speaker verification perturbations"
for entry in "${PERTURBATIONS[@]}"; do
  rec="${entry%% *}"; perturbation="${entry#* }"
  for model in $(models_of verification); do
    start=$SECONDS
    status=$(post verification/perturbation \
      "{\"model\":\"$model\",\"recording_id\":\"$rec\",\"perturbation\":$perturbation}")
    report "$rec $model $perturbation" "$status" $((SECONDS - start))
  done
done

echo "== Speaker verification pair ($PAIR_ENROLLMENT vs $PAIR_PROBE)"
pair_files=()
for name in $PAIR_ENROLLMENT; do pair_files+=(-F "enrollment_files=@$PAIR_DIR/$name;type=audio/wav"); done
pair_files+=(-F "probe_file=@$PAIR_DIR/$PAIR_PROBE;type=audio/wav")
for model in $(models_of verification); do
  start=$SECONDS
  status=$(post_form verification/verify -F "model=$model" "${pair_files[@]}")
  report "$model verify" "$status" $((SECONDS - start))
  for view in "${SALIENCY_VIEWS[@]}"; do
    # shellcheck disable=SC2086  # the view's -F flags are meant to split
    start=$SECONDS
    status=$(post_form verification/explain/saliency -F "model=$model" -F reference_type=enrollment \
      ${view#*:} "${pair_files[@]}")
    report "$model saliency ${view%%:*}" "$status" $((SECONDS - start))
  done
done

echo "== Speaker verification cluster saliency ($SWEEP_RECORDINGS)"
for model in $(models_of verification); do
  batch=$(post_body verification/batch/dataset "{\"model\":\"$model\",\"recording_ids\":$ids}")
  for rec in $SWEEP_RECORDINGS; do
    # Same membership the Cluster Saliency tab derives: the target's batch
    # cluster label and every other clip with that label, in batch order.
    members=$(python3 -c '
import json, sys
ids, batch, target = json.loads(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
labels = batch.get("cluster_labels") or []
if target not in ids or len(labels) != len(ids):
    sys.exit(1)
label = labels[ids.index(target)]
print(label)
for rid, l in zip(ids, labels):
    if l == label and rid != target:
        print(rid)' "$ids" "$batch" "$rec") || { echo "   $rec $model: no batch cluster"; failed=$((failed + 1)); continue; }
    cluster=$(head -n1 <<<"$members")
    refs=()
    while read -r rid; do [[ -n "$rid" ]] && refs+=(-F "reference_recording_ids=$rid"); done < <(tail -n +2 <<<"$members")
    if (( ${#refs[@]} == 0 )); then echo "   $rec $model: $cluster has no other members, skipped"; continue; fi
    for view in "${SALIENCY_VIEWS[@]}"; do
      start=$SECONDS
      # shellcheck disable=SC2086
      status=$(post_form verification/explain/saliency -F "model=$model" -F reference_type=cluster \
        -F "target_recording_id=$rec" -F "cluster_id=$cluster" ${view#*:} "${refs[@]}")
      report "$rec $model $cluster ($(( ${#refs[@]} / 2 )) refs) ${view%%:*}" "$status" $((SECONDS - start))
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
for pattern in 'verify:emb:*' 'verify:batch:*' 'verify:pair:*' 'verify:sweep:*' 'verify:perturb:*' 'result:diar:*'; do
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
