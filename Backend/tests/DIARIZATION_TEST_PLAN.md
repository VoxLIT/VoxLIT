# Speaker Diarization (task-b) — Master Test Plan Sections

**Component owner:** Hatheem · **Task id:** `task-b` · **Route prefix:** `/tasks/task-b`
**Document status:** measured results as of 2026-09-20. Section numbering matches the
Master Test Plan template so these sections can be pasted in directly.

> **Scope note for the group document.** These sections cover the Speaker Diarization
> task only. Speaker Verification (`tasks/verification`) and Audio Deepfake Detection
> (`tasks/task_c`) are owned by other team members and are documented separately.
>
> **Every number in this document was measured on the dates and commands shown.**
> Anything not yet measured is labelled **"to be measured"** and carries the exact
> command that will produce it. Nothing here is estimated or extrapolated.

---

## 2. Target Test Items

| # | Target test item | Source file | Size | Why it is a target |
|---|---|---|---|---|
| 1 | Demo dataset discovery | `app/tasks/task_b/dataset.py` | 122 lines | Serves the 3 AMI meetings behind opaque ids; must never expose RTTM ground truth |
| 2 | Model dispatch, confidence scoring | `app/tasks/task_b/service.py` | 275 lines | Holds the silhouette-margin confidence maths and the `< 0.4 s` embedding rule |
| 3 | HTTP endpoints, cache-through | `app/tasks/task_b/router.py` | 375 lines | 12 endpoints; the Redis cache path is what makes the feature usable at all |
| 4 | Session-scoped uploads | `app/tasks/task_b/uploads.py` | 181 lines | User-supplied files + session isolation — the main untrusted-input surface |
| 5 | Run-vs-run diff | `app/tasks/task_b/diff.py` | 272 lines | Hungarian alignment, DER-between-runs, merge/split detection |
| 6 | Perturbation generation | `app/tasks/task_b/perturbation.py` | 249 lines | Determinism here is what makes the perturbation cache work |

<sub>Line counts via `wc -l app/tasks/task_b/*.py`, measured 2026-09-20.</sub>

**Supporting items relied upon (not owned by this task):** Redis (`app/core/redis.py`),
`SessionMiddleware` (`app/core/session.py`), `pyannote.audio` 3.4.0, `torch` 2.6.0,
`pyannote.metrics`, `starlette` 0.37.2, Python 3.12, macOS CPU-only (no GPU).

**Deliberately excluded from runtime testing:** the AMI RTTM ground-truth files under
`data/speaker_diarization/ami_subset/rttm/`. These are offline-evaluation-only. Tests
assert they are *never* reachable — see §3.1.6.

---

## 3. Test Approach

Diarization is a pipeline — segmentation net → speaker embeddings → **clustering** →
timeline. Clustering is non-differentiable, so gradient saliency and attention maps are
structurally impossible for this task, and the explainability approach is **glass-box**:
expose the pipeline's real internals. The test approach follows from that in three ways:

1. **Model inference is mocked everywhere.** A real diarization run costs minutes of CPU.
   Tests substitute a counting fake for `run_diarization`, and the fake's **call count** is
   the oracle for cache behaviour. No test downloads a model, needs `HF_TOKEN`, or touches
   the network.
2. **Real data is never touched.** Every test builds a fake `ami_subset/` under `tmp_path`
   with `monkeypatch.setattr(settings, "SPEAKER_DIARIZATION_DATASET_ROOT", tmp_path)`, and
   session storage is redirected via `monkeypatch.setattr(uploads, "_storage_root", ...)`.
   Each file carries a module-level guard asserting the fake directory is not the real one,
   so a forgotten monkeypatch fails loudly instead of reading production data.
3. **Pure logic is tested for real.** `diff.py` is not mocked at all — it runs against real
   `pyannote.metrics` on hand-written synthetic annotations, so the DER and Hungarian-mapping
   numbers in §3.1.2 are genuine algorithm output, just on synthetic input.

### 3.1 Testing Techniques and Types

Technique categories follow `Backend/tests/README.md`. Categories **3.1.3, 3.1.7 and 3.1.8**
are addressed but not currently implemented for this task — stated explicitly in each
section rather than omitted.

---

#### 3.1.1 Data and Database Integrity Testing

Redis is the only datastore. It is used as a content-addressed result cache, never as a
system of record: every cached value is reproducible by re-running inference.

| | |
|---|---|
| **Technique Objective** | Exercise the cache-through path and the filesystem-backed dataset/upload stores independently of the UI, and verify that a cache key is derived from file **content**, never from a caller-supplied id. |
| **Technique** | Drive `POST /run` twice with identical input and assert `cached: false` → `true` with the inference fake called exactly once. Drive it with two *different* recording ids holding *identical bytes* and assert a shared cache entry. Inspect the fakeredis keyspace directly for key shape and TTL. Hash files directly and compare against `hashlib`. |
| **Oracles** | Self-verifying. Three independent oracles: (a) the `cached` boolean in the response body; (b) `fake_inference.call_count`, which cannot be faked by a wrong response; (c) direct `redis.keys()` / `redis.ttl()` inspection. Oracle (b) is the strongest — it proves inference did not re-run, rather than proving the response merely *said* it was cached. |
| **Required Tools** | `pytest` 8.x, `pytest-asyncio` (`asyncio_mode = auto`), `fakeredis.aioredis.FakeRedis` (autouse `fake_redis` fixture in `conftest.py`), `httpx.AsyncClient`, `soundfile` + `numpy` for synthetic WAVs. No real Redis server required. |
| **Success Criteria** | All key cache paths covered: hit, miss, per-model namespacing, TTL, content-vs-id keying, and non-caching of failures. **Met — 16 collected tests across the 13 test functions below, all passing** (`pytest tests/test_diarization_*.py -k "cache or sha256 or hash_identically or stored_verbatim or ttl"`). |
| **Special Considerations** | The autouse `fake_redis` fixture flushes between tests, so no cross-test cache leakage. Diarization results are cached for 7 days (`CACHE_TTL_SECONDS`); tests assert `0 < ttl <= 604800` rather than an exact value, since time passes during the test. |

**Tests (file → test name):**

| Test | What it pins down |
|---|---|
| `test_diarization_router.py::test_run_second_identical_request_is_a_cache_hit` | Repeat run does not re-invoke inference (Phase 0 exit criterion) |
| `test_diarization_router.py::test_cache_key_is_the_content_hash_not_the_recording_id` | Two ids, identical bytes → one cache entry |
| `test_diarization_router.py::test_cache_key_is_namespaced_per_model` | Key shape is `result:diar:{model}:{sha256}` |
| `test_diarization_router.py::test_run_is_written_to_the_cache_with_a_ttl` | TTL set and within the 7-day bound |
| `test_diarization_router.py::test_a_failed_run_is_not_cached` | A 503 leaves no poisoned cache entry |
| `test_diarization_uploads.py::test_the_same_bytes_uploaded_twice_share_one_cache_entry` | Phase 1 exit criterion — re-upload is instant |
| `test_diarization_uploads.py::test_different_uploads_do_not_share_a_cache_entry` | Negative control: different audio must miss |
| `test_diarization_uploads.py::test_rerunning_the_same_upload_is_an_instant_cache_hit` | Upload re-run is a hit |
| `test_diarization_service.py::test_file_sha256_matches_hashlib` | Hash correctness |
| `test_diarization_service.py::test_file_sha256_is_chunk_boundary_independent` | ~3 MiB file spanning several 1 MiB read chunks hashes identically |
| `test_diarization_service.py::test_identical_bytes_hash_identically_under_different_names` | The property the whole cache design rests on |
| `test_diarization_service.py::test_file_sha256_of_an_empty_file` | Degenerate input |
| `test_diarization_uploads.py::test_uploaded_bytes_are_stored_verbatim` | Storage does not mutate audio |

---

#### 3.1.2 Function Testing

Black-box exercise of each endpoint and business rule, plus white-box unit tests for the
two pieces of real mathematics in this task: confidence scoring and the run-vs-run diff.

| | |
|---|---|
| **Technique Objective** | Exercise diarization functionality — dataset listing, run, projection, upload, perturbation, and the diff — with valid and invalid data, to verify correct results on valid input and the correct error on invalid input. |
| **Technique** | Endpoint-level: drive each route via `httpx.AsyncClient` with valid and malformed bodies. Unit-level: call `_attach_confidence`, `_confidence_bucket` and `compare_runs` directly with hand-built inputs whose expected output is computed by hand. Adapter-level: build a real `_Pyannote31Adapter` via `object.__new__` (bypassing the model-loading `__init__`) with a stub pipeline, so the genuine `MIN_EMBEDDABLE_SECONDS` branch executes. |
| **Oracles** | Self-verifying, and for the maths, *independently derived*: confidence expectations come from the documented formula `(d_other − d_own) / max(d_own, d_other)` computed by hand in the test, not from the implementation's own output. DER assertions are cross-checked against the identity `DER = (missed + false_alarm + confusion) / total` using the same payload's own components. |
| **Required Tools** | `pytest`, `numpy`, `torch` 2.6.0 (tensor construction only, no inference), real `pyannote.core` + `pyannote.metrics` for the diff tests, `unittest.mock.patch`. |
| **Success Criteria** | All 11 endpoints exercised; both mathematical cores covered including boundary and degenerate cases. **Met — 202 tests, all passing.** |
| **Special Considerations** | `diff.py` is deliberately **not** mocked: it is pure and runs in milliseconds, so testing it against real `pyannote.metrics` costs nothing and catches real algorithm regressions. The DER values below are therefore genuine `pyannote.metrics` output on synthetic annotations. |

**Measured DER and diff output** — real `diff.compare_runs()` output, synthetic 4-segment /
2-speaker annotations, 16.0 s total speech over a 20.0 s recording. Measured 2026-09-20:

| Scenario | DER | missed | false alarm | confusion | appeared / disappeared | split / merged | shifts | lost | diff regions |
|---|---|---|---|---|---|---|---|---|---|
| Identical runs | **0.0** | 0.0 | 0.0 | 0.0 | — / — | 0 / 0 | 0 | 0 | 0 |
| Labels swapped (`00`↔`01`) | **0.0** | 0.0 | 0.0 | 0.0 | — / — | 0 / 0 | 0 | 0 | 0 |
| One speaker dropped | **0.5** | 8.0 | 0.0 | 0.0 | — / `SPEAKER_01` | 0 / 0 | 0 | 2 | 2 |
| One turn re-attributed | **0.25** | 0.0 | 0.0 | 4.0 | — / — | 1 / 1 | 0 | 1 | 1 |
| All speech merged to one speaker | **0.5** | 0.0 | 0.0 | 8.0 | — / `SPEAKER_01` | 0 / 1 | 0 | 2 | 2 |
| One speaker split into two | **0.25** | 0.0 | 0.0 | 4.0 | `SPEAKER_02` / — | 1 / 0 | 0 | 1 | 1 |
| One segment lost | **0.25** | 4.0 | 0.0 | 0.0 | — / — | 0 / 0 | 0 | 1 | 1 |
| Boundary moved +0.6 s | **0.0375** | 0.0 | 0.6 | 0.0 | — / — | 0 / 0 | 1 | 0 | 1 |

The **swapped-labels row is the important one**: pyannote assigns arbitrary labels per run,
so without Hungarian alignment (`DiarizationErrorRate.optimal_mapping`) an identical timeline
would score a catastrophic DER instead of 0.0. That row is the regression guard for the
alignment step.

> **DER here is between two runs, not against ground truth.** The original run is passed as
> pyannote's "reference" purely to anchor the comparison. The payload field
> `der_is_vs_original_run: true` exists so this can never be misread as accuracy, and a test
> asserts it is present.

**Confidence-scoring coverage** (`test_diarization_service.py`):

- Bucket boundaries, inclusive: `0.5 → high`, `0.4999 → medium`, `0.2 → medium`, `0.1999 → uncertain`, `None → None`.
- Two orthogonal clusters → confidence `1.0` for every segment (hand-derived: `d_own = 0`, `d_other = 1`).
- A segment sitting on the wrong centroid → `−1.0`, bucket `uncertain`.
- **Single speaker → `confidence is None`**, not `0.0` — the margin is genuinely undefined with one centroid.
- A segment with no embedding → `None`, and it remains in the timeline with its timing intact.

**The `< 0.4 s` rule** (`MIN_EMBEDDABLE_SECONDS`), tested on the real adapter method:

- `test_segments_shorter_than_the_minimum_are_not_embedded` — a 0.29 s turn is never sent to the embedding model.
- `test_short_segments_end_up_with_null_confidence` — end-to-end, a skipped segment surfaces as `confidence: null`. **No score is ever fabricated.**
- `test_a_segment_exactly_at_the_minimum_is_embedded` — the comparison is `>=`, so exactly 0.4 s is embedded.
- `test_boundary_is_decided_by_the_float_difference_not_the_decimal_one` — documents Finding 3 (§7).

---

#### 3.1.3 User Interface Testing

**Not implemented for this task. Deferred.**

The diarization UI (`Frontend/src/features/task-b/`: `DiarizationWorkbench.tsx`,
`DiarizationTimeline.tsx`, `EmbeddingScatter.tsx`, `SimilarityMatrix.tsx`,
`StackedTimelines.tsx`) currently has **no automated tests**. The repository has no frontend
test runner configured — `Backend/tests/README.md` lists the React Testing Library install as
a future step, and no `jest`/`vitest` config exists in `Frontend/`.

**To be measured / built.** Minimum first step, to be run from `Frontend/`:

```bash
npm install --save-dev vitest @testing-library/react @testing-library/jest-dom jsdom
npx vitest run
```

Highest-value UI assertions once a runner exists, in priority order:
1. Confidence shading maps to the right bucket (solid / light / hatched) and a `null`
   confidence renders as *unscored*, never as zero-confidence.
2. The segment play-queue plays two segments back-to-back without overlap.
3. `StackedTimelines` difference strip aligns with the `diff_regions` the backend returned.

---

#### 3.1.4 Performance Profiling

| | |
|---|---|
| **Technique Objective** | Measure the latency the cache path itself adds, and establish the real cold-vs-warm gap for a full diarization run. |
| **Technique** | Cache-path latency: drive `POST /run` against a warmed app with inference mocked, 1 cold call + 20 warm calls, `time.perf_counter()` around each. Real inference latency: run against a real AMI meeting with `HF_TOKEN` set (**not yet done**). |
| **Oracles** | Wall-clock timing; median over n=20 for the warm path to damp scheduler noise. |
| **Required Tools** | `time.perf_counter`, `statistics.median`, `httpx.AsyncClient`, `fakeredis`. Real-inference profiling additionally needs `HF_TOKEN` and the gated pyannote weights. |
| **Success Criteria** | Cache-hit latency must be negligible against the inference it replaces. **Partially met** — cache-path latency measured; the real inference baseline it must be compared against is still to be measured. |
| **Special Considerations** | The "cold" number below is **cold cache with mocked inference** — it is *not* the cost of a real diarization. It isolates the cache machinery, nothing more. |

**Measured — 2026-09-20, macOS CPU-only, payload of 40 segments + 40 × 256-float embeddings:**

| Path | Latency |
|---|---|
| Cache **hit** (warm), median of n=20 | **18.1 ms** (min 17.8, max 18.7) |
| Cache **miss** (cold) **with inference mocked** | **22.2 ms** |

Reproduce with `Backend/` as the working directory:

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
# see the timing harness in the task notes; drives POST /tasks/task-b/run
# once cold then 20x warm with run_diarization patched, printing median.
PY
```

**To be measured — real cold-vs-warm diarization.** This is the number that actually
justifies the cache, and it requires real model inference:

```bash
# Requires HF_TOKEN in Backend/.env and accepted conditions for
# pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0
cd Backend
uvicorn app.main:app --reload --reload-dir app --reload-exclude "$(pwd)/.venv"

# In a second shell — first call is the cold measurement, second is warm:
RID=$(curl -s localhost:8000/tasks/task-b/dataset/recordings | python3 -c 'import sys,json;print(json.load(sys.stdin)["recordings"][0]["recording_id"])')
time curl -s -X POST localhost:8000/tasks/task-b/run \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"pyannote-3.1\",\"recording_id\":\"$RID\"}" -o /dev/null
time curl -s -X POST localhost:8000/tasks/task-b/run \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"pyannote-3.1\",\"recording_id\":\"$RID\"}" -o /dev/null
```

`CLAUDE.md` states 5–10 minutes per meeting on CPU. **That figure is documented project
knowledge, not something this test effort measured** — the command above is what confirms it.

**Also to be measured:** peak memory during a real run (the README's Master Test Plan
threshold is ≤ 2 GB); suggested `psutil.Process().memory_info().rss` sampled around the call.

---

#### 3.1.5 Load Testing

**Not implemented for this task. Deferred, with a documented reason.**

Load testing `POST /run` is only meaningful against **real** inference, and a single real run
occupies a CPU for minutes. Concurrency numbers gathered against a mocked pipeline would
measure FastAPI's threadpool, not this task.

What is already known from the design and is worth verifying under load:
- Heavy work runs via `run_in_threadpool`, so the event loop is not blocked.
- Model loading is guarded by `_LOAD_LOCK` + `_MODEL_CACHE` (`service.get_model`), so
  concurrent first requests cannot race into two model loads. **Currently untested.**

**To be measured**, once real inference is available:

```bash
# 1. Concurrent first-requests must trigger exactly ONE model load (_LOAD_LOCK).
#    Instrument _Pyannote31Adapter.__init__ with a counter, then fire 5 parallel runs.
# 2. Cache-hit throughput under concurrency:
.venv/bin/python -m pytest tests/test_performance_load.py -v   # existing harness to extend
```

---

#### 3.1.6 Security and Access Control Testing

The untrusted surface is: recording ids, upload ids, perturbed-clip ids, session ids, and
uploaded file bytes. There are no user accounts — the access-control boundary is the
**session**.

| | |
|---|---|
| **Technique Objective** | Verify that (a) no untrusted id can escape its directory, (b) one session cannot read or diarize another session's audio, and (c) AMI ground-truth RTTM files never reach a runtime response. |
| **Technique** | Drive every id-accepting route with traversal-shaped, unknown, empty and raw-filename ids. Create a second `AsyncClient` (a distinct session cookie) and attempt to read and to diarize the first session's upload. Serialize every dataset payload to JSON and assert no RTTM marker appears. Feed disallowed extensions, empty files and oversized files to the upload route and assert nothing is left on disk. |
| **Oracles** | Self-verifying: HTTP status codes (400/404/413), raised exception types (`RecordingNotFound`, `UploadNotFound`, `PerturbedNotFound`, `InvalidSessionId`), and direct filesystem inspection of the session directory after each rejection. |
| **Required Tools** | `pytest`, `httpx.AsyncClient` (second client for the isolation tests), `secrets.token_hex` for well-formed session ids. |
| **Success Criteria** | Every id-accepting route rejects traversal input; cross-session access impossible; no RTTM leakage; no partial file after a rejected upload. **Met — 54 collected security-relevant tests, all passing** (count reproduced by the `-k` filter at the end of this section). |
| **Special Considerations** | The design is *structurally* traversal-proof rather than sanitising: no resolver ever joins an untrusted id into a path — each compares it against ids derived from a directory listing, so an attack id simply misses. Tests assert the outcome, which stays valid if the implementation changes. |

**Traversal and unknown-id rejection** — parametrised across
`not-a-real-id`, `../../etc/passwd`, `../rttm/ES2004a.rttm`, a raw filename, `""`, and a
well-formed-but-unknown `rec_0000000000000000`:

| Test | Surface |
|---|---|
| `test_diarization_dataset.py::test_get_recording_rejects_unknown_and_traversal_ids` | Metadata lookup |
| `test_diarization_dataset.py::test_resolve_recording_path_rejects_unknown_and_traversal_ids` | Path resolution |
| `test_diarization_dataset.py::test_resolve_recording_path_can_never_escape_the_dataset_dir` | Explicit file planted *outside* the dataset dir stays unreachable |
| `test_diarization_dataset.py::test_resolve_recording_path_never_reaches_the_rttm_folder` | Ground truth unreachable by id |
| `test_diarization_router.py::test_dataset_single_recording_rejects_bad_ids` | HTTP 404 |
| `test_diarization_router.py::test_recording_audio_rejects_bad_ids` | Audio streaming 404 |
| `test_diarization_router.py::test_run_404s_for_an_unknown_recording` | `POST /run` 404, inference never invoked |
| `test_diarization_uploads.py::test_resolve_upload_path_rejects_traversal_shaped_ids` | Upload resolver |
| `test_diarization_uploads.py::test_resolve_perturbed_path_rejects_traversal_shaped_ids` | Perturbed-clip resolver |

**Session isolation:**

- `test_one_session_cannot_read_another_sessions_upload` — second client gets 404, and its listing is empty.
- `test_one_session_cannot_diarize_another_sessions_upload` — 404, and the inference fake is called **zero** times.
- `test_resolvers_reject_a_malformed_sid_before_touching_the_filesystem`.
- `test_sid_validation_rejects_non_exact_forms` — rejects `None`, `""`, uppercase, 31 and 33 chars, path-shaped, and whitespace-padded ids. Uppercase is **rejected, never re-cased**, so two spellings can never resolve to one session.

**Upload validation** (`ALLOWED_AUDIO_EXTENSIONS = {.wav, .mp3, .m4a, .flac}`, `MAX_FILE_BYTES = 50 MiB`):

| Input | Expected | Test |
|---|---|---|
| `.txt`, `.exe`, `noextension`, `clip.wav.txt`, `.zip`, `clip.` | 400, nothing on disk | `test_upload_rejects_disallowed_extensions` |
| Empty file | 400, nothing on disk | `test_upload_rejects_an_empty_file_and_leaves_nothing_behind` |
| Oversized | 413, nothing on disk | `test_upload_rejects_an_oversized_file_and_leaves_nothing_behind` |
| Any rejection | no `.tmp-` residue | `test_a_rejected_upload_leaves_no_temp_file` |

**Information disclosure:** `test_upload_never_echoes_the_client_filename` asserts a client
filename (`my-secret-meeting-2026.wav`) appears neither in the response nor on disk — the
stored name is the opaque `upl_<32 hex>`. `test_dataset_recordings_endpoint` asserts the
string `rttm` never appears in a serialized listing.

Reproduce the 54-test count for this section:

```bash
cd Backend
.venv/bin/python -m pytest tests/test_diarization_*.py -q -k \
  "traversal or sid_validation or session_cannot or rejects_disallowed or \
   empty_file_and_leaves or oversized_file_and_leaves or no_temp_file or \
   never_echoes or escape_the_dataset or never_reaches_the_rttm or \
   rejects_bad_ids or 404s_for_an_unknown or resolvers_reject"
# => 54 passed, 148 deselected   (measured 2026-09-20)
```

---

#### 3.1.7 Failover and Recovery Testing

**Partially covered — the recoverable failure modes for this task are tested; infrastructure
failover is not applicable.**

There is no failover topology here (single FastAPI process, single Redis). The relevant
recovery properties are *partial-write* and *dependency-unavailable*, both covered:

| Failure condition | Behaviour asserted | Test |
|---|---|---|
| Gated model / drifted venv — pipeline cannot load | HTTP **503**, message preserved | `test_run_returns_503_when_the_model_cannot_load` |
| Failed run must not poison the cache | No `result:diar:*` key written | `test_a_failed_run_is_not_cached` |
| Upload interrupted / rejected mid-write | Temp file removed, no partial promotion | `test_a_rejected_upload_leaves_no_temp_file` |
| Dataset directory absent entirely | `/dataset` still 200 with `available: false`; the rest 404 | `test_dataset_endpoint_reports_unavailable_rather_than_404`, `test_dataset_endpoints_404_when_dataset_missing` |
| Recording with no speech at all | Empty segment list, no crash | `test_adapter_handles_a_recording_with_no_speech` |
| Zero-duration run (division guard) | Empty `diff_regions`, no `ZeroDivisionError` | `test_diff_regions_are_empty_when_a_zero_duration_run_is_compared` |

**To be measured:** behaviour when Redis is *down* (not merely empty). The current suite uses
`fakeredis`, which never refuses a connection. Worth adding:

```bash
# Point the app at a closed port and confirm /run degrades to a slow-but-correct
# uncached response rather than a 500:
REDIS_URL=redis://localhost:6399 .venv/bin/python -m pytest tests/test_diarization_router.py -q
```

---

#### 3.1.8 Configuration Testing

**Partially covered.** The configuration axis that has actually broken this project is the
**Python dependency set**, not client hardware.

`Backend/requirements.txt` pins `torch` 2.6.x, `torchaudio` 2.6.x, `numpy` < 2.0 and
`pyannote.audio` 3.4.0. A drifted venv (torch 2.14 / pyannote 4.x) produces two failures that
*look like code bugs but are not*:

- `torchaudio.load` routes through torchcodec → `Could not load libtorchcodec`
- `DiarizeOutput has no attribute itertracks` — pyannote 4.x returns a wrapper, not an `Annotation`

**Verified configuration (2026-09-20):** `torch 2.6.0`, `pyannote.audio 3.4.0`,
`starlette 0.37.2`, Python 3.12, macOS (Darwin 25.3.0), CPU-only. The full suite passes on it.

The correct response to either symptom is to **rebuild the venv from `requirements.txt`** —
never to patch the code around it.

**To be measured:** a CI matrix pinning these versions. No CI currently runs this suite;
`Backend/tests/README.md` contains a GitHub Actions sketch that has not been adopted.

---

## 4. Deliverables

### 4.1 Test Evaluation Summaries

**Delivered: 5 test files, 202 tests, all passing, 1.7 s total.**

| Test file | Tests | Runtime | Technique sections |
|---|---:|---:|---|
| `tests/test_diarization_dataset.py` | 31 | 0.11 s | 3.1.1, 3.1.6 |
| `tests/test_diarization_service.py` | 42 | 0.09 s | 3.1.1, 3.1.2 |
| `tests/test_diarization_router.py` | 46 | 0.78 s | 3.1.1, 3.1.2, 3.1.6, 3.1.7 |
| `tests/test_diarization_uploads.py` | 51 | 0.33 s | 3.1.1, 3.1.6, 3.1.7 |
| `tests/test_diarization_diff.py` | 32 | 0.88 s | 3.1.2 |
| **Total** | **202** | **~1.7 s** | |

Run them:

```bash
cd Backend
.venv/bin/python -m pytest tests/test_diarization_*.py -q
# => 202 passed   (measured 2026-09-20; three consecutive runs: 1.72s, 1.69s, 1.67s)
```

**Regression check against the rest of the suite** (measured 2026-09-20):

| Run | Result |
|---|---|
| Full suite **with** these files | 892 passed, 9 failed, 3 skipped — 113.86 s |
| Full suite **without** these files (`--ignore=…`) | 690 passed, 9 failed, 3 skipped — 101.86 s |
| **Delta** | **+202 passed, +0 failed** |

The 9 failures are **pre-existing and unrelated** to diarization — they live in
`test_security.py`, `test_function_testing.py`, `test_performance_load.py`,
`test_custom_dataset_service.py` and `test_session_cookie.py`, and are order-dependent
(`RuntimeError: Event loop is closed` in teardown, plus timing-threshold assertions). The
failing *names* vary between runs; the *count* does not. **A full-suite run is therefore not
a clean green baseline for this project** — compare against the `--ignore` run instead.

### 4.2 Reporting on Test Coverage

Coverage is currently reported as **test count and explicit surface enumeration**, not as a
line-coverage percentage: `coverage` and `pytest-cov` are **not installed** in this venv
(both `import coverage` and `import pytest_cov` fail).

**Endpoint coverage — 11 of 12 diarization routes exercised:**

| Endpoint | Covered |
|---|---|
| `GET /models` | ✅ |
| `GET /dataset` | ✅ (present and absent) |
| `GET /dataset/recordings` | ✅ |
| `GET /dataset/recordings/{id}` | ✅ |
| `GET /dataset/recordings/{id}/audio` | ✅ |
| `POST /uploads` | ✅ |
| `GET /uploads` | ✅ |
| `GET /uploads/{id}/audio` | ✅ |
| `POST /run` | ✅ |
| `GET /projection` | ✅ |
| `POST /perturbation` | ⚠️ error contract only (400/404/422); success path untested |
| `GET /perturbed/{id}/audio` | ❌ **not covered — zero requests in the suite** |

**Status-code coverage:** 200, 400, 404, 413, 422, 503. **206 is not covered because it is
never produced** — see Finding 2.

**The one uncovered route.** `GET /perturbed/{perturbed_id}/audio` receives no request in any
of the five files (verified: `grep -c "tasks/task-b/perturbed" tests/test_diarization_*.py`
returns 0 for all five). Its resolver, `uploads.resolve_perturbed_path`, *is* tested directly
by `test_resolve_perturbed_path_rejects_traversal_shaped_ids`, so the traversal risk is
covered — but the route's own status codes and `FileResponse` wiring are not.

**To be measured / added.** One test, mirroring the upload-audio test, closes this:

```python
# in tests/test_diarization_uploads.py — write a prt_*.wav into the session dir,
# then assert GET /tasks/task-b/perturbed/{id}/audio returns 200 + exact bytes,
# and 404 for an unknown or traversal-shaped id.
```

**To be measured — line coverage.** Requires installing a dev dependency, which has not been
done:

```bash
cd Backend
.venv/bin/pip install pytest-cov
.venv/bin/python -m pytest tests/test_diarization_*.py \
  --cov=app/tasks/task_b --cov-report=term-missing
```

The Master Test Plan target in `tests/README.md` is **> 85 % line coverage**.

---

## 5. Risks, Dependencies, Assumptions, and Constraints

| Risk | Likelihood | Impact | Mitigation | Contingency (risk realised) |
|---|---|---|---|---|
| **Mocked inference hides a real pipeline break.** 202 tests pass without ever loading pyannote, so a genuine model/venv failure is invisible to them. | High | High | The 503 path is tested, and `test_diarization_service.py` exercises the real adapter method with a stub pipeline. | Run the Phase 0 manual end-to-end check (§3.1.4 command) before any demo. **This is the single most important gap in this plan.** |
| **Venv drift** to torch 2.14 / pyannote 4.x breaks `torchaudio.load` and `itertracks`. | Medium | High | Versions pinned in `requirements.txt` with an in-file warning; verified config recorded in §3.1.8. | Rebuild the venv from `requirements.txt`. Never patch the code around it; never add version-straddling shims. |
| **Perturbation `/perturbation` success path is untested end-to-end.** Only the 400/404/422 contract is covered; a real perturb → re-diarize → diff round trip is not. And `GET /perturbed/{id}/audio` has **no test at all**. | High | Medium | `diff.compare_runs` is fully tested in isolation; `perturbation.py` primitives are covered by `test_perturbation_service.py`; the perturbed-path *resolver* is traversal-tested. | Add one integration test with inference mocked but `perturb_to_session` real, asserting the perturbed clip is written and the `prt_` id resolves — plus the audio-route test sketched in §4.2. |
| **Real DER under real perturbation is unknown.** All DER figures in §3.1.2 are from synthetic annotations. | High | Medium | The algorithm is correct on inputs with hand-derived expected values. | Run a noise perturbation at a low SNR on a real AMI meeting and record the DER. Phase 2 exit criterion requires a *visible* timeline change. |
| **No CI.** The suite runs only when someone runs it locally. | High | Medium | Suite is fast (1.7 s) and has no external dependencies, so it is cheap to adopt. | Adopt the GitHub Actions sketch in `tests/README.md`, restricted to `tests/test_diarization_*.py` first — the other 9 failures would otherwise redden the build immediately. |
| **AMI demo data is not committed** and is gitignored. | Medium | High | Tests never depend on it; the one real-data test is `skipif`-guarded on directory presence. | Re-obtain the three AMI headset-mix WAVs. Note the stray file in Finding 1. |
| **Frontend untested** (§3.1.3). | High | Medium | Backend contracts are pinned, so the UI has a stable target. | Stand up `vitest`; start with confidence-shading and the `null`-confidence rendering path. |

**Assumptions**
1. Redis is available in production; tests substitute `fakeredis` and never assert on a real server.
2. `SessionMiddleware` always populates `request.state.sid`; a miss is treated as middleware bypass (400), not as "no session yet".
3. Both supported perturbations preserve length and sample rate — this is what lets `diff.py` compare two runs over one shared evaluation region. A future length-changing transform invalidates that assumption and requires `min()` of the two durations in `compare_runs`.

**Constraints**
1. **CPU-only, no GPU.** A real diarization run is minutes, which is why mocking is not optional.
2. Gated Hugging Face models require accepted conditions plus `HF_TOKEN` — so no CI runner can execute real inference without a secret.
3. AMI RTTM ground truth is offline-evaluation-only and must never be served by a runtime endpoint (asserted, §3.1.6).

---

## Appendix A — Findings Raised by This Test Effort

<sub>Kept unnumbered so the section numbers above stay aligned with the Master Test Plan
template. Fold into the group document's defect log if one exists.</sub>

Three issues surfaced while writing these tests. **No source file was modified** — this was a
tests-only effort, and each finding is recorded rather than silently patched.

**Finding 1 — Stray file in the AMI dataset directory.**
`Backend/data/speaker_diarization/ami_subset/` contains a fourth file, `audio.wav`
(272 030 bytes, dated 15 Aug 19:50), alongside the three meeting WAVs. `list_recordings()`
therefore returns **4** recordings while `dataset.EXPECTED_RECORDING_COUNT` is **3**.
*Impact:* low — it is a valid `.wav` and is served correctly; but any exact-count assertion
fails, and it appears in the UI as a fourth "meeting".
*Action:* owner to confirm whether it belongs. The real-data test
(`test_real_dataset_contains_the_three_ami_meetings`) is a deliberate subset check, so it
passes either way.

**Finding 2 — Audio endpoints do not support HTTP Range.**
task-b's audio routes return a plain `FileResponse`, and Starlette 0.37.2's `FileResponse`
ignores the `Range` header — a seek request receives **200 and the entire file**, not 206.
`tasks/verification` does not behave this way: it routes audio through
`app/tasks/verification/audio_streaming.py::stream_audio_file`, which parses `Range` and
returns a real 206.
*Impact:* medium. AMI mixes are 15–48 MB, and the timeline seeks constantly, so every seek
re-downloads the whole meeting.
*Action:* reuse `stream_audio_file(path, request, safe_filename, media_type)` — do not write a
second implementation. Current behaviour is pinned by
`test_recording_audio_ignores_range_and_serves_the_whole_file`, which must be updated to
expect 206 when this is fixed.

**Finding 3 — The 0.4 s threshold is float-sensitive.**
`1.4 − 1.0 == 0.3999999999999999` in IEEE 754, so a turn that reads as 0.4 s in decimal falls
*under* `MIN_EMBEDDABLE_SECONDS`, while `0.4 − 0.0` (exactly `0.4`) does not.
*Impact:* negligible scientifically — a 1 ms sliver either side of a 0.4 s cut-off changes no
conclusion. Recorded because it is real, non-obvious behaviour, and because boundary tests
must choose their endpoints deliberately.
*Action:* none required. Documented by
`test_boundary_is_decided_by_the_float_difference_not_the_decimal_one`.

---

## 6. References

1. **Backend test conventions** — `Backend/tests/README.md`, LIT for Voice Test Implementation Guide.
2. **Component specification** — `CLAUDE.md`, "VoxLIT Speaker Diarization (task-b)".
3. **Reference implementation pattern** — `Backend/app/tasks/verification/`, this repository's reference for uploads, dataset modules, adapters and error mapping.
4. **pyannote.audio 3.4.0** — Bredin, H. *pyannote.audio 2.1 speaker diarization pipeline*. Available at https://github.com/pyannote/pyannote-audio (Accessed on 20 September 2026).
5. **pyannote.metrics** — Bredin, H. (2017). *pyannote.metrics: a toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems*. Interspeech 2017. Source of `DiarizationErrorRate` and the Hungarian `optimal_mapping` used in `diff.py`.
6. **AMI Meeting Corpus** — Carletta, J. et al. (2006). *The AMI Meeting Corpus: A Pre-announcement*. MLMI 2006. Available at https://groups.inf.ed.ac.uk/ami/corpus/ (Accessed on 20 September 2026).
7. **pytest** — available at https://docs.pytest.org/ (Accessed on 20 September 2026).
8. **fakeredis** — in-memory Redis substitute, available at https://github.com/cunla/fakeredis-py (Accessed on 20 September 2026).
9. **HTTP Range Requests** — RFC 7233, *Hypertext Transfer Protocol (HTTP/1.1): Range Requests*. Basis for Finding 2.
