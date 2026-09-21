# VoxLIT: Voice Learning & Interpretability Tool

## Master Test Plan — Speaker Verification Module

**Version 1.0**
**Component:** Speaker Verification Vertical Slice (`Backend/app/tasks/verification` & `Frontend/src/features/verification`)
**Architecture:** FastAPI, PyTorch/TorchAudio, Redis, React, TypeScript, Vitest, Pytest

---

## Revision History

| Date        | Version | Description                                                           | Author                    |
| :---------- | :-----: | :-------------------------------------------------------------------- | :------------------------ |
| 20/Sep/2026 |   1.0   | Final Master Test Plan & Unit Testing Report for Speaker Verification | Speaker Verification Lead |

---

## Table of Contents

1. [Evaluation Mission and Test Motivation](#1-evaluation-mission-and-test-motivation)
2. [Target Test Items](#2-target-test-items)
3. [Test Approach](#3-test-approach)
   - [3.1 Testing Techniques and Types](#31-testing-techniques-and-types)
     - [3.1.1 Data and Database Integrity Testing](#311-data-and-database-integrity-testing)
     - [3.1.2 Function Testing](#312-function-testing)
     - [3.1.3 User Interface Testing](#313-user-interface-testing)
     - [3.1.4 Performance Profiling](#314-performance-profiling)
     - [3.1.5 Load Testing](#315-load-testing)
     - [3.1.6 Security and Access Control Testing](#316-security-and-access-control-testing)
     - [3.1.7 Failover and Recovery Testing](#317-failover-and-recovery-testing)
     - [3.1.8 Configuration Testing](#318-configuration-testing)
4. [Deliverables](#4-deliverables)
   - [4.1 Test Evaluation Summaries](#41-test-evaluation-summaries)
   - [4.2 Reporting on Test Coverage](#42-reporting-on-test-coverage)
5. [Risks, Dependencies, Assumptions, and Constraints](#5-risks-dependencies-assumptions-and-constraints)
6. [References](#6-references)

---

## 1. Evaluation Mission and Test Motivation

### 1.1 Background & Context

VoxLIT extends audio interpretability into task-specific platforms. The **Speaker Verification** subsystem provides identity comparison, batch similarity analysis (2–100 recordings), unsupervised agglomerative clustering, temporal occlusion saliency mapping, and acoustic perturbation robustness across state-of-the-art architectures (`ecapa-tdnn` and `resnet34-lm`).

Because speaker verification operates on high-dimensional speaker embeddings and handles real-time audio analysis, testing is critical to guarantee mathematical accuracy, data isolation across user sessions, robust error handling, and low-latency interactive visualization.

### 1.2 Evaluation Mission

The primary testing objectives for the Speaker Verification module are:

1. **Verify Algorithmic Correctness:** Ensure enrollment centroid L2-normalization, pairwise cosine similarity matrix calculation, average-linkage agglomerative clustering, and separate threshold calibration behave strictly according to scientific specification.
2. **Guarantee Data Integrity & Privacy:** Enforce strict isolation between user sessions in Redis and ephemeral storage, ensuring ground-truth speaker labels are used exclusively for offline evaluation and never leak into predictions or clusters.
3. **Verify Interactive UI Components:** Ensure frontend cards (`PairComparisonCard`, `ClusterSummaryList`, `ClusterEvaluationMetricsCard`, `SpeakerSaliencyMap`) render accurate data, handle edge cases (singletons, missing batches), and preserve responsive state.
4. **Detect Defects Early:** Prevent regression and memory leaks using fast, automated test suites decoupled from GPU/download bottlenecks via deterministic mock adapters.

---

## 2. Target Test Items

The targets of test (TOT) for Speaker Verification are partitioned according to the system's modular architecture:

### 2.1 User Interfaces (Views)

- **Workbench Shell:** `SpeakerVerificationWorkbench.tsx` (toolbar, layout integration).
- **Batch Analysis & Clustering Panels:** `BatchAnalysisPanel.tsx`, `PairComparisonCard.tsx`, `ClusterSummaryList.tsx`, `ClusterEvaluationMetricsCard.tsx`.
- **Datapoint Editor Integration:** `ClusterAssignmentResults.tsx`.
- **Explainability & Attribution:** `SpeakerSaliencyMap.tsx` (waveform integration, occlusion heatmap, segment slider).
- **Frontend Utilities & Reactive Stores:** `clusterColors.ts`, `clusterAssignmentStore.ts`, `audioUrl.ts`.

### 2.2 Data Models and Storage

- **Redis In-Memory Cache:** SHA-256 content-based keys, embedding TTLs (~24h), batch similarity matrix TTLs (~6h).
- **Audio Asset Storage:** Ephemeral session uploads (`asset_`), custom recording datasets (`crec_`), and static demo audio (`voxceleb1-indian-demo`, 92 files).
- **Serialization Schemas:** Pydantic models for batch verification, clustering summaries, saliency responses, and CSV/JSON export formats.

### 2.3 Functions and Controllers (Business Logic)

- **Verification Service:** Lazy model initialization (`_load_once`), centroid calculation, cosine similarity evaluation.
- **Clustering Pipeline:** Unsupervised average-linkage hierarchical clustering, silhouette fit scores, and anonymous labeling.
- **Explainability Pipeline:** Temporal occlusion slicing (3–20 segments) and audio perturbations (additive noise, pitch shifting, time stretching, frequency/time masking).
- **FastAPI Endpoints:** `/tasks/verification/verify`, `/batch-recordings`, `/saliency`, `/robustness`, `/export`, `/session-assets`, `/custom-recordings`.

---

## 3. Test Approach

### 3.1 Testing Techniques and Types

#### 3.1.1 Data and Database Integrity Testing

| Attribute                        | Specification                                                                                                                                                                                                                                                                                                          |
| :------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Technique Objective**    | Exercise Redis caching mechanisms and audio storage independently of the UI to guarantee data integrity, deterministic key hashing, TTL expiration, and cross-session isolation.                                                                                                                                       |
| **Technique**              | • Inject known audio byte streams and verify SHA-256 cache key generation.• Validate TTL assignment: 24h for embeddings/sessions, 6h for batch matrices.• Simulate cache corruption and verify fallback to recalculation without crash.• Test transactional rollback on disk/Redis failure during asset promotion. |
| **Oracles**                | Cache hits return exact bit-for-bit serialized tensors; expired keys return`None`; corrupted entries trigger seamless recomputation; session assets cannot be resolved by unauthorized session IDs.                                                                                                                  |
| **Required Tools**         | Pytest,`pytest-asyncio`, `fakeredis`, `unittest.mock`.                                                                                                                                                                                                                                                           |
| **Success Criteria**       | All cache writes, reads, invalidations, and TTLs function with 100% test pass rate.                                                                                                                                                                                                                                    |
| **Special Considerations** | Unit tests use`fakeredis` to avoid external Redis daemon dependencies during automated execution.                                                                                                                                                                                                                    |

---

#### 3.1.2 Function Testing

| Attribute                        | Specification                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| :------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Technique Objective**    | Verify the core mathematical, algorithmic, and endpoint operations of Speaker Verification against all functional use cases.                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Technique**              | • Test enrollment centroid L2-normalization with varying reference audio counts (3–5 clips).• Verify $N \times N$ similarity matrix symmetry ($S_{ij} = S_{ji}$) and diagonal identity ($S_{ii} = 1.0$).• Execute average-linkage agglomerative clustering and test cluster fit margins and silhouette scores.• Validate temporal occlusion saliency calculation against fixed centroids.• Test perturbation engines (noise, pitch shift, time stretch, masking) and verdict delta tracking.• Validate cluster evaluation metrics against ground truth (ARI, NMI, Purity, Pairwise F1). |
| **Oracles**                | Unit assertion comparing mathematical outputs against known vector norms; HTTP status codes (200 for valid requests, 400/404/422 for boundary violations); JSON schemas match typed contracts.                                                                                                                                                                                                                                                                                                                                                                                                       |
| **Required Tools**         | Pytest, PyTorch, TorchAudio, NumPy, SciPy, Scikit-learn, HTTPX`AsyncClient`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **Success Criteria**       | All functional test cases across verification math, clustering, saliency, and endpoints pass with zero errors.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **Special Considerations** | Heavy neural network weights are isolated using mock embedding adapters to ensure unit tests run in seconds without requiring GPU hardware.                                                                                                                                                                                                                                                                                                                                                                                                                                                          |

---

#### 3.1.3 User Interface Testing

| Attribute                        | Specification                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| :------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Technique Objective**    | Verify that React UI components render correctly, react to user input, maintain reactive state, and gracefully display status/error messages.                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Technique**              | • Render UI cards within headless DOM using Vitest and React Testing Library.• Exercise selection states in `PairComparisonCard` (0, 1, 2, and invalid point selections).• Simulate cluster focus toggling in `ClusterSummaryList` and verify callback emission.• Test threshold disagreement warning banner when verification verdict and cluster membership diverge.• Test `SpeakerSaliencyMap` segment slider interactions, loading states, and error alerts.• Test `clusterAssignmentStore` subscriber lifecycle and reactive UI updating. |
| **Oracles**                | Elements exist in DOM (`toBeInTheDocument`), text formatting matches localized specifications (4-decimal floats, localized integer counts), buttons reflect active states via ARIA attributes (`aria-pressed`).                                                                                                                                                                                                                                                                                                                                          |
| **Required Tools**         | Vitest, React Testing Library,`@testing-library/jest-dom`, JSDOM, User Event.                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **Success Criteria**       | All 8 Speaker Verification component test suites pass with 100% assertions satisfied.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Special Considerations** | Web Audio API and WaveSurfer instances are mocked in JSDOM to isolate component rendering logic from audio hardware.                                                                                                                                                                                                                                                                                                                                                                                                                                         |

---

#### 3.1.4 Performance Profiling

| Attribute                        | Specification                                                                                                                                                                                                                                                 |
| :------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Technique Objective**    | Measure and verify execution times, memory usage, and UI rendering performance under standard workloads.                                                                                                                                                      |
| **Technique**              | • Benchmark batch similarity matrix calculation for 2 to 100 recordings.• Monitor memory usage (RSS) during embedding extraction and matrix generation using `psutil`.• Benchmark frontend test suite execution to maintain rapid CI/CD feedback cycles. |
| **Oracles**                | Batch processing for test matrices executes in$< 0.1$s; memory growth remains strictly below threshold ($< 500$MB delta); unit test suite executes in under 45 seconds.                                                                                   |
| **Required Tools**         | Python`time`, `psutil`, Vitest benchmarking timer.                                                                                                                                                                                                        |
| **Success Criteria**       | All performance benchmarks are met without memory leaks or uncollected garbage.                                                                                                                                                                               |
| **Special Considerations** | Benchmarking is performed on standard CPU hardware to reflect baseline server constraints.                                                                                                                                                                    |

---

#### 3.1.5 Load Testing

| Attribute                        | Specification                                                                                                                                                                                                                         |
| :------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Technique Objective**    | Evaluate system behavior and cache throughput under high numbers of concurrent requests.                                                                                                                                              |
| **Technique**              | • Execute concurrent async cache read/write operations using`asyncio.gather`.• Submit multiple concurrent requests to verification endpoints.• Fill cache with high-frequency keys to verify LRU eviction under memory pressure. |
| **Oracles**                | Success rate$\ge 90\%$; average cache response time $< 0.1$s; zero race conditions or data corruption.                                                                                                                            |
| **Required Tools**         | Pytest-asyncio, concurrent workers simulation, Redis pipeline mocks.                                                                                                                                                                  |
| **Success Criteria**       | No thread deadlocks, race conditions, or unhandled exceptions under concurrent access.                                                                                                                                                |
| **Special Considerations** | Simulated via asynchronous coroutines to validate concurrency primitives without network instability.                                                                                                                                 |

---

#### 3.1.6 Security and Access Control Testing

| Attribute                        | Specification                                                                                                                                                                                                                                                                                                                                                                                                                |
| :------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Technique Objective**    | Guarantee protection against unauthorized access, cross-session data leakage (IDOR), and malicious file uploads.                                                                                                                                                                                                                                                                                                             |
| **Technique**              | • Attempt directory traversal attacks using encoded path segments (`..%2f..%2f`, Windows backslashes).• Attempt cross-session audio access (requesting Session A's asset ID using Session B's cookie).• Attempt uploading non-audio payloads (executable binaries, PHP scripts) and oversized files.• Verify that sensitive filesystem paths and ground-truth speaker names are never returned in public API payloads. |
| **Oracles**                | Directory traversal attempts are rejected with 400 or uniform 404; cross-session requests return 404 Not Found; malicious files return 400 Bad Request; server paths and ground-truth IDs never appear in JSON responses.                                                                                                                                                                                                    |
| **Required Tools**         | HTTPX`AsyncClient`, Pytest security test fixtures.                                                                                                                                                                                                                                                                                                                                                                         |
| **Success Criteria**       | All security assertions pass; zero IDOR or path traversal vulnerabilities detected.                                                                                                                                                                                                                                                                                                                                          |
| **Special Considerations** | Session cookies are validated strictly against 32-character hexadecimal format.                                                                                                                                                                                                                                                                                                                                              |

---

#### 3.1.7 Failover and Recovery Testing

| Attribute                        | Specification                                                                                                                                                                                                                                                                 |
| :------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Technique Objective**    | Ensure the verification pipeline handles subsystem failures gracefully without crashing.                                                                                                                                                                                      |
| **Technique**              | • Simulate Redis connection failure during batch operations and verify fallback/503 handling.• Inject corrupted audio files and verify transactional rollback leaving no orphaned files.• Test corrupted cache entry handling to ensure clean recalculation miss behavior. |
| **Oracles**                | Redis pipeline write failure yields an explicit 503 error; corrupt audio returns 400/422 and unlinks temporary files; system logs error without process crash.                                                                                                                |
| **Required Tools**         | Pytest, unittest mock side effects.                                                                                                                                                                                                                                           |
| **Success Criteria**       | System fails closed safely and recovers automatically on subsequent valid requests.                                                                                                                                                                                           |
| **Special Considerations** | Failover tests ensure temporary disk assets are always purged via`finally` blocks.                                                                                                                                                                                          |

---

#### 3.1.8 Configuration Testing

| Attribute                        | Specification                                                                                                                                                                                                                                                   |
| :------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Technique Objective**    | Validate that the verification subsystem functions portably across operating systems (Windows and Linux).                                                                                                                                                       |
| **Technique**              | • Verify filesystem path construction using`pathlib.Path` derived from `Backend` root.• Validate cross-platform pitch shifting without relying on Unix-specific `signal.SIGALRM`.• Verify TypeScript build and frontend bundle compilation on Node.js. |
| **Oracles**                | All paths resolve regardless of Windows (`\`) or POSIX (`/`) separators; audio perturbation timeout triggers safely across platforms; `npm run build` exits with code 0.                                                                                  |
| **Required Tools**         | Python`platform`, Node.js, Vite build compiler.                                                                                                                                                                                                               |
| **Success Criteria**       | 100% build and test compatibility on Windows and Linux targets.                                                                                                                                                                                                 |
| **Special Considerations** | Model weights are stored under`Backend/pretrained_models/speaker_verification/` with paths created lazily.                                                                                                                                                    |

---

## 4. Deliverables

### 4.1 Test Evaluation Summaries

#### Backend Pytest Execution Summary

- **Command:** `.venv\Scripts\python -m pytest -k "verification"`
- **Execution Output:**

```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-8.2.0, pluggy-1.6.0
rootdir: D:\DSE project\VoxLIT\Backend
configfile: pytest.ini
plugins: anyio-4.4.0, asyncio-0.23.7
asyncio: mode=Mode.AUTO

collected 387 items

tests/test_speaker_verification.py ......................... [  1%]
tests/test_speaker_verification_audio_streaming.py ...       [  2%]
tests/test_speaker_verification_batch.py ................... [  9%]
tests/test_speaker_verification_cache.py ................... [ 20%]
tests/test_speaker_verification_clustering.py .............. [ 29%]
tests/test_speaker_verification_dataset.py ................. [ 39%]
tests/test_speaker_verification_evaluation_metrics.py ...... [ 41%]
tests/test_speaker_verification_export.py .................. [ 47%]
tests/test_speaker_verification_model_storage.py .....       [ 48%]
tests/test_speaker_verification_projection.py .............. [ 60%]
tests/test_speaker_verification_robustness.py .............. [ 78%]
tests/test_speaker_verification_saliency.py ................ [ 86%]
tests/test_speaker_verification_session_assets.py .......... [ 92%]
tests/test_speaker_verification_temporal_occlusion.py ...... [ 94%]
tests/test_verification_custom_recordings.py ............... [100%]

============= 387 passed, 315 deselected, 230 warnings in 37.83s ==============
```

#### Frontend Vitest Execution Summary

- **Command:** `npm test`
- **Execution Output:**

```
 RUN  v2.1.9 D:/DSE project/VoxLIT/Frontend

 ✓ src/features/verification/__tests__/audioUrl.test.ts (4 tests) 11ms
 ✓ src/features/verification/__tests__/clusterAssignmentStore.test.ts (4 tests) 13ms
 ✓ src/features/verification/__tests__/clusterColors.test.ts (5 tests) 19ms
 ✓ src/features/verification/__tests__/ClusterEvaluationMetricsCard.test.tsx (4 tests) 328ms
 ✓ src/features/verification/__tests__/SpeakerSaliencyMap.test.tsx (6 tests) 495ms
 ✓ src/features/verification/__tests__/ClusterAssignmentResults.test.tsx (5 tests) 273ms
 ✓ src/features/verification/__tests__/PairComparisonCard.test.tsx (6 tests) 336ms
 ✓ src/features/verification/__tests__/ClusterSummaryList.test.tsx (5 tests) 575ms
 ✓ src/tests/ui-components.test.tsx (21 tests) 3943ms

 Test Files  9 passed (9)
      Tests  60 passed (60)
   Duration  14.30s
```

#### Production Build Inspection

- **Command:** `npm run build`
- **Status:** Successful (0 TypeScript or Rollup errors).

```
✓ 2598 modules transformed.
dist/index.html                     0.94 kB
dist/assets/index-B0tYqhxD.css     79.05 kB
dist/assets/index-Ba2hLzOP.js   5,962.66 kB
✓ built in 38.23s
```

---

### 4.2 Reporting on Test Coverage

| Test Module / Suite                                 |     Layer     | Test Items Covered                                             |     Pass     |    Fail    |   Pass Rate   |
| :-------------------------------------------------- | :-----------: | :------------------------------------------------------------- | :-----------: | :---------: | :------------: |
| `test_speaker_verification.py`                    |    Backend    | Centroid L2 norm, model registry, baseline exclusion           |       7       |      0      |      100%      |
| `test_speaker_verification_audio_streaming.py`    |    Backend    | Byte-range streaming (HTTP 206), header parity                 |       3       |      0      |      100%      |
| `test_speaker_verification_batch.py`              |    Backend    | $N \times N$ matrix, symmetry, diagonal identity, thresholds |      27      |      0      |      100%      |
| `test_speaker_verification_cache.py`              |    Backend    | Redis SHA-256 keys, TTL expiration, corrupt recovery           |      41      |      0      |      100%      |
| `test_speaker_verification_clustering.py`         |    Backend    | Agglomerative clustering, silhouette scores, anonymous IDs     |      34      |      0      |      100%      |
| `test_speaker_verification_dataset.py`            |    Backend    | VoxCeleb demo (92 clips), safe recording IDs                   |      38      |      0      |      100%      |
| `test_speaker_verification_evaluation_metrics.py` |    Backend    | Ground-truth metrics: ARI, NMI, Purity, Pairwise F1            |       8       |      0      |      100%      |
| `test_speaker_verification_export.py`             |    Backend    | CSV and JSON batch exports, matrix formatting                  |      24      |      0      |      100%      |
| `test_speaker_verification_model_storage.py`      |    Backend    | Lazy loading (`_load_once`), cache isolation                 |       5       |      0      |      100%      |
| `test_speaker_verification_projection.py`         |    Backend    | PCA, t-SNE, UMAP 2D/3D projections, small batch handling       |      44      |      0      |      100%      |
| `test_speaker_verification_robustness.py`         |    Backend    | Noise, pitch shifting, time stretch, masking, verdict deltas   |      70      |      0      |      100%      |
| `test_speaker_verification_saliency.py`           |    Backend    | Temporal occlusion, segment attribution, ranking               |      30      |      0      |      100%      |
| `test_speaker_verification_session_assets.py`     |    Backend    | Uploaded asset lifecycle, TTL renewal, sweeps, rollbacks       |      24      |      0      |      100%      |
| `test_speaker_verification_temporal_occlusion.py` |    Backend    | Occlusion API route validation, segment count bounds           |       8       |      0      |      100%      |
| `test_verification_custom_recordings.py`          |    Backend    | `crec_` IDs, custom datasets, Redis 503 resilience           |      24      |      0      |      100%      |
| `clusterColors.test.ts`                           |   Frontend   | Base palette, golden-angle HSL, determinism                    |       5       |      0      |      100%      |
| `clusterAssignmentStore.test.ts`                  |   Frontend   | Module store subscribe, publish, unsubscribe                   |       4       |      0      |      100%      |
| `audioUrl.test.ts`                                |   Frontend   | Audio URL resolution (`asset_`, `crec_`, `rec_`)         |       4       |      0      |      100%      |
| `PairComparisonCard.test.tsx`                     |   Frontend   | Pair metrics, margin formatting, disagreement alerts           |       6       |      0      |      100%      |
| `ClusterEvaluationMetricsCard.test.tsx`           |   Frontend   | ARI/NMI/Purity display, pairwise rates, contingency counts     |       4       |      0      |      100%      |
| `ClusterSummaryList.test.tsx`                     |   Frontend   | Cluster sorting, member counts, interactive focus toggling     |       5       |      0      |      100%      |
| `ClusterAssignmentResults.test.tsx`               |   Frontend   | Selected recording stats, nearest clip badge                   |       5       |      0      |      100%      |
| `SpeakerSaliencyMap.test.tsx`                     |   Frontend   | Heatmap rendering, segment slider, loading/error states        |       6       |      0      |      100%      |
| `ui-components.test.tsx`                          |   Frontend   | Audio controls, drag-and-drop, sliders, tabs                   |      21      |      0      |      100%      |
| **Total Test Suite**                          | **All** | **Speaker Verification Vertical Slice**                  | **447** | **0** | **100%** |

---

## 5. Risks, Dependencies, Assumptions, and Constraints

| Risk                                                                          | Mitigation Strategy                                                                                         | Contingency (Risk is realized)                                                |
| :---------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------- |
| **Neural network model weights cause slow test execution.**             | Use lightweight mock adapters in unit tests; load real models only in dedicated integration tests.          | Mock adapter bypasses weight download; tests run in$< 45$s.                 |
| **Redis daemon unavailable in local testing environment.**              | Employ`fakeredis` in pytest test harness to emulate Redis operations purely in-memory.                    | Tests execute without requiring external Redis container.                     |
| **Ground-truth labels inadvertently influence clustering predictions.** | Strictly isolate ground-truth metadata; unit tests assert that predictions rely only on pairwise distances. | Reject pull request if ground-truth IDs appear in clustering input arguments. |
| **Audio perturbation timeout hangs on Windows machines.**               | Replace Unix-specific`signal.SIGALRM` with cross-platform Python thread worker timeouts.                  | Perturbation tests fail gracefully on timeout without hanging test runner.    |
| **Web Audio API not implemented in headless JSDOM.**                    | Mock`AudioContext`, `WaveSurfer`, and `ResizeObserver` in `Frontend/src/test/setup.ts`.             | UI component tests evaluate DOM and state without audio driver dependencies.  |

---

## 6. References

1. **IEEE Standard for Software and System Test Documentation:** IEEE Std 829-2008.
2. **Rational Unified Process (RUP):** Test Plan Guidelines and Artifact Templates.
3. **SpeechBrain (ECAPA-TDNN):** Ravanelli, M., et al. "SpeechBrain: A General-Purpose Speech Toolkit." *arXiv:2106.04624*, 2021.
4. **ResNet for Speaker Verification:** Chung, J. S., et al. "In defence of metric learning for speaker recognition." *Interspeech*, 2020.
5. **FastAPI Framework:** Tiangolo, S. FastAPI Documentation, Available at: `https://fastapi.tiangolo.com/` (Accessed: September 2026).
6. **Pytest Testing Framework:** Pytest Development Team. Available at: `https://docs.pytest.org/` (Accessed: September 2026).
7. **Vitest Unit Test Framework:** Vitest Team. Available at: `https://vitest.dev/` (Accessed: September 2026).
8. **React Testing Library:** Testing Library Team. Available at: `https://testing-library.com/docs/react-testing-library/intro/` (Accessed: September 2026).
