# Audio Deepfake Detection — Model Evaluation

Offline evaluation and error analysis for the Audio Deepfake Detection task. Sibling of
`../speaker_verification_evaluation/`, and the source of every DS number in
`Local_Files/DEEPFAKE_TEST_PLAN.md`.

**Owner:** Chanupa Gurusinghe · **Measured:** 20 Sep 2026, macOS (Apple Silicon), CPU only.

## What makes this different from a normal eval script

It does not reimplement the model. `score_dataset.py` imports the running
application's own detector (`app.tasks.deepfake.service`), so the preprocessing, the
analysis window, the spoof-class resolution and the threshold are the ones the API
uses. The EER and DET curve are then computed by the app's own
`app.tasks.deepfake.metrics`, not by scikit-learn.

That is deliberate: if this folder computed its own answer, agreement with the app
would prove nothing. As written, a disagreement between
`POST /tasks/deepfake/scores` and `outputs/metrics/*.json` is a real defect — and
`score_dataset.py --verify` cross-checks a sample against `service.run_detection` on
every run, failing loudly if the single-pass shortcut it uses ever drifts.

## Dataset

`Backend/data/deepfake/asvspoof2019_la/` — a 200-clip balanced subset of the ASVspoof
2019 Logical Access evaluation set, built by
`Backend/scripts/prepare_asvspoof_la_subset.py`: 100 bona fide clips and 100 spoofed
ones spread across all 13 attacks, A07–A19.

`protocol.txt` holds the bona fide/spoof answers. It is read **here only**. The runtime
API never returns a per-clip label — the workbench's exercise is to read the score
first and check the protocol afterwards — so the labels in `outputs/` must not travel
back into the app.

## Models

| Key | Checkpoint | Tier | Status |
|---|---|---|---|
| `xlsr-deepfake` | `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification` | A | measured |
| `xlsr-mamba` | `AustinXiao/XLSR-Mamba-LA` | B (our own architecture code) | measured |
| `ast-fakeaudio` | `WpythonW/ast-fakeaudio-detector` | A | **blocked** — gated repo, needs `HF_TOKEN` in `Backend/.env` |

## Pipeline

```text
protocol.txt + flac/          Backend/app/tasks/deepfake/service.py
        |                                    |
        +-------------- score_dataset.py ----+
                            |
              outputs/scores/{model}_scores.csv      (one row per clip)
              outputs/embeddings/{model}.npz         (head-input vectors, gitignored)
                            |
        +-------------------+--------------------+
        |                                        |
  evaluate_model.py                       error_analysis.py
        |                                        |
  outputs/metrics/                        outputs/error_analysis/
    {model}_metrics.json                    {model}_error_analysis.json
    {model}_det_curve.csv                   {model}_failures.csv
                                            {model}_silence_ablation.csv
                            |
                     compare_models.py
                            |
              outputs/error_analysis/{a}_vs_{b}.json
```

## Running it

From this folder, with the backend's virtualenv:

```bash
PYTHONPATH=../../Backend ../../Backend/.venv/bin/python src/score_dataset.py --model xlsr-deepfake
```

```bash
PYTHONPATH=../../Backend ../../Backend/.venv/bin/python src/evaluate_model.py --model xlsr-deepfake
```

```bash
PYTHONPATH=../../Backend ../../Backend/.venv/bin/python src/error_analysis.py --model xlsr-deepfake --probe-sample 10
```

```bash
PYTHONPATH=../../Backend ../../Backend/.venv/bin/python src/compare_models.py --models xlsr-deepfake xlsr-mamba
```

`score_dataset.py` is resumable — clips already in the CSV are skipped — so an
interrupted run costs nothing.

## Headline results

| | Model A (wav2vec2 XLS-R) | Model C (XLSR-Mamba) |
|---|---|---|
| EER on the subset | **0.00 %** | **0.00 %** |
| EER threshold | 0.142884 | 0.017944 |
| ROC-AUC | 1.000 | 1.000 |
| Errors at the shipped 0.5 cut | 2 (both A19) | 1 (A18) |
| Weakest attack | A19, mean 0.594 | A18, mean 0.803 |
| Analysis window | 30.0 s | 4.17 s (41 clips truncated) |
| Scoring cost | 0.167 s/clip | 0.254 s/clip |

Both detectors separate this subset perfectly. **That result does not survive the
silence ablation.** Trimming the leading and trailing silence from ten genuine clips
flips 5 of 10 to "spoof" under Model A and **10 of 10** under Model C, while every
spoofed clip stays spoofed. The genuine clips in this corpus carry 51 % non-speech
against the spoofed clips' 23 %, and the detectors are reading that gap.

The full write-up, with the per-clip numbers, is in
`Local_Files/DEEPFAKE_TEST_PLAN.md` §5.
