# Test data

Isolated inputs for evaluating every resume against the complete recent-job set.

- `resumes/`: five normalized text resumes plus a source manifest.
- `jobs/`: 250 JSON jobs, preserving every field from the source CSV, plus a source manifest.
- `results/`: destination for timestamped metric CSV or Parquet files.

Run the all-pairs metric harness from the repository root:

```powershell
python webapp/scripts/run_test_ranking_harness.py
```

This uses the centralized pinned MiniLM loader, creates/reuses ignored job-side
embeddings under `generated/embeddings/`, keeps every candidate through every
local metric, and writes a timestamped CSV under `results/`. Add
`--disable-cross-encoder` for a faster run that omits that local metric. The
paid LLM filter is always disabled by this harness.
Generated embeddings and routine result files are intentionally gitignored.

Metric rows are flushed to an append-only JSONL checkpoint after each resume.
If a run is interrupted, rerunning the same command detects completed resumes
and continues with the first unfinished one. After all pairings are present,
the harness atomically creates the final CSV and archives the JSONL checkpoint.

To enrich an already completed result with whole-resume versus
title/requirements cosine distance, plus consistently recomputed Mahalanobis
and multi-metric bad-fit values, run:

```powershell
python webapp/scripts/run_test_ranking_harness.py --backfill-requirements-metric
```

This reuses the latest completed JSONL and cached job embeddings. It embeds
each resume once and does not rerun the expensive source metrics. Use
`--source-result PATH` to select a specific completed JSONL.

The original files under `harness_data/` are unchanged.
