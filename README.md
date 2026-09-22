# jev-test

A small test harness for [TypeSafe's Jev](https://docs.typesafe.ai/introduction), a "System One" model:
you send it a `state` plus typed `questions` (Choice / Score / Noul) and get back typed answers with
confidence scores instead of generated text.

This harness runs batches of test cases through Jev's `/v1/systemone` endpoint, logs the raw request/response
for every call, and computes basic stats (accuracy where an expected answer is known, confidence distribution,
calibration buckets) exportable to CSV/JSON for charting.

Background and results: [Stress-Testing Jev's Confidence](https://claude.ai/artifact/f4ae2d48-9cad-49cf-8c24-55cb6f2c3e6f).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in your TYPESAFE_API_KEY
```

## Usage

```bash
set -a && source .env && set +a
.venv/bin/python run.py testcases/off_menu.yaml
.venv/bin/python run.py testcases/calibration.yaml --model jev-1.13.0
```

Each run writes to `results/<batch>/<timestamp>/`:

- `raw_responses.jsonl` — the literal API request/response body for every call, one line per case
- `results.csv` / `results.json` — flattened per-question rows (case, question, type, value, confidence,
  expected, correct)
- `summary.json` — accuracy, confidence histogram, and confidence-bucketed calibration stats

## Project layout

```
jev_harness/
  client.py         wraps typesafe-sdk, always logs the raw HTTP response body
  testcases.py       loads YAML/JSON batch files into Batch/Case objects
  runner.py          runs a batch, writes raw log + CSV/JSON + summary
  stats.py           grading and aggregate stats (accuracy, calibration buckets)
run.py                CLI: python run.py <batch.yaml> [--model] [--out-dir]
testcases/
  off_menu.yaml       schema questions where none of the options fit, with/without an escape option
  ambiguous.yaml       tickets deliberately written to sit on a tone/urgency boundary
  calibration.yaml     90 synthetic cases with known-correct labels (generated, see below)
  generate_calibration.py   regenerates calibration.yaml deterministically (seed=20260921)
```

## Test-case format

A batch file is YAML (or JSON): a `state`, one or more typed `questions` sent to the API as-is, and an
optional `expected` map (harness-only bookkeeping — never sent to Jev):

```yaml
batch: example
cases:
  - id: case_01
    state: "Customer message here."
    tags: [example]
    questions:
      department:
        type: choice
        instructions: "Which team should handle this?"
        criteria:
          billing: "Payment or subscription issues"
          technical: "Bugs or integration problems"
          shipping: "Delivery status, delays, lost packages"
    expected:
      department: billing
```

## Notes

- Every run pins a model version (`--model jev-1.13.0`) rather than the `jev-latest` alias for
  reproducibility; omit it to use the batch file's `model:` field or the SDK default.
- `results/` and `.env` are gitignored — nothing in this repo requires secrets to read.
