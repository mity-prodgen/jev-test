#!/usr/bin/env bash
# Fetches the datasets for the RAG tests into data/ (gitignored).
#   SQuAD v1.1 dev  (CC BY-SA 4.0)  https://rajpurkar.github.io/SQuAD-explorer/
#   BEIR SciFact    (BEIR packaging says CC BY-SA 4.0; the original allenai/scifact card says
#                    CC BY-NC 2.0 - treat as non-commercial; we do not redistribute the data)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/data/squad" "$ROOT/data/scifact"
curl -sL -o "$ROOT/data/squad/dev-v1.1.json" "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
curl -sL -o "$ROOT/data/scifact/scifact.zip" "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
unzip -oq "$ROOT/data/scifact/scifact.zip" -d "$ROOT/data/scifact"
ls -la "$ROOT/data/squad" "$ROOT/data/scifact/scifact"
