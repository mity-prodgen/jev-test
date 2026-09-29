#!/usr/bin/env bash
# Fetches the two RAGTruth dataset files + its LICENSE into data/ragtruth/.
# RAGTruth (MIT licensed): https://github.com/ParticleMedia/RAGTruth
#
# Not committed to this repo - data/ is gitignored. The underlying passage
# text (CNN/DM, MS MARCO, Yelp) carries its own terms beyond RAGTruth's MIT
# license on the annotations/code, so we regenerate our sample deterministically
# from a seed (hallucination/build_sample.py) rather than committing derived text.
set -euo pipefail

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/ragtruth"
mkdir -p "$DEST"

BASE="https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main"
curl -sL -o "$DEST/response.jsonl" "$BASE/dataset/response.jsonl"
curl -sL -o "$DEST/source_info.jsonl" "$BASE/dataset/source_info.jsonl"
curl -sL -o "$DEST/LICENSE" "$BASE/LICENSE"

echo "Downloaded to $DEST:"
wc -l "$DEST/response.jsonl" "$DEST/source_info.jsonl"
