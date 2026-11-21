#!/bin/sh
# Run the teacher in short chunks. Each chunk takes the GPU lock on its own,
# so other training jobs get a turn between chunks instead of waiting hours.
#   sh train/experts/teach_loop.sh [total=1400] [chunk=120]
cd "$(dirname "$0")/../.." || exit 1
TOTAL=${1:-1400}
CHUNK=${2:-120}
touch data/experts/teacher.jsonl
while [ "$(wc -l < data/experts/teacher.jsonl)" -lt "$TOTAL" ]; do
    flock /tmp/veris-gpu.lock .venv/bin/python -m train.experts.build teach --limit "$CHUNK" || sleep 30
    sleep 5
done
echo TEACH_DONE
