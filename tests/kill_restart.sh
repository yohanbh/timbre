#!/usr/bin/env bash
# Phase 0 acceptance: SIGKILL at random points must still yield a
# byte-identical store. SIGKILL (-9), not SIGTERM: no cleanup handlers,
# simulating power loss / OOM-kill.
set -u
ROOT="$1"; DB="$2"; ST="$3"; KILLS="${4:-5}"; MINW="${5:-3}"; MAXW="${6:-12}"
cd /home/yohanb/timbre

run() { PYTHONPATH=src python3 -m timbre.ingest --audio-root "$ROOT" --db "$DB" --store "$ST" >/dev/null 2>&1; }
hash_store() { sha256sum "$ST" | cut -d' ' -f1; }
hash_db() { PYTHONPATH=src python3 -c "
import sys;from timbre.verify import hash_db;print(hash_db('$DB'))" 2>/dev/null | tail -1; }

rm -f "$DB"* "$ST"; run
REF_S=$(hash_store); REF_D=$(hash_db)
echo "reference store: $REF_S"
echo "reference db   : $REF_D"

rm -f "$DB"* "$ST"
for i in $(seq 1 "$KILLS"); do
  PYTHONPATH=src python3 -m timbre.ingest --audio-root "$ROOT" --db "$DB" --store "$ST" >/dev/null 2>&1 &
  PID=$!
  WAIT=$(python3 -c "import random;print(round(random.uniform($MINW,$MAXW),1))")
  sleep "$WAIT"
  if kill -9 $PID 2>/dev/null; then
    pkill -9 -P $PID 2>/dev/null
    echo "  kill $i after ${WAIT}s"
  else
    echo "  kill $i: already finished (${WAIT}s)"
  fi
  wait $PID 2>/dev/null
done

run  # final restart to completion
GOT_S=$(hash_store); GOT_D=$(hash_db)
echo "final store    : $GOT_S"
echo "final db       : $GOT_D"
[ "$GOT_S" = "$REF_S" ] && [ "$GOT_D" = "$REF_D" ] \
  && { echo "PASS: byte-identical after $KILLS kills"; exit 0; } \
  || { echo "FAIL: store differs"; exit 1; }
