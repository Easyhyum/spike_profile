#!/usr/bin/env bash
# run_all.sh — the whole sequence in one result folder, then report.md / report.json.
#   sudo ./run_all.sh            # 00 preflight (+ shape probe), 01 baseline, 02 duration, 03 clock (UAC prompt once), 04 wave, report
#   sudo ./run_all.sh 01 04      # only those tests (+ report)
# Every test skips itself when its hardware requirement is missing. A current probe (I_CH) makes every test V x I automatically.
# RESUME_DIR=<existing results folder> continues an interrupted sequence in that folder (pass the tests still missing, e.g.
#   sudo env RESUME_DIR=results/20261123_124149_run_all_pl200_warm20_20ns FREQMON=1 POWER_LIMIT_W=200 PREWARM_GAP_MS=20 SCOPE_SAMPLE_NS=20 DURATION_ROWS= BURSTS=5 ./run_all.sh 02 03 04
# ); the environment must repeat the original settings (they are in the folder's sequence.log header and config.env.used).
if [[ -n "${RESUME_DIR:-}" ]]; then
  [[ -d "$RESUME_DIR" ]] || { echo "RESUME_DIR $RESUME_DIR does not exist"; exit 1; }
  export RESULT_DIR="$(cd "$RESUME_DIR" && pwd)"
else
  export RESULT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results/$(date +%Y%m%d_%H%M%S)_run_all$([[ "${FREQMON:-1}" == "0" ]] && echo _allsm)${POWER_LIMIT_W:+_pl${POWER_LIMIT_W}}${PREWARM_GAP_MS:+_warm${PREWARM_GAP_MS}}${PREWARM_MS:+x${PREWARM_MS}}$([[ -n "${SCOPE_SAMPLE_NS:-}" && "${SCOPE_SAMPLE_NS}" != "1000" ]] && echo _${SCOPE_SAMPLE_NS}ns)"
fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
log "===== spike test sequence $STAMP${RESUME_DIR:+ (RESUMED)} -> $RESULT_DIR (FREQMON=${FREQMON:-1}: $([[ "${FREQMON:-1}" == "0" ]] && echo "no clock monitor, all SMs run the GEMM" || echo "freqmon holds one SM")) ====="
"$HERE/00_preflight.sh" || true
source "$HERE/logs/auto_shape.env"; load_shape
hw_refresh; export HW_JSON
CC="$PY $HERE/clockctl.py"
$CC limits > "$RESULT_DIR/power_limits.json" 2>/dev/null || true      # limits in force (re-written after POWER_LIMIT_W is applied)
if [[ -n "${POWER_LIMIT_W:-}" ]]; then                     # raise/lower the board power limit for the whole sequence
  log "power limit: $($CC limits) -> requesting ${POWER_LIMIT_W} W (one UAC prompt)"
  if ! $CC start 240 || ! $CC pl "$POWER_LIMIT_W"; then log "ABORT: power limit ${POWER_LIMIT_W} W could not be applied"; exit 1; fi
  export KEEP_CLOCK_SERVER=1                                 # 03_clock.sh must not stop the server (its exit restores the default limit)
  trap '$CC plreset || true; $CC stop || true' EXIT
  log "power limit now: $($CC limits)"; $CC limits > "$RESULT_DIR/power_limits.json" 2>/dev/null || true
fi
sel=("$@"); [[ ${#sel[@]} -eq 0 ]] && sel=(01 02 03 04)
for id in "${sel[@]}"; do
  f="$(ls "$HERE"/${id}_*.sh 2>/dev/null | head -1)"
  [[ -n "$f" ]] || { log "no test $id"; continue; }
  log "----- $(basename "$f") -----"
  HW_JSON="$HW_JSON" RESULT_DIR="$RESULT_DIR" DTYPE="$DTYPE" SIZE="$SIZE" bash "$f" || log "FAILED $(basename "$f") (exit $?)"
done
cp "$HERE/config.env" "$RESULT_DIR/config.env.used"
"$PY" "$HERE/make_report.py" "$RESULT_DIR" || true
chown -R "${SUDO_USER:-$USER}":"${SUDO_USER:-$USER}" "$RESULT_DIR" 2>/dev/null || true
if [[ -n "${POWER_LIMIT_W:-}" ]]; then $CC plreset || true; $CC stop || true; trap - EXIT; log "power limit restored: $($CC limits)"; fi
log "===== sequence finished; report: $RESULT_DIR/report.md ====="
