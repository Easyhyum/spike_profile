#!/usr/bin/env bash
# 02 — cycle-length sweep: the same GEMM with M = AUTO_DURATION_ROWS (~1/16, 1/4, 1/1 of the max shape -> short / mid / long
#      single cycle) plus bursts of BURSTS back-to-back launches of the max shape (default 2 and 5: ~125 ms and ~315 ms of
#      load, long enough to see whether the power limiter pulls the clock / power down below the 180 W cap level the PMD
#      shows at ~200 W). Shows how the di/dt drop (unchanged), the settled droop and the DVFS clock step depend on how long
#      the load lasts. CYCLES cycles per point.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -z "${HW_JSON:-}" ]] && hw_refresh
require gpu pmd scope v_ch_ok ${REQUIRE_I:+i_ch_ok} || exit 0   # REQUIRE_I=1 (current campaign): skip unless the current probe is live
: "${DURATION_ROWS=${AUTO_DURATION_ROWS:-$SIZE}}"   # unset -> default rows; DURATION_ROWS= (empty) -> no row sweep
for rows in $DURATION_ROWS; do
  run_eval "t02_dur_rows${rows}" -b "$CYCLES" $(workload_args) --rows "$rows" $(scope_args) $(trig_args)
done
: "${BURSTS=2 5}"                                   # BURSTS= (empty) -> no bursts; e.g. DURATION_ROWS= BURSTS=5 ./run_all.sh 02
for nb in $BURSTS; do
  run_eval "t02_dur_rows${SIZE}_x${nb}" -b "$CYCLES" $(workload_args | sed "s/--launches 1/--launches $nb/") $(scope_args) $(trig_args)
done
"$PY" "$HERE/summary_runs.py" "$RESULT_DIR" || true
