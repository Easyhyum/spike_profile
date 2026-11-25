#!/usr/bin/env bash
# 01 — baseline: CYCLES single cycles of the strongest-spike GEMM (auto shape) with PMD2X + nvidia-smi + freqmon + scope
#      (both voltage channels, or V x I when a current probe is live) on the same launch.
#      Metrics (summary_runs.py): di/dt drop in the first 300 us, settled droop, whole-cycle minimum, PMD peak W / A,
#      kernel clock, and the lane comparison CH(V) vs CH(V2) (idle / load / max instantaneous difference).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -z "${HW_JSON:-}" ]] && hw_refresh
require gpu pmd scope v_ch_ok ${REQUIRE_I:+i_ch_ok} || exit 0   # REQUIRE_I=1 (current campaign): skip unless the current probe is live
run_eval t01_base -b "$CYCLES" $(workload_args) $(scope_args) $(trig_args)
"$PY" "$HERE/summary_runs.py" "$RESULT_DIR" || true
