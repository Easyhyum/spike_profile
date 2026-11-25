#!/usr/bin/env bash
# ramp_workloads.sh — is the ~200 us current ramp at a kernel start a property of the workload or of the 12 V delivery path?
#   Same conditions as the campaign (all-SM, warm clock 500 ms + 20 ms gap, current+voltage on the HDO, 1 us), 3 cycles each:
#     cublas_bf16   cuBLAS bf16 GEMM 4096x12288x4096 (the ws shape)           - compute + loads
#     ws_normal     wsgemm bf16 (producer warps load, consumer warps mma)     - compute + loads
#     ws_noload     wsgemm with the global-memory loads disabled              - compute only (no DRAM traffic after the first tiles)
#     ws_loadonly   wsgemm with the consumers idle (free-running loader)      - memory traffic only, almost no compute
#   Then hdo_capture/ramp_analysis.py on the four runs. Run: sudo env FREQMON=0 PREWARM_GAP_MS=20 PREWARM_MS=500 ./ramp_workloads.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -z "${HW_JSON:-}" ]] && hw_refresh
require gpu pmd scope v_ch_ok i_ch_ok || exit 0
: "${RAMP_CYCLES:=3}"
common="-b $RAMP_CYCLES --dtype bf16 --rows 4096 --size 12288 --cols 4096 --launches 1 -o 1 -d $OFF_MS --smi-ms $SMI_MS --idle-before 3 --no-freqmon${PREWARM_GAP_MS:+ --prewarm-gap-ms $PREWARM_GAP_MS}${PREWARM_MS:+ --prewarm-ms $PREWARM_MS} --scope-sample-ns ${SCOPE_SAMPLE_NS:-1000}"
sargs="$(scope_args | sed 's/--scope-sample-ns [0-9.]*//') $(trig_args)"
run_eval ramp_cublas_bf16 $common --kernel cublas $sargs
run_eval ramp_ws_normal   $common --kernel ws $sargs
run_eval ramp_ws_noload   $common --kernel ws --ws-noload $sargs
run_eval ramp_ws_loadonly $common --kernel ws --ws-nocompute 1 $sargs
"$PY" "$HERE/summary_runs.py" "$RESULT_DIR" > /dev/null || true
"$PY" "$ROOT/hdo_capture/ramp_analysis.py" "$RESULT_DIR"/spike_eval_*ramp_* | tee "$RESULT_DIR/ramp_analysis.txt"
