#!/usr/bin/env bash
# 04 — di/dt drop vs the number of waves. wavegemm.cu (fp32, one 256x128 tile per CTA, 1 CTA/SM, same blocking as the
#      cuBLAS SGEMM, 80-90 % of its throughput) gives an exact grid: waves = (M/256)(N/128) / usable SMs, with usable SMs
#      = device SMs - 1 (freqmon). wave_targets.py turns WAVES into (CTAs, M, N) on any GPU; the top target is the max
#      shape (M=N=SIZE) itself. WAVE_CYCLES single cycles per point.
#      WAVE_K (default SIZE): list of K values -> the same wave targets at several kernel lengths (K scales the per-CTA work and
#      so the cycle length; this replaces the separate duration test). WAVE_REF=1 (default) adds the cuBLAS max shape
#      (M=N=K=SIZE, the old 01 baseline) as a reference point in the same table, tag t04_cublas_*.
#      Tags: t04_K<k>_M<m>_N<n>_w<w>_<cta>cta. Outputs: wave_sweep_summary.csv/png (one curve per K), per K:
#      overlay_04_waves_K<k>_CH1_CH2.png (1 wave in red), overlay_04_waves_K<k>_freqmon.png, overlay_04_waves_K<k>_combined.png
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -z "${HW_JSON:-}" ]] && hw_refresh
require gpu pmd scope v_ch_ok ${REQUIRE_I:+i_ch_ok} || exit 0   # REQUIRE_I=1 (current campaign): skip unless the current probe is live
USABLE="${AUTO_USABLE_SMS:-35}"
TOP="$("$PY" -c "import sys; print(round(($SIZE//256)*($SIZE//128)/$USABLE, 2))")"
: "${WAVES:=0.5 0.6 0.7 0.8 0.9 1.0 1.5 1.6 1.7 1.8 1.9 2.0 3 4 8 16 32 $TOP}"
: "${WAVE_K:=$SIZE}"; : "${WAVE_REF:=1}"
log "04: wave targets: $WAVES x K in {$WAVE_K} ($WAVE_CYCLES single cycles each, $USABLE usable SMs$([[ "${FREQMON:-1}" == "0" ]] && echo " = all SMs, no freqmon"))$([[ "$WAVE_REF" == "1" ]] && echo "; + cuBLAS reference M=N=K=$SIZE")"
: "${SKIP_DONE:=1}"                                   # RESUME: a target whose complete run (summary.json) is already in RESULT_DIR is skipped
done_already() {   # complete = summary.json present AND one scope record per requested cycle (a scope drop leaves fewer)
  local r; for r in "$RESULT_DIR"/spike_eval_*_"$1"; do [[ -f "$r/summary.json" && $(ls "$r"/burst_[0-9][0-9].npz 2>/dev/null | wc -l) -ge "$WAVE_CYCLES" ]] && return 0; done; return 1; }
if [[ "$WAVE_REF" == "1" ]]; then
  tag="t04_cublas_K${SIZE}_M${SIZE}_N${SIZE}"
  if [[ "$SKIP_DONE" == "1" ]] && done_already "$tag"; then log "04: $tag already complete -> skipped"; else
  run_eval "$tag" -b "$WAVE_CYCLES" $(workload_args) $(scope_args) $(trig_args); fi
fi
mapfile -t TARGETS < <("$PY" "$HERE/wave_targets.py" $WAVES)      # array, not a pipe: spike_eval would eat the pipe's stdin
for k in $WAVE_K; do
  for line in "${TARGETS[@]}"; do
    read -r lab ctas m n <<<"$line"
    tag="t04_K${k}_M${m}_N${n}_${lab}_${ctas}cta"
    if [[ "$SKIP_DONE" == "1" ]] && done_already "$tag"; then log "04: $tag already complete -> skipped"; continue; fi
    log "04: K=$k $lab -> $ctas CTAs, GEMM M=$m N=$n K=$k"
    run_eval "$tag" -b "$WAVE_CYCLES" $(workload_args | sed "s/--dtype [a-z0-9]*/--dtype fp32/; s/--size [0-9]*/--size $k/") --kernel wave --rows "$m" --cols "$n" $(scope_args) $(trig_args)
  done
done
"$PY" "$HERE/summary_wave.py" "$RESULT_DIR" || true
"$PY" "$HERE/overlay_summary.py" "$RESULT_DIR" --param waves --signal voltage --out "$RESULT_DIR/overlay_04_waves_allK_CH1_CH2.png" || true   # all K together (same waves -> line style)
[[ "${FREQMON:-1}" != "0" ]] && { "$PY" "$HERE/overlay_summary.py" "$RESULT_DIR" --param waves --signal freq --out "$RESULT_DIR/overlay_04_waves_allK_freqmon.png" || true; }
for k in $WAVE_K; do                                   # and one overlay family per K
  "$PY" "$HERE/overlay_summary.py" "$RESULT_DIR" --param waves --k "$k" --signal voltage --out "$RESULT_DIR/overlay_04_waves_K${k}_CH1_CH2.png" || true
  if [[ "${FREQMON:-1}" != "0" ]]; then
    "$PY" "$HERE/overlay_summary.py" "$RESULT_DIR" --param waves --k "$k" --signal freq --out "$RESULT_DIR/overlay_04_waves_K${k}_freqmon.png" || true
    "$PY" "$HERE/overlay_summary.py" "$RESULT_DIR" --param waves --k "$k" --signal combined --out "$RESULT_DIR/overlay_04_waves_K${k}_combined.png" || true
  fi
done
