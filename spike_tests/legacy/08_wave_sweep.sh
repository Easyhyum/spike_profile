#!/usr/bin/env bash
# 08 — di/dt voltage drop vs the number of waves of the GEMM. wavegemm.cu puts exactly one 128x128 output tile per CTA
#      and one CTA per SM, so waves = (M/128)(N/128) / usable SMs (device SMs - 1 held by freqmon). Targets (WAVES):
#      0.5 .. 0.9, 1.0 (full wave, no tail), 1.5 .. 1.9, 2.0, then 3 4 8 16 32 64 and the largest shape evaluated so far
#      (M=N=K=8192 = 2048 CTAs of 256x128 = 58.5 waves). Every point is a single cycle (1 launch), K=8192, WAVE_CYCLES repetitions, fp32.
#      PMD2X + nvidia-smi + freqmon + scope (both voltage channels) on the same launch.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -z "${HW_JSON:-}" ]] && hw_refresh
require gpu pmd scope v_ch_ok || exit 0
: "${WAVES:=0.5 0.6 0.7 0.8 0.9 1.0 1.5 1.6 1.7 1.8 1.9 2.0 3 4 8 16 32 58.51}"
: "${WAVE_CYCLES:=3}"
if [[ "$(hw i_ch_ok)" == "1" ]]; then
  IARGS=(--i-ch "$I_CH" --i-scale "$I_SCALE" --i-offset "$I_OFFSET" --i-div "$I_DIV" --i-center "$I_CENTER" --trig-src i --trig-level "$TRIG_I_LEVEL" --trig-slope pos)
else
  IARGS=(--no-current --trig-src v --trig-level "$TRIG_V_LEVEL" --trig-slope neg)
fi
log "08: wave targets: $WAVES ($WAVE_CYCLES single cycles each)"
mapfile -t TARGETS < <("$PY" "$HERE/wave_targets.py" $WAVES)      # array, not a pipe: spike_eval would eat the pipe's stdin
for line in "${TARGETS[@]}"; do
  read -r lab ctas m n <<<"$line"
  log "08: $lab -> $ctas CTAs, GEMM M=$m N=$n K=8192"
  run_eval "t08_${lab}_${ctas}cta" -b "$WAVE_CYCLES" -o 1 -d "$OFF_MS" --launches 1 --kernel wave --rows "$m" --cols "$n" --size 8192 \
           --idle-before 3 --smi-ms "$SMI_MS" $(scope_args) "${IARGS[@]}" < /dev/null
done
"$PY" "$HERE/wave_sweep_summary.py" "$RESULT_DIR" || true
"$PY" "$HERE/overlay_summary.py" "$RESULT_DIR" --param waves --out "$RESULT_DIR/overlay_08_waves_CH1_CH2.png" || true
