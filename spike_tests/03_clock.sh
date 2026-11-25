#!/usr/bin/env bash
# 03 — di/dt drop vs graphics clock: default (unlocked) and locked max, max*0.9 .. 0.5 (nearest supported clock from
#      `nvidia-smi -q -d SUPPORTED_CLOCKS`, so it works on any GPU). CLOCK_CYCLES single cycles of the max shape per setting.
#      Clocks are locked through clockctl.py (one elevated nvidia-smi server on Windows = one UAC prompt); freqmon in the
#      result tells the clock the kernel really ran at (the GPU may not hold a lock above its power-limited clock).
#      Outputs: clock_sweep_summary.csv/png, overlay_03_clock_CH1_CH2.png, overlay_03_clock_freqmon.png
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -z "${HW_JSON:-}" ]] && hw_refresh
require gpu pmd scope v_ch_ok ${REQUIRE_I:+i_ch_ok} || exit 0   # REQUIRE_I=1 (current campaign): skip unless the current probe is live
: "${CLOCK_RATIOS:=0.9 0.8 0.7 0.6 0.5}"
CC="$PY $HERE/clockctl.py"
tj="$($CC targets $CLOCK_RATIOS)"; log "03: supported max = $(echo "$tj" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["max"])') MHz; targets = $tj"
MAX="$(echo "$tj" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["max"])')"
TGT=( $(echo "$tj" | "$PY" -c 'import json,sys; print(" ".join(map(str, json.load(sys.stdin)["targets"])))') )
cleanup() { $CC reset || true; [[ -z "${KEEP_CLOCK_SERVER:-}" ]] && $CC stop || true; }
trap cleanup EXIT

run_eval t03_clk_default -b "$CLOCK_CYCLES" $(workload_args) $(scope_args) $(trig_args)
if ! $CC start; then log "03: clock server not available (UAC declined?) -> locked settings skipped"; exit 0; fi
i=0
for mhz in "$MAX" "${TGT[@]}"; do
  lab=$([[ $i -eq 0 ]] && echo "max" || echo "r$(echo "$CLOCK_RATIOS" | cut -d' ' -f$i)"); i=$((i+1))
  if $CC lock "$mhz"; then
    run_eval "t03_clk_${lab}_${mhz}MHz" -b "$CLOCK_CYCLES" $(workload_args) $(scope_args) $(trig_args)
  else
    log "03: could not lock $mhz MHz -> skipped"
  fi
done
$CC reset; [[ -z "${KEEP_CLOCK_SERVER:-}" ]] && $CC stop; trap - EXIT
"$PY" "$HERE/summary_clock.py" "$RESULT_DIR" || true
"$PY" "$HERE/overlay_summary.py" "$RESULT_DIR" --param clk --signal voltage --out "$RESULT_DIR/overlay_03_clock_CH1_CH2.png" || true
if [[ "${FREQMON:-1}" != "0" ]]; then
  "$PY" "$HERE/overlay_summary.py" "$RESULT_DIR" --param clk --signal freq --out "$RESULT_DIR/overlay_03_clock_freqmon.png" || true
  "$PY" "$HERE/overlay_summary.py" "$RESULT_DIR" --param clk --signal combined --out "$RESULT_DIR/overlay_03_clock_combined.png" || true
fi
