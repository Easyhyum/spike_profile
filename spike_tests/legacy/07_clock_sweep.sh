#!/usr/bin/env bash
# 07 — di/dt voltage drop vs graphics clock. One burst (1 cycle) per setting:
#      default (unlocked boost), max, max*0.9, 0.8, 0.7, 0.6, 0.5  (nearest supported clock <= target, device-generic:
#      the list comes from `nvidia-smi -q -d SUPPORTED_CLOCKS`). Clocks are locked through clockctl.py, which runs one
#      elevated nvidia-smi "server" on Windows (a single UAC prompt). Every setting is verified (clocks.sm) and the
#      in-kernel freqmon in the result confirms the clock the kernel really ran at.
#      Needs the scope voltage channel; uses the current probe when available.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -z "${HW_JSON:-}" ]] && hw_refresh
require gpu pmd scope v_ch_ok || exit 0
: "${CLOCK_RATIOS:=0.9 0.8 0.7 0.6 0.5}"
: "${CLOCK_BURSTS:=1}"
CC="$PY $HERE/clockctl.py"
if [[ "$(hw i_ch_ok)" == "1" ]]; then
  IARGS=(--i-ch "$I_CH" --i-scale "$I_SCALE" --i-offset "$I_OFFSET" --i-div "$I_DIV" --i-center "$I_CENTER" --trig-src i --trig-level "$TRIG_I_LEVEL" --trig-slope pos)
else
  IARGS=(--no-current --trig-src v --trig-level "$TRIG_V_LEVEL" --trig-slope neg)
fi
WARGS=(-b "$CLOCK_BURSTS" -o "$ON_MS" -d "$OFF_MS" --idle-before 3 --smi-ms "$SMI_MS"); [[ -n "$ROWS" ]] && WARGS+=(--rows "$ROWS")

tj="$($CC targets $CLOCK_RATIOS)"; log "07: supported max = $(echo "$tj" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["max"])') MHz; targets = $tj"
MAX="$(echo "$tj" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["max"])')"
TGT=( $(echo "$tj" | "$PY" -c 'import json,sys; print(" ".join(map(str, json.load(sys.stdin)["targets"])))') )
cleanup() { $CC reset || true; $CC stop || true; }
trap cleanup EXIT

# 1) default: unlocked boost
run_eval t07_clk_default "${WARGS[@]}" $(scope_args) "${IARGS[@]}"

# 2) locked settings: max, then max*ratio
if ! $CC start; then log "07: clock server not available (UAC declined?) -> locked settings skipped"; exit 0; fi
i=0
for mhz in "$MAX" "${TGT[@]}"; do
  lab=$([[ $i -eq 0 ]] && echo "max" || echo "r$(echo "$CLOCK_RATIOS" | cut -d' ' -f$i)"); i=$((i+1))
  if $CC lock "$mhz"; then
    run_eval "t07_clk_${lab}_${mhz}MHz" "${WARGS[@]}" $(scope_args) "${IARGS[@]}"
  else
    log "07: could not lock $mhz MHz -> skipped"
  fi
done
$CC reset; $CC stop; trap - EXIT
"$PY" "$HERE/clock_sweep_summary.py" "$RESULT_DIR" || true
"$PY" "$HERE/overlay_summary.py" "$RESULT_DIR" --param clk --out "$RESULT_DIR/overlay_07_clock_CH1_CH2.png" || true
