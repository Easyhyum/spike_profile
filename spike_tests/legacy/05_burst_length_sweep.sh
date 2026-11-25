#!/usr/bin/env bash
# 05 — burst-length sweep with V x I: 4 ms / 17 ms / 60 ms / 120 ms bursts (rows 512 / 2048 / 8192 / 8192x2).
#      Shows how peak power, overshoot and the rail droop depend on how long the SGEMM runs. Falls back to
#      voltage-only if no current probe is usable.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -z "${HW_JSON:-}" ]] && hw_refresh
require gpu pmd scope v_ch_ok || exit 0
if [[ "$(hw i_ch_ok)" == "1" ]]; then
  IARGS=(--i-ch "$I_CH" --i-scale "$I_SCALE" --i-offset "$I_OFFSET" --i-div "$I_DIV" --i-center "$I_CENTER" --trig-src i --trig-level "$TRIG_I_LEVEL" --trig-slope pos)
else
  log "05: no current probe -> voltage-only sweep"; IARGS=(--no-current --trig-src v --trig-level "$TRIG_V_LEVEL" --trig-slope neg)
fi
for spec in "512:4" "2048:17" "8192:60" "8192:120"; do
  rows="${spec%%:*}"; on="${spec##*:}"
  run_eval "t05_sweep_${on}ms" -b "$BURSTS" -o "$on" -d "$OFF_MS" --rows "$rows" --smi-ms "$SMI_MS" $(scope_args) "${IARGS[@]}"
done
