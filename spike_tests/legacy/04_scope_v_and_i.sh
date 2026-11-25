#!/usr/bin/env bash
# 04 — voltage (V_CH) + current probe (I_CH): per-sample P = V x I at 1 us, peak / overshoot / undershoot per burst.
#      Trigger on the current rising edge (cleaner than the voltage edge).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -z "${HW_JSON:-}" ]] && hw_refresh
require gpu pmd scope v_ch_ok i_ch_ok || exit 0
run_eval t04_vi $(common_args) $(scope_args) --i-ch "$I_CH" --i-scale "$I_SCALE" --i-offset "$I_OFFSET" --i-div "$I_DIV" --i-center "$I_CENTER" \
         --trig-src i --trig-level "$TRIG_I_LEVEL" --trig-slope pos
