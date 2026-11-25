#!/usr/bin/env bash
# 06 — dtype comparison at the same burst length: fp32 FFMA (~190 W, 230 W peak) vs bf16 tensor core (~145 W).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -z "${HW_JSON:-}" ]] && hw_refresh
require gpu pmd scope v_ch_ok || exit 0
if [[ "$(hw i_ch_ok)" == "1" ]]; then
  IARGS=(--i-ch "$I_CH" --i-scale "$I_SCALE" --i-offset "$I_OFFSET" --i-div "$I_DIV" --i-center "$I_CENTER" --trig-src i --trig-level "$TRIG_I_LEVEL" --trig-slope pos)
else
  IARGS=(--no-current --trig-src v --trig-level "$TRIG_V_LEVEL" --trig-slope neg)
fi
for dt in fp32 bf16; do
  run_eval "t06_$dt" $(common_args) --dtype "$dt" $(scope_args) "${IARGS[@]}"
done
