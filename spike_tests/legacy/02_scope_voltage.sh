#!/usr/bin/env bash
# 02 — scope rail voltage (1 us) + PMD2X + nvidia-smi + freqmon on the same launches. Trigger: V falling edge.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -z "${HW_JSON:-}" ]] && hw_refresh
require gpu pmd scope v_ch_ok || exit 0
run_eval t02_v $(common_args) $(scope_args) --no-current --trig-src v --trig-level "$TRIG_V_LEVEL" --trig-slope neg
