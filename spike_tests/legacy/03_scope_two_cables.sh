#!/usr/bin/env bash
# 03 — two 12 V sources on V_CH and V2_CH (2026-11-12: CH1 PSU cable vs CH2 PCIe slot): same-voltage check (per-burst CH difference figure + V2-V1 mV column).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -z "${HW_JSON:-}" ]] && hw_refresh
[[ -n "${V2_CH}" ]] || { log "SKIP 03: V2_CH not set"; exit 0; }
require gpu pmd scope v_ch_ok v2_ch_ok || exit 0
run_eval t03_2cable $(common_args) $(scope_args) --no-current --trig-src v --trig-level "$TRIG_V_LEVEL" --trig-slope neg
