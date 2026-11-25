#!/usr/bin/env bash
# lecroy_check.sh — dry run of the HDO4034A path with THIS folder's config.env (no GPU load). sudo ./lecroy_check.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/config.env"
args=(--v-ch "$V_CH" --v-probe "$V_PROBE" --v-div "$V_DIV" --v-center "$V_CENTER" --trig-level "$TRIG_V_LEVEL" --trig-slope neg --trig-src v)
[[ -n "${V2_CH:-}" ]] && args+=(--v2-ch "$V2_CH")
[[ -n "${I_CH:-}" ]] && args+=(--i-ch "$I_CH" --i-scale "$I_SCALE" --i-div "$I_DIV" --i-center "$I_CENTER" --trig-src i --trig-level "$TRIG_I_LEVEL" --trig-slope pos)
[[ -n "${SCOPE_RESOURCE:-}" ]] && args+=(--resource "$SCOPE_RESOURCE")
exec "$PY" "$HERE/lecroy_check.py" "${args[@]}"
