#!/usr/bin/env bash
# common.sh — sourced by every test script: config, auto shape, logging, hardware gating, shared spike_eval arguments.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
mkdir -p "$HERE/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
# one result folder per shell invocation (run_all.sh exports RESULT_DIR so its tests share one folder)
: "${RESULT_DIR:=$HERE/results/${STAMP}_$(basename "${0%.sh}")}"
mkdir -p "$RESULT_DIR"

if [[ $EUID -ne 0 ]]; then
  echo "run with sudo (PMD2X serial port and USBTMC need root):  sudo $0 $*" >&2; exit 2
fi

log() { echo "[$(date +%H:%M:%S)] $*" >> "$HERE/logs/sequence.log"; echo "[$(date +%H:%M:%S)] $*" >> "$RESULT_DIR/sequence.log"; echo "[$(date +%H:%M:%S)] $*"; }

# ---- workload shape: logs/auto_shape.env from find_max_shape.py (00_preflight.sh) unless DTYPE/SIZE are pinned
load_shape() {
  if [[ ! -f "$HERE/logs/auto_shape.env" && ( -z "${DTYPE}" || -z "${SIZE}" ) ]]; then
    log "no logs/auto_shape.env yet -> probing the GPU (find_max_shape.py)"
    "$PY" "$HERE/find_max_shape.py" 2>/dev/null | tail -2
  fi
  [[ -f "$HERE/logs/auto_shape.env" ]] && source "$HERE/logs/auto_shape.env"
  DTYPE="${DTYPE:-${AUTO_DTYPE:-fp32}}"; SIZE="${SIZE:-${AUTO_SIZE:-8192}}"
  : "${AUTO_DURATION_ROWS:=}"; : "${AUTO_LAUNCH_MS:=}"; : "${AUTO_SMS:=}"
  if [[ "${FREQMON:-1}" == "0" && -n "$AUTO_SMS" ]]; then AUTO_USABLE_SMS="$AUTO_SMS"; fi   # no monitor -> every SM runs the GEMM
  export DTYPE SIZE AUTO_USABLE_SMS FREQMON
  cp "$HERE/logs/auto_shape.env" "$RESULT_DIR/auto_shape.env.used" 2>/dev/null || true
}
load_shape

# ---- hw_json: cached preflight result for this shell (re-run with hw_refresh)
hw_refresh() {
  local extra=()
  [[ -n "${SCOPE_RESOURCE}" ]] && extra+=(--resource "$SCOPE_RESOURCE")
  HW_JSON="$("$PY" "$HERE/check_hw.py" --meter "$SCOPE_METER" --v-ch "$V_CH" --v2-ch "${V2_CH:-0}" --i-ch "${I_CH:-0}" \
            --v-div "$V_DIV" --v-center "$V_CENTER" --v-probe "$V_PROBE" "${extra[@]}" 2>/dev/null | tail -1)"
  echo "$HW_JSON" > "$HERE/logs/hw_$STAMP.json"
}
hw() { "$PY" -c "import json,sys; d=json.loads(sys.argv[1]); v=d.get(sys.argv[2]); print('1' if v is True else ('0' if v in (False,None) else v))" "$HW_JSON" "$1"; }

require() {   # require gpu pmd scope v_ch_ok ... -> returns 1 (skip) if any is missing
  local k; for k in "$@"; do
    if [[ "$(hw "$k")" != "1" ]]; then log "SKIP $(basename "$0"): requirement '$k' not met"; return 1; fi
  done; return 0
}

# ---- shared spike_eval arguments
workload_args() {   # shape + logger; every test is single-cycle (one launch per measurement) unless it overrides --launches
  echo "--dtype $DTYPE --size $SIZE --launches 1 -o 1 -d $OFF_MS --smi-ms $SMI_MS --idle-before 3${FREQMON:+}$([[ "${FREQMON:-1}" == "0" ]] && echo " --no-freqmon")${PREWARM_GAP_MS:+ --prewarm-gap-ms $PREWARM_GAP_MS}${PREWARM_MS:+ --prewarm-ms $PREWARM_MS}"
}
scope_args() {      # scope connection + voltage channels (+ names; underscores are turned back into spaces by spike_eval)
  local a=(-m "$SCOPE_METER" --v-ch "$V_CH" --v-probe "$V_PROBE" --v-div "$V_DIV" --v-center "$V_CENTER" --v-name "${V_NAME// /_}" --scope-sample-ns "${SCOPE_SAMPLE_NS:-1000}")
  [[ -n "$SCOPE_RESOURCE" ]] && a+=(--resource "$SCOPE_RESOURCE")
  [[ -n "${V2_CH:-}" && -n "${HW_JSON:-}" && "$(hw v2_ch_ok)" == "1" ]] && a+=(--v2-ch "$V2_CH" --v2-name "${V2_NAME// /_}")
  echo "${a[@]}"
}
trig_args() {       # current probe present -> V x I and current trigger, else voltage-only and voltage trigger
  if [[ -n "${I_CH:-}" && -n "${HW_JSON:-}" && "$(hw i_ch_ok)" == "1" ]]; then
    echo "--i-ch $I_CH --i-scale $I_SCALE --i-offset $I_OFFSET --i-div $I_DIV --i-center $I_CENTER --trig-src i --trig-level $TRIG_I_LEVEL --trig-slope pos"
  else
    echo "--no-current --trig-src v --trig-level $TRIG_V_LEVEL --trig-slope neg"
  fi
}
ensure_usb() {      # re-attach scope / PMD if the usbipd link dropped (idempotent, a few seconds)
  [[ -n "${SCOPE_BUSID:-}" ]] && "$HERE/usb_attach.sh" "$SCOPE_BUSID" 2>&1 | grep -E "attached|stale|timed out|not listed|detach" | sed 's/^/  [usb] /' || true
  [[ -n "${PMD_BUSID:-}" ]] && "$HERE/usb_attach.sh" "$PMD_BUSID" 2>&1 | grep -E "stale|timed out|not listed|detach" | sed 's/^/  [usb] /' || true
}
run_eval() {        # run_eval <tag> <args...>  -> runs spike_eval, logs, prints the output folder
  local tag="$1"; shift
  [[ " $* " == *" -m 1 "* || " $* " == *" -m 2 "* ]] && ensure_usb
  log "RUN  $tag: spike_eval.py $*"
  ( cd "$ROOT" && "$PY" -u spike_eval.py "$@" --tag "$tag" --outdir "$RESULT_DIR" < /dev/null 2>&1 | grep -v -E "UserWarning|FutureWarning|DeprecationWarning|warnings.warn|Locator attempting" | tee "$RESULT_DIR/${tag}.log" )   # keep spike_eval's own WARNING: lines
  local out; out="$(ls -td "$RESULT_DIR"/spike_eval_*"_$tag" 2>/dev/null | head -1 || true)"
  if [[ -n "$out" ]]; then chown -R "${SUDO_USER:-$USER}":"${SUDO_USER:-$USER}" "$RESULT_DIR" 2>/dev/null || true; log "DONE $tag -> $out"; fi
}
