#!/usr/bin/env bash
# 00_preflight.sh — GPU / PMD2X / scope channel go-no-go + workload shape probe (find_max_shape.py).
#   sudo ./00_preflight.sh            # re-probes the shape when logs/auto_shape.env is older than 1 day or PROBE=1
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
log "PREFLIGHT"
echo "usbipd (Windows side):"; /mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "usbipd list" 2>/dev/null | grep -E "2a8d|0483|05ff|Keysight|LeCroy|DSOX|HDO" || echo "  (usbipd list unavailable)"
ensure_usb
hw_refresh
echo "$HW_JSON" | "$PY" -m json.tool
echo
printf "%-12s %s\n" "GPU"      "$([[ $(hw gpu) == 1 ]] && echo "OK  $(hw gpu_info)" || echo "MISSING -> connect 12 V, cold boot, then: wsl --shutdown and reopen")"
printf "%-12s %s\n" "PMD2X"    "$([[ $(hw pmd) == 1 ]] && echo OK || echo "MISSING -> usbipd attach --wsl --busid $PMD_BUSID")"
printf "%-12s %s\n" "scope"    "$([[ $(hw scope) == 1 ]] && echo "OK  $(hw idn)" || echo "MISSING -> usbipd attach --wsl --busid $SCOPE_BUSID (or set SCOPE_RESOURCE)")"
printf "%-12s %s\n" "V ch $V_CH"   "$([[ $(hw v_ch_ok) == 1 ]] && echo "OK (~12 V)  $V_NAME" || echo "NOT USABLE (pegged / not ~12 V)")"
[[ -n "${V2_CH}" ]] && printf "%-12s %s\n" "V2 ch $V2_CH" "$([[ $(hw v2_ch_ok) == 1 ]] && echo "OK (~12 V)  $V2_NAME (recorded in every test)" || echo "NOT USABLE -> tests run with one voltage channel")"
[[ -n "${I_CH}" ]]  && printf "%-12s %s\n" "I ch $I_CH"   "$([[ $(hw i_ch_ok) == 1 ]] && echo "live -> every test records V x I (I_SCALE=$I_SCALE A/V)" || echo "NOT USABLE -> voltage-only")"
echo
if [[ "${PROBE:-0}" == "1" || -z "$(find "$HERE/logs/auto_shape.env" -mtime -1 2>/dev/null)" ]]; then
  log "probing the strongest-spike GEMM shape on this GPU (find_max_shape.py, ~1 min)"
  "$PY" "$HERE/find_max_shape.py" 2>/dev/null; source "$HERE/logs/auto_shape.env"
fi
echo "workload: ${DTYPE:-$AUTO_DTYPE} M=N=K=${SIZE:-$AUTO_SIZE} (1 launch = ${AUTO_LAUNCH_MS:-?} ms, ${AUTO_TFLOPS:-?} TFLOP/s, ${AUTO_PMEAN_W:-?} W mean), duration rows ${AUTO_DURATION_ROWS:-?}, ${AUTO_SMS:-?} SMs (${AUTO_USABLE_SMS:-?} usable for waves)"
