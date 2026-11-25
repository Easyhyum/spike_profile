#!/usr/bin/env bash
# usb_attach.sh <busid> — (re)attach a usbipd device to WSL, recovering a wedged vhci port first.
# A scope that is unplugged / power-cycled / detached mid-transfer leaves the WSL vhci port in state 006 with a dead
# socket; every USB call then hangs in D state and `usbipd attach` blocks. Detaching the port via sysfs clears it.
#   sudo ./usb_attach.sh 6-3     # scope      sudo ./usb_attach.sh 6-4     # PMD2X
set -euo pipefail
busid="${1:?busid, e.g. 6-3}"
bus="${busid%%-*}"; dev="${busid##*-}"
devid="$(printf '%04x%04x' "$bus" "$dev")"          # vhci status 'dev' column: bus/port in hex, e.g. 6-3 -> 00060003
state="$(timeout 20 /mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "usbipd list" 2>/dev/null | grep -E "^${busid}\s" | sed 's/\r//' || true)"
echo "usbipd: ${state:-"(busid $busid not listed — is the device plugged in / bound?)"}"
st="/sys/devices/platform/vhci_hcd.0/status"
if [[ -r "$st" ]]; then
  port="$(awk -v d="$devid" 'NR>1 && $5==d {print $2+0}' "$st" || true)"
  if [[ -n "$port" ]] && ! grep -q "Attached" <<<"$state"; then
    echo "vhci port $port still holds $busid (stale) -> detaching via sysfs"
    echo "$port" > /sys/devices/platform/vhci_hcd.0/detach; sleep 2
  fi
fi
probe_alive() {   # the vhci port can look healthy (state 006, live socket) while every URB fails (-104) after a usbipd link drop:
                  # ask the device for its descriptor through pyusb; a wedged port times out
  local vid="$(grep -oE '[0-9a-f]{4}:[0-9a-f]{4}' <<<"$state" | head -1 | cut -d: -f1)"; [[ -n "$vid" ]] || return 0
  local py="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)/venv/bin/python"; [[ -x "$py" ]] || py=python3
  local rc=0
  timeout 15 "$py" - "$vid" <<'PY' || rc=$?
import sys
try: import usb.core
except Exception: sys.exit(0)                       # no pyusb here: cannot judge -> assume alive (never detach on a tooling problem)
d = usb.core.find(idVendor=int(sys.argv[1], 16))
if d is None: sys.exit(0)                           # not visible to libusb (e.g. still enumerating): the caller re-checks vhci anyway
try: d.ctrl_transfer(0x80, 6, 0x0100, 0, 18, timeout=2000); sys.exit(0)
except Exception as e:
    errno = getattr(e, "errno", None); print(f"probe: {type(e).__name__} errno {errno}: {str(e)[:60]}")
    sys.exit(2 if errno in (110, 104, 32, 19, 5) else 0)   # timeout / reset / pipe / no device / io -> dead; anything else -> assume alive
PY
  return $rc
}
if grep -q "Attached" <<<"$state"; then
  if [[ -r "$st" ]] && grep -q "$devid" "$st"; then
    if probe_alive; then echo "$busid already attached"; exit 0; fi
    port="$(awk -v d="$devid" 'NR>1 && $5==d {print $2+0}' "$st" || true)"
    echo "$busid is attached but does not answer (wedged vhci port after a link drop) -> sysfs detach port $port, usbipd detach, re-attach"
    [[ -n "$port" ]] && { echo "$port" > /sys/devices/platform/vhci_hcd.0/detach; sleep 2; }
    timeout 30 /mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "usbipd detach --busid $busid" >/dev/null 2>&1 || true
    sleep 1
  else
  # Windows still says Attached but the vhci port is empty (link died, e.g. after a network drop): detach on the Windows side first
  echo "usbipd says Attached but no vhci port holds $busid -> usbipd detach, then re-attach"
  timeout 30 /mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "usbipd detach --busid $busid" >/dev/null 2>&1 || true
  sleep 1
  fi
fi
timeout 60 /mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "usbipd attach --wsl --busid $busid" 2>&1 | sed 's/\r//' | tail -2 || { echo "attach timed out: run 'wsl --shutdown' from Windows and reopen"; exit 1; }
for i in $(seq 1 20); do
  grep -q "$devid" "$st" 2>/dev/null && { sleep 1; echo "attached: $(timeout 20 /mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "usbipd list" 2>/dev/null | grep -E "^${busid}\s" | sed 's/\r//')"; exit 0; }
  sleep 1
done
echo "device did not appear in vhci within 20 s"; exit 1
