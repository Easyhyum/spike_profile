#!/usr/bin/env python
"""
clockctl.py — lock / unlock the GPU graphics clock from WSL2.

nvidia-smi inside WSL cannot change clocks ("Unknown Error") and the Windows nvidia-smi needs an elevated shell,
so this starts ONE elevated PowerShell "clock server" on the Windows side (a single UAC prompt), which executes
`nvidia-smi -lgc/-rgc` on request. Requests and replies go through files under C:\\Users\\<you>\\.spike_clockctl\\.

    python clockctl.py supported            # supported graphics clocks (MHz, descending) from the WSL nvidia-smi
    python clockctl.py start                # launch the elevated server (UAC prompt appears on the Windows desktop)
    python clockctl.py lock 2400            # -lgc 2400,2400, then verifies clocks.sm == 2400 (+/- 15 MHz)
    python clockctl.py reset                # -rgc
    python clockctl.py stop                 # end the server
    python clockctl.py targets 0.9 0.8 ...  # nearest supported clock <= max*ratio for each ratio
    python clockctl.py limits               # power limits (current / default / min / max W) from the WSL nvidia-smi
    python clockctl.py pl 200               # -pl 200 (board power limit, W; max = "Max Power Limit"), verified via power.limit
    python clockctl.py plreset              # -pl <default limit>
    python clockctl.py start [minutes]      # server lifetime (default 60); run_all.sh uses 240 when it owns the server
Exit code != 0 when the lock could not be applied/verified; the test scripts then skip that setting.
When the server exits (stop / lifetime / window closed) it restores -rgc AND the default power limit.
"""
import os, re, subprocess, sys, time, json

WIN_USER = None
PS_EXE = "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe"   # absolute path: sudo PATH has no /mnt/c


def win_user():
    global WIN_USER
    if WIN_USER is None:
        out = subprocess.run([PS_EXE, "-NoProfile", "-Command", "$env:USERNAME"], capture_output=True, text=True, timeout=30).stdout
        WIN_USER = out.strip().splitlines()[-1].strip()
    return WIN_USER


def ctl_dir_win():   return f"C:\\Users\\{win_user()}\\.spike_clockctl"
def ctl_dir_wsl():   return f"/mnt/c/Users/{win_user()}/.spike_clockctl"


def nvsmi():
    for c in ("/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi"):
        if os.path.exists(c):
            return c
    return "nvidia-smi"


def supported():
    out = subprocess.run([nvsmi(), "-q", "-d", "SUPPORTED_CLOCKS"], capture_output=True, text=True, timeout=30).stdout
    clks = sorted({int(m) for m in re.findall(r"Graphics\s*:\s*(\d+)\s*MHz", out)}, reverse=True)
    return clks


def current_sm():
    out = subprocess.run([nvsmi(), "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    try:
        return int(float(out.strip().splitlines()[-1]))
    except Exception:
        return None


SERVER_PS1 = r'''
$dir = "{dir}"
$cmdf = Join-Path $dir "cmd.txt"; $resf = Join-Path $dir "result.txt"; $alive = Join-Path $dir "alive.txt"
$smi = "C:\Windows\System32\nvidia-smi.exe"
Set-Content -Path $alive -Value "up $(Get-Date -Format o)"
$deadline = (Get-Date).AddMinutes({minutes})
while ((Get-Date) -lt $deadline) {{
  Set-Content -Path $alive -Value "up $(Get-Date -Format o)"
  if (Test-Path $cmdf) {{
    $cmd = (Get-Content $cmdf -Raw).Trim(); Remove-Item $cmdf -Force
    if ($cmd -eq "quit") {{ Set-Content -Path $resf -Value "bye"; break }}
    $args = $cmd -split " "
    $out = & $smi @args 2>&1 | Out-String
    Set-Content -Path $resf -Value ("rc=" + $LASTEXITCODE + "`n" + $out)
  }}
  Start-Sleep -Milliseconds 200
}}
& $smi -rgc | Out-Null
$dpl = (& $smi --query-gpu=power.default_limit --format=csv,noheader,nounits | Out-String).Trim()
if ($dpl -match "^[0-9.]+$") {{ & $smi -pl $dpl | Out-Null }}
Remove-Item $alive -Force -ErrorAction SilentlyContinue
'''


def server_alive():
    f = os.path.join(ctl_dir_wsl(), "alive.txt")
    try:
        return time.time() - os.path.getmtime(f) < 5
    except Exception:
        return False


def start(minutes=60):
    d = ctl_dir_wsl(); os.makedirs(d, exist_ok=True)
    for f in ("cmd.txt", "result.txt"):
        try: os.remove(os.path.join(d, f))
        except Exception: pass
    if server_alive():
        print("clock server already running"); return True
    ps1 = os.path.join(d, "server.ps1")
    open(ps1, "w").write(SERVER_PS1.format(dir=ctl_dir_win(), minutes=minutes))
    win_ps1 = ctl_dir_win() + "\\server.ps1"
    cmd = (f"Start-Process powershell.exe -Verb RunAs -WindowStyle Minimized "
           f"-ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','{win_ps1}'")
    # Start-Process -Verb RunAs only returns once the UAC prompt is answered, and killing the launcher cancels the prompt:
    # launch it detached and never wait on it; the server's alive.txt tells us when the elevation went through
    subprocess.Popen([PS_EXE, "-NoProfile", "-Command", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    print("waiting for the UAC prompt to be accepted on the Windows desktop ...", flush=True)
    t0 = time.time()
    while time.time() - t0 < 240:                    # up to 4 min for someone to click "Yes"
        if server_alive():
            print("clock server up"); return True
        time.sleep(1)
    print("clock server did not start (UAC declined / timed out)"); return False


def request(args, timeout=30):
    d = ctl_dir_wsl(); resf = os.path.join(d, "result.txt")
    try: os.remove(resf)
    except Exception: pass
    open(os.path.join(d, "cmd.txt"), "w").write(args)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(resf):
            time.sleep(0.2)
            try:
                return open(resf, errors="replace").read()
            except Exception:
                pass
        time.sleep(0.2)
    return None


def lock(mhz, tol=15):
    if not server_alive() and not start():
        return False
    r = request(f"-lgc {mhz},{mhz}")
    if r is None or "rc=0" not in r:
        print("lock failed:", (r or "no reply").strip()[:200]); return False
    time.sleep(1.0)
    for _ in range(10):
        cur = current_sm()
        if cur is not None and abs(cur - mhz) <= tol:
            print(f"locked: clocks.sm = {cur} MHz"); return True
        time.sleep(0.5)
    cur = current_sm()
    if cur is not None and cur > 1000 and cur < mhz:
        # the driver accepted the lock but the GPU tops out below the requested value (e.g. 3090 requested, 2850 reached)
        print(f"locked (clamped): clocks.sm = {cur} MHz for requested {mhz}"); return True
    print(f"lock not verified: clocks.sm = {cur} MHz (requested {mhz})"); return False


def reset():
    if not server_alive():
        print("no clock server; nothing to reset (clocks are unlocked when the server exits)"); return True
    r = request("-rgc")
    ok = r is not None and "rc=0" in r
    print("reset:", "ok" if ok else (r or "no reply").strip()[:200]); return ok


def stop():
    if server_alive():
        request("quit", timeout=10)
    print("clock server stopped")


def limits():
    out = subprocess.run([nvsmi(), "--query-gpu=power.limit,power.default_limit,power.min_limit,power.max_limit", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout
    try:
        cur, dflt, mn, mx = [float(x) for x in out.strip().splitlines()[-1].split(",")]
        return dict(current=cur, default=dflt, min=mn, max=mx)
    except Exception:
        return {}


def power_limit(watts, tol=0.5):
    """-pl <watts> through the server, verified with the WSL nvidia-smi (power.limit)."""
    if not server_alive():
        print("no clock server (clockctl.py start)"); return False
    lim = limits()
    if lim and not (lim["min"] - tol <= watts <= lim["max"] + tol):
        print(f"power limit {watts} W outside the allowed range {lim['min']:.0f}-{lim['max']:.0f} W"); return False
    r = request(f"-pl {watts:g}")
    if r is None or "rc=0" not in r:
        print("power limit request failed:", (r or "no reply").strip()[:200]); return False
    for _ in range(10):
        time.sleep(0.5); got = limits().get("current")
        if got is not None and abs(got - watts) <= tol:
            print(f"power limit {got:.0f} W (default {limits().get('default', float('nan')):.0f}, max {limits().get('max', float('nan')):.0f})"); return True
    print(f"power limit not verified (reads {got})"); return False


def power_limit_reset():
    lim = limits()
    if not lim: print("cannot read the limits"); return False
    if abs(lim["current"] - lim["default"]) <= 0.5: print(f"power limit already default ({lim['default']:.0f} W)"); return True
    return power_limit(lim["default"])


def targets(ratios):
    clks = supported()
    if not clks:
        return []
    mx = clks[0]; out = []
    for r in ratios:
        want = mx * r
        cand = [c for c in clks if c <= want + 0.5]
        out.append(cand[0] if cand else clks[-1])
    return mx, out


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    op = sys.argv[1]
    if op == "supported":
        print(" ".join(map(str, supported())))
    elif op == "max":
        print(supported()[0])
    elif op == "targets":
        mx, t = targets([float(x) for x in sys.argv[2:]])
        print(json.dumps(dict(max=mx, targets=t)))
    elif op == "start":
        sys.exit(0 if start(int(sys.argv[2]) if len(sys.argv) > 2 else 60) else 1)
    elif op == "limits":
        print(json.dumps(limits()))
    elif op == "pl":
        sys.exit(0 if power_limit(float(sys.argv[2])) else 1)
    elif op == "plreset":
        sys.exit(0 if power_limit_reset() else 1)
    elif op == "lock":
        sys.exit(0 if lock(int(sys.argv[2])) else 1)
    elif op == "reset":
        sys.exit(0 if reset() else 1)
    elif op == "stop":
        stop()
    elif op == "current":
        print(current_sm())
    else:
        print(__doc__); sys.exit(2)


if __name__ == "__main__":
    main()
