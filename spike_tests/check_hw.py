#!/usr/bin/env python
"""
check_hw.py — preflight for the spike test sequence. Prints a JSON line with what is usable:
    gpu           CUDA visible in WSL (nvidia-smi + torch)
    pmd           PMD2X serial port present and answering
    scope         scope reachable (*IDN?)
    ch[n]         per-channel verdict: "ok" (live signal / noise), "pegged" (ADC stuck at one code), "absent"
    v_ch_ok       the configured voltage channel reads ~12 V
    v2_ch_ok      the second voltage channel reads ~12 V
    i_ch_ok       the current channel is live (not pegged) and shows an idle level (|V| < 1 V after scaling check)
Exit code 0 always; the shell scripts decide what to skip. Run with sudo (serial + USB).
Usage: sudo venv/bin/python spike_tests/check_hw.py [--meter 1|2] [--resource R] [--v-ch 2] [--v2-ch 1] [--i-ch 1] [--v-div 0.5] [--v-center 12] [--v-probe 1]
"""
import argparse, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def check_gpu():
    try:
        import pmd2_fast as pf
        smi = pf.find_nvidia_smi() or "nvidia-smi"          # absolute path: sudo's PATH lacks /usr/lib/wsl/lib
        out = subprocess.run([smi, "--query-gpu=name,power.draw,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        if out.returncode != 0 or "Failed" in out.stdout + out.stderr:
            return False, (out.stdout + out.stderr).strip()[:80]
        import torch
        f = [x.strip() for x in out.stdout.strip().split(",")]
        info = f"{f[0]}, {f[1]}" + (f", PCIe gen {f[2]} (max {f[3]}) x{f[4]} (slot x{f[5]})" if len(f) >= 6 else "")
        return torch.cuda.is_available(), info
    except Exception as e:
        return False, str(e)[:80]


def check_pmd():
    try:
        import pmd2_fast as pf
        port = pf.find_port()
        if port is None:
            return False, "no PMD2X port"
        ser = pf.open_device(port); buf = pf.read_packet(ser); ser.close()
        return buf is not None, port
    except Exception as e:
        return False, str(e)[:80]


def read_channel(inst, ch, v_probe, v_div, v_center):
    import numpy as np
    q = lambda s: inst.query(s).strip(); w = inst.write
    w(f":CHANnel{ch}:DISPlay ON"); w(f":CHANnel{ch}:PROBe {v_probe}"); w(f":CHANnel{ch}:COUPling DC")
    for sc in (v_div, v_div * 2, v_div * 5, v_div * 10):
        w(f":CHANnel{ch}:SCALe {sc}"); w(f":CHANnel{ch}:OFFSet {v_center}")
        if q(":SYSTem:ERRor?").startswith("+0"):
            break
    w(":SINGle"); q("*OPC?"); time.sleep(0.3); t0 = time.time()
    while int(q(":OPERegister:CONDition?")) & 8 and time.time() - t0 < 5:
        time.sleep(0.02)
    w(f":WAVeform:SOURce CHANnel{ch}"); w(":WAVeform:FORMat WORD"); w(":WAVeform:BYTeorder LSBFirst"); w(":WAVeform:UNSigned 0")
    w(":WAVeform:POINts:MODE RAW"); w(":WAVeform:POINts 5000")
    pre = q(":WAVeform:PREamble?").split(",")
    raw = inst.query_binary_values(":WAVeform:DATA?", datatype="h", is_big_endian=False, container=np.array)
    y = (raw.astype(float) - float(pre[9])) * float(pre[7]) + float(pre[8])
    uniq = len(np.unique(raw))
    verdict = "pegged" if uniq <= 2 else "ok"
    return dict(verdict=verdict, mean=float(y.mean()), std=float(y.std()), codes=uniq)


def read_channel_lecroy(inst, ch, v_probe, v_div, v_center, current=False):
    """MAUI: set the channel, take one AUTO acquisition at 1 ms/div, read the waveform (WAVEDESC + DAT1)."""
    import numpy as np, time
    from spike_eval import LecroyHDO
    w = inst.write; q = lambda s: inst.query(s).strip()
    w("CHDR OFF"); w(f"C{ch}:TRA ON")
    if not current:
        w(f"C{ch}:CPL D1M"); w(f"C{ch}:ATTN {v_probe}")
    for sc in (v_div, v_div * 2, v_div * 5, v_div * 10):
        w(f"C{ch}:VDIV {sc}"); w(f"C{ch}:OFST {-v_center}")
        try:
            if abs(float(q(f"C{ch}:OFST?")) + v_center) <= 0.05 * max(abs(v_center), 1.0): break
        except Exception:
            break
    w("TDIV 1e-3"); w("TRMD STOP"); w("TRSE EDGE,SR,C%d,HT,OFF" % ch); w("C%d:TRLV 1e3" % ch)        # parked trigger, then SINGLE + force =
    for _ in range(2):                                                                              # a guaranteed fresh record (an AUTO sweep
        w("TRMD SINGLE"); q("*OPC?"); time.sleep(0.15); w("FRTR"); t0 = time.time()                 # can leave the previous run's trace in memory)
        while time.time() - t0 < 3 and not q("TRMD?").upper().startswith("STOP"): time.sleep(0.02)
    w("CFMT DEF9,WORD,BIN"); w("CORD LO")
    d = LecroyHDO.parse_wavedesc(inst.query_binary_values(f"C{ch}:WF? DESC", datatype="B", container=bytes))
    raw = inst.query_binary_values(f"C{ch}:WF? DAT1", datatype="h" if d["comm_type"] == 1 else "b", is_big_endian=(d["order"] == ">"), container=np.array)
    raw = raw[:20000]; y = raw.astype(float) * d["gain"] - d["offset"]; uniq = len(np.unique(raw))
    return dict(verdict="pegged" if uniq <= 2 else "ok", mean=float(y.mean()), std=float(y.std()), codes=uniq, unit=d["unit"])


def check_scope(a):
    res = {}
    try:
        from spike_eval import _visa_open, KeysightDSOX, LecroyHDO     # shared open: device clear, REN quirk, timeouts
        vids = KeysightDSOX.VIDS if a.meter == 1 else LecroyHDO.VIDS
        try:
            inst, r = _visa_open(a.resource or None, vids, 15000)
        except RuntimeError as e:
            return dict(scope=False, reason=str(e)[:120])
        inst.chunk_size = (1 << 20) if a.meter == 1 else LecroyHDO.CHUNK      # HDO: >= 64 KB requests never complete
        res["scope"] = True; res["idn"] = inst.query("*IDN?").strip(); res["resource"] = r
        if a.meter != 1:
            inst.write("CHDR OFF"); inst.write("*CLS"); inst.write("TRMD STOP")
            for ch in sorted({c for c in (a.v_ch, a.v2_ch) if c}):
                res[f"ch{ch}"] = read_channel_lecroy(inst, ch, a.v_probe, a.v_div, a.v_center)
            if a.i_ch:
                res[f"ch{a.i_ch}_as_current"] = read_channel_lecroy(inst, a.i_ch, 1, 10.0, 0.0, current=True)   # probe units (A for a ProBus probe)
            inst.write("TRMD AUTO"); inst.close(); return res
        inst.write("*CLS"); inst.write(":STOP"); inst.write(":TRIGger:SWEep AUTO"); inst.write(":TIMebase:SCALe 1e-3"); inst.write(":ACQuire:TYPE HRESolution")
        chans = sorted({c for c in (a.v_ch, a.v2_ch, a.i_ch) if c})
        for ch in chans:
            res[f"ch{ch}"] = read_channel(inst, ch, a.v_probe, a.v_div, a.v_center)
        # current channel: look at it on a 1 V/div, 0 V centre scale (probe output volts)
        if a.i_ch:
            res[f"ch{a.i_ch}_as_current"] = read_channel(inst, a.i_ch, 1, 1.0, 0.0)
        inst.write(":RUN"); inst.close()
    except Exception as e:
        res["scope"] = False; res["reason"] = str(e)[:120]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meter", type=int, default=1); ap.add_argument("--resource", default="")
    ap.add_argument("--v-ch", type=int, default=2); ap.add_argument("--v2-ch", type=int, default=0); ap.add_argument("--i-ch", type=int, default=0)
    ap.add_argument("--v-div", type=float, default=0.5); ap.add_argument("--v-center", type=float, default=12.0); ap.add_argument("--v-probe", type=float, default=1)
    ap.add_argument("--skip-scope", action="store_true")
    a = ap.parse_args()
    out = {}
    out["gpu"], out["gpu_info"] = check_gpu()
    out["pmd"], out["pmd_info"] = check_pmd()
    if not a.skip_scope:
        out.update(check_scope(a))
        v = out.get(f"ch{a.v_ch}"); out["v_ch_ok"] = bool(v and v["verdict"] == "ok" and 10.5 < v["mean"] < 13.5)
        v2 = out.get(f"ch{a.v2_ch}") if a.v2_ch else None; out["v2_ch_ok"] = bool(v2 and v2["verdict"] == "ok" and 10.5 < v2["mean"] < 13.5)
        i = out.get(f"ch{a.i_ch}_as_current") if a.i_ch else None
        lim = 10.0 if (i and str(i.get("unit", "")).upper().startswith("A")) else 2.0   # idle: < 10 A when the scope reads amperes, < 2 V of probe output otherwise
        out["i_ch_ok"] = bool(i and i["verdict"] == "ok" and abs(i["mean"]) < lim)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
