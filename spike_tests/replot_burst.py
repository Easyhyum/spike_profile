#!/usr/bin/env python
"""replot_burst.py <run_dir> [...] — rebuild burst_NN.png (4 x-windows, CH(V) + CH(V2)) and burst_NN_freq.png (+ freqmon
clock) of finished spike_eval runs from burst_NN.npz, marks.json and freqmon.csv, e.g. after a plotting change."""
import glob, json, os, sys
import numpy as np
from onset import load_csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.argv, argv = ['x'], sys.argv
import spike_eval as se


def replot(d):
    M = json.load(open(os.path.join(d, "marks.json"))); marks = M["marks"]; on_ms = M["on_ms"]
    fq = load_csv(os.path.join(d, "freqmon.csv")) if os.path.exists(os.path.join(d, "freqmon.csv")) else None
    stamp = os.path.basename(d).replace("spike_eval_", "")
    for f in sorted(glob.glob(os.path.join(d, "burst_[0-9][0-9].npz"))):
        z = np.load(f); meta = json.loads(str(z["meta"])); kb = int(os.path.basename(f)[6:8])
        t, v, i, p = z["t"].astype(np.float64), z["v"].astype(np.float64), z["i"].astype(np.float64), z["p"].astype(np.float64); v2 = z["v2"].astype(np.float64) if "v2" in z.files else None
        s0 = meta.get("onset_s", se.scope_onset(t, v)); tp = t - s0
        vch, v2ch = meta.get("v_ch", 2), meta.get("v2_ch")
        dev = "Keysight DSOX1202A" if "DSOX" in meta.get("idn", "DSOX") else "LeCroy HDO4034A"
        dev_name = dev + f" CH{vch}" + (f"+CH{v2ch}" if v2ch else "")
        v_lab = f"CH{vch} {meta.get('v_name') or 'rail voltage'} (V)"; v2_lab = f"CH{v2ch} {meta.get('v2_name') or 'rail voltage (2nd channel)'} (V)" if v2 is not None else None
        ttl = f"{stamp}  burst {kb}  measured with {dev_name}  {meta['dt']*1e6:.2f} us/sample  load {on_ms:.0f} ms"
        cap = se.power_cap_w(d)
        se.plot_burst(os.path.join(d, f"burst_{kb:02d}.png"), tp, v, i, p, ttl, device=dev_name, dt_us=meta["dt"] * 1e6, v2=v2, v_lab=v_lab, v2_lab=v2_lab, cap_w=cap)
        for span_ms in (100, 200):
            se.plot_burst(os.path.join(d, f"burst_{kb:02d}_{span_ms}ms.png"), tp, v, i, p, ttl + f"  [{span_ms} ms window]", device=dev_name, dt_us=meta["dt"] * 1e6, v2=v2, v_lab=v_lab, v2_lab=v2_lab, cap_w=cap, windows=[(1e-3, span_ms * 1e-3)])
        if not np.all(np.isnan(i)): se.plot_rise(os.path.join(d, f"burst_{kb:02d}_rise.png"), tp, i, f"{stamp}  cycle {kb}  current rise, {dev}", unit="A")
        fq_t = fq_f = None
        if fq is not None and kb < len(marks) and marks[kb].get("gt_before_ns") is not None:
            rel = (fq["t_gpu_ns"] - marks[kb]["gt_before_ns"]) / 1e9; sel = (rel > tp[0] - 0.01) & (rel < tp[-1] + 0.01)
            fq_t, fq_f = rel[sel], fq["sm_clock_MHz"][sel]
        se.plot_burst_freq(os.path.join(d, f"burst_{kb:02d}_freq.png"), tp, v, None if np.all(np.isnan(p)) else p, fq_t, fq_f, ttl,
                           device=dev_name, dt_us=meta["dt"] * 1e6, v2=v2, v_lab=v_lab, v2_lab=v2_lab, cap_w=cap)
        print("replotted", d, "burst", kb)
    # PMD2X 1 ms figures (burst_NN_pmd.png) from burst_NN_pmd.csv
    g = M.get("gemm") or {}; K = M.get("K", 1)
    for f in sorted(glob.glob(os.path.join(d, "burst_[0-9][0-9]_pmd.csv"))):
        kb = int(os.path.basename(f)[6:8]); a4 = np.loadtxt(f, delimiter=",", skiprows=1)     # header: t_s, v (cable avg), i, p (PCIE1+2)
        if a4.ndim != 2 or a4.shape[1] < 4: continue
        t, v, i, p = a4[:, 0], a4[:, 1], a4[:, 2], a4[:, 3]
        ttl = (f"{stamp}  {'single cycle' if K == 1 else f'burst (K={K} launches)'} {kb}  GEMM M={g.get('M')} N={g.get('N')} K={g.get('K')} {g.get('dtype')}"
               f"  measured with ElmorLabs PMD2X (PCIE1+PCIE2 cables, 1 ms raw)  load {on_ms:.0f} ms")
        se.plot_burst(os.path.join(d, f"burst_{kb:02d}_pmd.png"), t, v, i, p, ttl, windows=[(1e-3, None)], device="PMD2X", dt_us=1000.0, cap_w=se.power_cap_w(d))


for d in argv[1:]:
    replot(d.rstrip("/"))
