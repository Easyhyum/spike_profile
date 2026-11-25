#!/usr/bin/env python
"""summary_runs.py <result_dir> — one row per cycle of every spike_eval run in the folder (any test) -> runs_summary.csv/json
and a per-run median table on stdout. Metrics per cycle (scope, aligned on the voltage onset; CH(V) = primary channel,
CH(V2) = second channel when recorded):
  didt_drop_300us   idle - min within the first 300 us after the onset (the di/dt event)      [V]
  t_min_us          time of that minimum after the onset                                        [us]
  droop_settled     idle - median level during the load (0.3 ms after onset .. 0.3 ms before end) [V]
  drop_cycle_min    idle - min over the whole cycle                                             [V]
  v2_*              the same for the second channel; lane_idle_mV / lane_load_mV / lane_max_mV = CH(V2) - CH(V)
                    at idle, averaged during the load, and the maximum instantaneous difference (20 us smoothed)
  scope_i_* / scope_p_*   when a current probe is recorded (1 us): idle / max in the first 300 us (di/dt step in A) / cycle max /
                    load median current, and P = V x I max / load / idle
  pmd_p_max / pmd_i_max   PMD2X peak power / current (PCIE1 + PCIE2 cables, 1 ms)
  clk_kernel        median in-kernel SM clock (freqmon) during the cycle                        [MHz]
  clk_smi           nvidia-smi clocks.current.sm max during the cycle (100 ms; the clock column when freqmon is off)
  edge_triggered    scope edge trigger fired (else the software force was used; both are onset-aligned)
"""
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from onset import load_csv, onset_from_drop, smooth, rise_time, max_slope


def cycle_metrics(run, kb, M, S):
    f = os.path.join(run, f"burst_{kb:02d}.csv"); r = load_csv(f)
    t, v = r["t_s"], r["v_V"]; v2 = r["v2_V"] if "v2_V" in r.dtype.names else None
    meta = {}
    try: meta = json.loads(str(np.load(f.replace(".csv", ".npz"))["meta"]))
    except Exception: pass
    s0, idle, end = onset_from_drop(t, v); tt = t - s0; load_s = (end - s0) if end else M["on_ms"] * 1e-3
    pre = tt < -0.5e-3; w = (tt >= 0) & (tt <= 300e-6); cyc = (tt >= 0) & (tt <= load_s); on = (tt > 0.3e-3) & (tt < load_s - 0.3e-3)
    mk = M["marks"][kb]; kms = mk.get("gpu_ms") or ((mk["t_end"] - mk["t_start"]) * 1e3 if mk.get("t_end") and mk.get("t_start") else None)   # host time without the GPU stamp
    d = dict(cycle=kb, kernel_ms=kms, v_idle=float(idle),
             didt_drop_300us=float(idle - v[w].min()) if w.sum() else None, t_min_us=float(tt[w][v[w].argmin()] * 1e6) if w.sum() else None,
             droop_settled=float(idle - np.median(v[on])) if on.sum() > 50 else None, drop_cycle_min=float(idle - v[cyc].min()) if cyc.sum() else None,
             edge_triggered=meta.get("edge_triggered"), onset_from_trigger_ms=(meta.get("onset_s") or 0) * 1e3)
    if v2 is not None and pre.sum() > 50:
        i2 = float(np.median(v2[pre])); dd = v2 - v; ds = smooth(v2, 20) - smooth(v, 20)
        d.update(v2_idle=i2, v2_didt_drop_300us=float(i2 - v2[w].min()) if w.sum() else None, v2_droop_settled=float(i2 - np.median(v2[on])) if on.sum() > 50 else None,
                 lane_idle_mV=float(dd[pre].mean() * 1e3), lane_load_mV=float(dd[on].mean() * 1e3) if on.sum() > 50 else None,
                 lane_max_mV=float(ds.max() * 1e3), lane_max_at_ms=float(tt[ds.argmax()] * 1e3))
    if "i_A" in r.dtype.names and np.isfinite(r["i_A"]).all():
        i, pw = r["i_A"], r["p_W"]; i_idle = float(np.median(i[pre])) if pre.sum() > 50 else float(i[:50].mean())
        d.update(scope_i_idle=i_idle, scope_i_max_300us=float(i[w].max()) if w.sum() else None, scope_didt_A_300us=float(i[w].max() - i_idle) if w.sum() else None,
                 scope_i_max=float(i[cyc].max()) if cyc.sum() else None, scope_i_load=float(np.median(i[on])) if on.sum() > 50 else None,
                 scope_p_max=float(pw[cyc].max()) if cyc.sum() else None, scope_p_load=float(np.median(pw[on])) if on.sum() > 50 else None,
                 scope_p_idle=float(np.median(pw[pre])) if pre.sum() > 50 else None)
        rt = rise_time(tt, i, 0.0)                       # current rise at the kernel start (tt = 0 at the voltage onset)
        if rt:
            d.update(scope_i_level_A=rt["level"], scope_i_rise_start_us=(rt["t_start"] * 1e6) if rt["t_start"] is not None else None,
                     scope_i_t10_us=(rt["t10"] * 1e6) if rt["t10"] is not None else None, scope_i_t90_us=(rt["t90"] * 1e6) if rt["t90"] is not None else None,
                     scope_i_rise_10_90_us=(rt["rise_s"] * 1e6) if rt["rise_s"] is not None else None,
                     scope_didt_A_per_us=(rt["step"] * 0.8 / (rt["rise_s"] * 1e6)) if rt["rise_s"] else None)
        sl, at = max_slope(tt, i, 0.0)
        if sl is not None: d.update(scope_didt_max_A_per_us=sl / 1e6, scope_didt_max_at_us=at * 1e6)
        # peaks as a function of the averaging window (the PMD2X samples at 1 ms): 10 us / 100 us / 1 ms boxcar means over the cycle
        dts = float(np.median(np.diff(tt[:1000]))) if len(tt) > 1 else 1e-6
        for wlab, wsec in (("10us", 10e-6), ("100us", 100e-6), ("1ms", 1e-3)):
            k = max(1, int(round(wsec / dts))); ic = i[cyc]; pc = pw[cyc]; m_ = (len(ic) // k) * k
            if m_ >= k:
                d[f"scope_i_max_{wlab}"] = float(ic[:m_].reshape(-1, k).mean(1).max()); d[f"scope_p_max_{wlab}"] = float(pc[:m_].reshape(-1, k).mean(1).max())
    pm = (S.get("pmd") or {}).get("per_burst") or []; pb = pm[kb] if kb < len(pm) and pm[kb] else {}
    d.update(pmd_p_max=pb.get("p_max"), pmd_i_max=((pb.get("i1_max") or 0) + (pb.get("i2_max") or 0)) if pb else None, pmd_p_idle=(S.get("pmd") or {}).get("idle_p"))
    d["clk_smi"] = smi_clock(run, M["marks"][kb])
    if os.path.exists(os.path.join(run, "freqmon.csv")) and M["marks"][kb].get("gt_before_ns"):
        F = load_csv(os.path.join(run, "freqmon.csv"))
        rel = (F["t_gpu_ns"] - M["marks"][kb]["gt_before_ns"]) / 1e9; sel = (rel > 0) & (rel < M["on_ms"] * 1e-3)
        d["clk_kernel"] = float(np.median(F["sm_clock_MHz"][sel])) if sel.sum() else None
        d["clk_min"] = float(F["sm_clock_MHz"][sel].min()) if sel.sum() else None
    return d


def smi_clock(run, mark):
    """nvidia-smi clocks.current.sm (100 ms samples, wall clock) max during the cycle: the only clock when freqmon is off."""
    import datetime, csv
    fs = sorted(glob.glob(os.path.join(run, "gpu_*.csv")))
    if not fs or not mark.get("wall_start"): return None
    try:
        out = []
        for row in csv.reader(open(fs[-1])):
            if len(row) < 2 or row[0].startswith("timestamp"): continue
            try: ts = datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp(); c = float(row[1].strip().split()[0])
            except Exception: continue
            if mark["wall_start"] <= ts <= mark["wall_end"] + 0.15: out.append(c)     # 100 ms samples: the cycle and the boost right after it
        return float(max(out)) if out else None
    except Exception:
        return None


def main():
    d = sys.argv[1].rstrip("/"); rows = []
    for run in sorted(glob.glob(os.path.join(d, "spike_eval_*"))):
        try:
            M = json.load(open(os.path.join(run, "marks.json"))); S = json.load(open(os.path.join(run, "summary.json")))
        except Exception:
            continue
        tag = re.sub(r"^spike_eval_\d+_\d+_", "", os.path.basename(run))      # spike_eval_<date>_<time>_<tag>
        g = M.get("gemm") or {}; wv = M.get("wave") or {}
        for f in sorted(glob.glob(os.path.join(run, "burst_[0-9][0-9].csv"))):
            kb = int(os.path.basename(f)[6:8])
            if kb >= len(M["marks"]): continue
            m = dict(run=os.path.basename(run), tag=tag, dtype=g.get("dtype"), kernel=g.get("kernel"), M=g.get("M"), N=g.get("N"), K=g.get("K"),
                     launches=M.get("K"), waves=wv.get("waves"), ctas=wv.get("ctas"))
            m.update(cycle_metrics(run, kb, M, S)); rows.append(m)
    if not rows:
        print("no runs in", d); return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(os.path.join(d, "runs_summary.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows: f.write(",".join("" if r.get(k) is None else str(r[k]) for k in keys) + "\n")
    json.dump(rows, open(os.path.join(d, "runs_summary.json"), "w"), indent=1)
    # per-run medians
    tags = []
    for r in rows:
        if r["tag"] not in tags: tags.append(r["tag"])
    g = lambda x, fmt: (fmt % x) if x is not None else " " * (len(fmt % 0) - 3) + "n/a"
    has_i = any(r.get("scope_i_max") is not None for r in rows)
    print(f"{'run':<28}{'n':>3}{'launch':>7}{'kern ms':>8}{'V idle':>7}{'dV300':>7}{'@us':>5}{'settle':>7}{'min':>7}" + (f"{'I idle':>7}{'dI300':>7}{'I max':>7}{'P max':>7}{'P load':>7}{'rise us':>8}" if has_i else f"{'V2 dV300':>9}{'lane load':>10}{'lane max':>9}") + f"{'PMD W':>7}{'PMD A':>6}{'clk':>6}{'edge':>5}")
    for tg in tags:
        rs = [r for r in rows if r["tag"] == tg]
        med = lambda k: (float(np.median([r[k] for r in rs if r.get(k) is not None])) if any(r.get(k) is not None for r in rs) else None)
        line = f"{tg:<28}{len(rs):>3}{rs[0]['launches'] or 1:>7}{g(med('kernel_ms'),'%8.2f')}{g(med('v_idle'),'%7.3f')}{g(med('didt_drop_300us'),'%7.3f')}{g(med('t_min_us'),'%5.0f')}"
        line += f"{g(med('droop_settled'),'%7.3f')}{g(med('drop_cycle_min'),'%7.3f')}"
        if has_i:
            line += f"{g(med('scope_i_idle'),'%7.1f')}{g(med('scope_didt_A_300us'),'%7.1f')}{g(med('scope_i_max'),'%7.1f')}{g(med('scope_p_max'),'%7.0f')}{g(med('scope_p_load'),'%7.0f')}{g(med('scope_i_rise_10_90_us'),'%8.0f')}"
        else:
            line += f"{g(med('v2_didt_drop_300us'),'%9.3f')}{g(med('lane_load_mV'),'%10.0f')}{g(med('lane_max_mV'),'%9.0f')}"
        line += f"{g(med('pmd_p_max'),'%7.1f')}{g(med('pmd_i_max'),'%6.1f')}{g(med('clk_kernel') if med('clk_kernel') is not None else med('clk_smi'),'%6.0f')}{sum(1 for r in rs if r.get('edge_triggered')):>5}"
        print(line)
    print("->", os.path.join(d, "runs_summary.{csv,json}"))


if __name__ == "__main__":
    main()
