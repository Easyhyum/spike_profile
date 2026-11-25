#!/usr/bin/env python
"""
summary_clock.py <result_dir> — collect the t03_clk_* runs (test 03; legacy t07) of one result folder into
clock_sweep_summary.csv / .json / .png: per clock setting the di/dt voltage drop at the kernel start
(scope, 1 us: min within the first 300 us and the time of that minimum), the settled droop, the full-burst minimum,
PMD2X peak power/current and the clock the kernel really ran at (freqmon median during the burst).
"""
import re, glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from onset import load_csv, onset_from_drop


def clk_fallback(tag, run, kb):
    """No freqmon: nvidia-smi clock during the cycle (summary_runs.smi_clock), else the locked MHz in the tag."""
    try:
        import summary_runs, json
        M = json.load(open(os.path.join(run, "marks.json"))); c = summary_runs.smi_clock(run, M["marks"][kb])
        if c: return c
    except Exception:
        pass
    m = re.search(r"_(\d+)MHz", tag); return float(m.group(1)) if m else None


def main():
    d = sys.argv[1]
    rows = []
    for run in sorted(glob.glob(os.path.join(d, "spike_eval_*_t0[37]_clk_*"))):
        tag = re.split(r"_t0[37]_clk_", run)[-1]
        try:
            S = json.load(open(os.path.join(run, "summary.json")))
            V = json.load(open(os.path.join(run, "voltage_plots_summary.json")))
        except Exception:
            continue
        for f in sorted(glob.glob(os.path.join(run, "burst_[0-9][0-9].csv"))):
            kb = int(os.path.basename(f)[6:8])
            rec = load_csv(f); t, v = rec["t_s"], rec["v_V"]
            vs = [x for x in V if x["burst"] == kb]
            if not vs: continue
            vs = vs[0]
            # onset like plot_spike_csv: shift so that the voltage onset is t=0
            pre = t < -0.5e-3; s0, idle, _ = onset_from_drop(t, v); tt = t - s0
            w = (tt >= 0) & (tt <= 300e-6)
            pm = (S.get("pmd") or {}).get("per_burst") or []
            pb = pm[kb] if kb < len(pm) and pm[kb] else {}
            rows.append(dict(setting=tag, burst=kb, v_idle=float(idle), v_min_300us=float(v[w].min()) if w.sum() else None,
                             t_min_300us_us=float(tt[w][v[w].argmin()] * 1e6) if w.sum() else None,
                             didt_drop_300us=float(idle - v[w].min()) if w.sum() else None,
                             v_settled=vs.get("scope_settled"), v_min_burst=vs.get("scope_min"),
                             droop_settled=float(idle - vs["scope_settled"]) if vs.get("scope_settled") else None,
                             pmd_p_max=pb.get("p_max"), pmd_i_max=(pb.get("i1_max") or 0) + (pb.get("i2_max") or 0) if pb else None,
                             clk_kernel=vs.get("fq_min") or clk_fallback(tag, run, kb), clk_idle=vs.get("fq_idle"), host_ms=vs.get("host_ms"),
                             clk_source="freqmon" if vs.get("fq_min") else "nvidia-smi / lock target (no freqmon)",
                             run=os.path.basename(run)))
    if not rows:
        print("no t07 runs in", d); return
    keys = ["setting", "burst", "v_idle", "v_min_300us", "t_min_300us_us", "didt_drop_300us", "v_settled", "droop_settled", "v_min_burst",
            "pmd_p_max", "pmd_i_max", "clk_kernel", "clk_idle", "clk_source", "host_ms", "run"]
    with open(os.path.join(d, "clock_sweep_summary.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(k) is None else str(r[k]) for k in keys) + "\n")
    json.dump(rows, open(os.path.join(d, "clock_sweep_summary.json"), "w"), indent=1)
    print(f"{'setting':<22}{'clk kern':>9}{'V idle':>8}{'dV 300us':>9}{'@us':>6}{'dV settled':>11}{'V min':>8}{'PMD Pmax':>9}{'host ms':>8}")
    for r in rows:
        g = lambda x, fmt: (fmt % x) if x is not None else "   n/a"
        print(f"{r['setting']:<22}{g(r['clk_kernel'],'%9.0f')}{r['v_idle']:8.3f}{g(r['didt_drop_300us'],'%9.3f')}{g(r['t_min_300us_us'],'%6.0f')}{g(r['droop_settled'],'%11.3f')}{g(r['v_min_burst'],'%8.3f')}{g(r['pmd_p_max'],'%9.1f')}{g(r['host_ms'],'%8.1f')}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        x = [r["clk_kernel"] or 0 for r in rows]; lab = [r["setting"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].plot(x, [r["didt_drop_300us"] for r in rows], "o-", label="di/dt drop, first 300 us (idle - min)")
        ax[0].plot(x, [r["droop_settled"] for r in rows], "s--", label="settled droop (idle - settled)")
        for xi, r in zip(x, rows): ax[0].annotate(r["setting"], (xi, r["didt_drop_300us"] or 0), fontsize=7, rotation=30)
        ax[0].set_xlabel("SM clock during the kernel (MHz; freqmon, or nvidia-smi / lock target when freqmon is off)"); ax[0].set_ylabel("rail voltage drop (V)"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
        ax[1].plot(x, [r["pmd_p_max"] for r in rows], "o-", color="#d62728", label="PMD2X peak power (W)")
        try:
            import spike_eval as se; se._cap_line(ax[1], se.power_cap_w(os.path.join(d, rows[0]["run"])))
        except Exception: pass
        ax[1].set_xlabel("SM clock during the kernel (MHz)"); ax[1].set_ylabel("W"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
        fig.suptitle(f"clock sweep: {os.path.basename(d)}"); plt.tight_layout(); plt.savefig(os.path.join(d, "clock_sweep_summary.png"), dpi=120)
        print("->", os.path.join(d, "clock_sweep_summary.{csv,json,png}"))
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
