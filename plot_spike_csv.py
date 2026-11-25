"""
plot_spike_csv.py — per-burst voltage/clock figures from a spike_eval run folder.

Inputs (written by spike_eval.py): marks.json, burst_NN.csv (scope, 1 us), pmd.csv (PMD2X ~1 ms), gpu_*.csv
(nvidia-smi), freqmon.csv (in-kernel SM clock, 10 us, GPU globaltimer).
Time axis of every figure: t = 0 at the kernel start of that burst.
    scope      : onset of the voltage drop (first departure from the pre-trigger idle level)
    PMD2X      : first row whose cable current rises above idle (+/-1 ms)
    freqmon    : %globaltimer stamp written on the GPU just before the first launch (+ ~10 us launch gap)
    nvidia-smi : wall-clock timestamp relative to the host launch time (100 ms samples, coarse)
Dotted vertical lines = kernel launch boundaries (start, end, and K-1 inner boundaries when K launches were queued).

Per burst:
    burst_NN_V_scope.png         scope rail voltage, whole record
    burst_NN_V_scope_0-300us.png scope rail voltage 0..300 us at 1 us/sample (+ freqmon clock on the right axis)
    burst_NN_V_pmd.png           PMD2X rail voltages (zero-order hold)
    burst_NN_V_overlay.png       scope V + PMD V (left axis) + nvidia-smi SM clock + freqmon SM clock (right axis)

Usage: ./venv/bin/python plot_spike_csv.py spike_eval_<stamp>/ [--rail PCIE1|PCIE2|both]
"""
import sys, os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import argparse, csv, glob, json, os, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

SCOPE_NAME = "Keysight DSOX1202A"       # default; overridden per run from marks.json (args.meter) / the record meta (idn)


def scope_name(M, meta):
    """Instrument name for legends: from the record's *IDN? when present, else from the meter argument of the run."""
    idn = str(meta.get("idn", "") or "")
    if "HDO" in idn.upper() or "LECROY" in idn.upper(): return "LeCroy " + (idn.split(",")[1].strip() if idn.count(",") >= 2 else "HDO4034A")
    if "DSOX" in idn.upper() or "KEYSIGHT" in idn.upper(): return "Keysight " + (idn.split(",")[1].strip() if idn.count(",") >= 2 else "DSOX1202A")
    m = (M.get("args") or {}).get("meter")
    return "LeCroy HDO4034A" if m == 2 else SCOPE_NAME


def load_pmd(path):
    r = csv.reader(open(path)); h = next(r); rows = [x for x in r if len(x) == len(h)]
    a = np.array(rows, dtype=float); c = {k: i for i, k in enumerate(h)}
    return a, c


def load_smi(path):
    """nvidia-smi csv -> (wall_epoch[s], sm_clock[MHz], power_instant[W])."""
    rows = list(csv.reader(open(path)))
    if len(rows) < 2:
        return None
    h = [x.strip() for x in rows[0]]
    ci = [k for k, x in enumerate(h) if x.startswith("clocks.current.sm")][0]
    cp = [k for k, x in enumerate(h) if x.startswith("power.draw.instant")][0]
    t, f, p = [], [], []
    for r in rows[1:]:
        try:
            ts = datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            f.append(float(r[ci].strip().split()[0])); p.append(float(r[cp].strip().split()[0])); t.append(ts)
        except Exception:
            pass
    return np.array(t), np.array(f), np.array(p)


def _smooth(x, n):
    """moving mean without edge artifacts (edges are padded with the edge value)."""
    xp = np.concatenate([np.full(n, x[:n].mean()), x, np.full(n, x[-n:].mean())])
    return np.convolve(xp, np.ones(n) / n, mode="same")[n:-n]


from onset import load_csv, onset_from_drop   # robust onset (smoothed, MAD sigma; drops of a few tens of mV)


def vlines(ax, t0, t1, K, scale=1e3):
    ax.axvline(t0 * scale, color="k", ls=":", lw=1.2, label="kernel launch boundary")
    if t1 is not None:
        ax.axvline(t1 * scale, color="k", ls=":", lw=1.2)
        for k in range(1, K):
            ax.axvline((t0 + (t1 - t0) * k / K) * scale, color="k", ls=":", lw=0.8, alpha=0.6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder"); ap.add_argument("--rail", default="both", choices=["PCIE1", "PCIE2", "both"])
    a = ap.parse_args()
    d = a.folder.rstrip("/")
    M = json.load(open(os.path.join(d, "marks.json"))); K = int(M.get("K", 1)); marks = M["marks"]
    rails = ["PCIE1", "PCIE2"] if a.rail == "both" else [a.rail]
    pmd, c = load_pmd(os.path.join(d, "pmd.csv")) if os.path.exists(os.path.join(d, "pmd.csv")) else (None, None)
    gpu = sorted(glob.glob(os.path.join(d, "gpu_*.csv"))); smi = load_smi(gpu[-1]) if gpu else None
    fq = load_csv(os.path.join(d, "freqmon.csv")) if os.path.exists(os.path.join(d, "freqmon.csv")) else None
    summary = []
    for f in sorted(glob.glob(os.path.join(d, "burst_[0-9][0-9].csv"))):
        kb = int(os.path.basename(f)[6:8]); m = marks[kb] if kb < len(marks) else None
        rec = load_csv(f); t_raw, v = rec["t_s"], rec["v_V"]
        v2 = rec["v2_V"] if "v2_V" in rec.dtype.names else None
        meta = {}
        try: meta = json.loads(str(np.load(f.replace(".csv", ".npz"))["meta"]))
        except Exception: pass
        ch1 = meta.get("v_ch", 2); ch2n = meta.get("v2_ch")
        dt_us = float(np.median(np.diff(t_raw)) * 1e6)
        s0, v_idle, s_end = onset_from_drop(t_raw, v)
        t = t_raw - s0                                     # 0 = kernel start on the scope
        load_s = (s_end - s0) if s_end is not None else None
        title = f"{os.path.basename(d)}  burst {kb}  (K={K} launch/burst)"
        n1 = meta.get("v_name") or "rail voltage"; n2 = meta.get("v2_name") or "rail voltage (2nd channel)"
        sname = scope_name(M, meta)
        lab_scope = f"{sname} CH{ch1} {n1}, 1 sample = {dt_us:.2f} us"
        lab_v2 = f"{sname} CH{ch2n} {n2}, 1 sample = {dt_us:.2f} us" if v2 is not None else None

        # ---------- scope, whole record
        fig, ax = plt.subplots(figsize=(13, 4.2))
        ax.plot(t * 1e3, v, lw=0.5, color="#1f77b4", label=lab_scope)
        if v2 is not None: ax.plot(t * 1e3, v2, lw=0.5, color="#ff7f0e", alpha=0.8, label=lab_v2)
        vlines(ax, 0.0, load_s, K)
        ax.set_xlabel("time from kernel start (ms)"); ax.set_ylabel("12 V rail voltage (V)"); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
        ax.set_title(title + (f"   load {load_s*1e3:.1f} ms   idle {v_idle:.3f} V   min {v.min():.3f} V" if load_s else ""))
        plt.tight_layout(); plt.savefig(f.replace(".csv", "_V_scope.png"), dpi=120); plt.close(fig)
        if v2 is not None:
            # second channel alone + difference
            fig, (axa, axb) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
            axa.plot(t * 1e3, v2, lw=0.5, color="#ff7f0e", label=lab_v2); vlines(axa, 0.0, load_s, K)
            axa.set_ylabel("12 V rail voltage (V)"); axa.grid(alpha=0.3); axa.legend(loc="lower right", fontsize=8)
            axa.set_title(title + f"   CH{ch2n}: idle {v2[t_raw < -0.5e-3].mean():.3f} V  min {v2.min():.3f} V")
            dvv = v2 - v; axb.plot(t * 1e3, dvv, lw=0.5, color="#7f7f7f", label=f"CH{ch2n} ({n2}) - CH{ch1} ({n1})"); vlines(axb, 0.0, load_s, K)
            axb.set_ylabel("difference (V)"); axb.set_xlabel("time from kernel start (ms)"); axb.grid(alpha=0.3); axb.legend(loc="lower right", fontsize=8)
            axb.set_title(f"CH{ch2n} ({n2}) - CH{ch1} ({n1}): mean {dvv.mean()*1e3:+.1f} mV, during load {dvv[(t > 1e-3) & (t < (load_s or t[-1]) - 1e-3)].mean()*1e3:+.1f} mV, std {dvv.std()*1e3:.1f} mV")
            plt.tight_layout(); plt.savefig(f.replace(".csv", f"_V_scope_ch{ch2n}.png"), dpi=120); plt.close(fig)

        # ---------- freqmon for this burst (GPU time -> t via the stamp)
        fq_t = fq_f = None
        if fq is not None and m is not None and m.get("gt_before_ns") is not None:
            tt = (fq["t_gpu_ns"] - m["gt_before_ns"]) / 1e9
            sel = (tt > -0.02) & (tt < (t[-1] + 0.02))
            fq_t, fq_f = tt[sel], fq["sm_clock_MHz"][sel]

        # ---------- scope 0..300 us at 1 us
        fig, ax = plt.subplots(figsize=(13, 4.6))
        sel = (t >= -20e-6) & (t <= 300e-6)
        ax.plot(t[sel] * 1e6, v[sel], lw=1.0, color="#1f77b4", marker=".", ms=3, label=lab_scope)
        if v2 is not None: ax.plot(t[sel] * 1e6, v2[sel], lw=1.0, color="#ff7f0e", marker=".", ms=3, alpha=0.8, label=lab_v2)
        ax.axvline(0, color="k", ls=":", lw=1.2, label="kernel start (voltage onset)")
        ax.set_xlim(-20, 300); ax.xaxis.set_major_locator(MultipleLocator(20)); ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(which="major", alpha=0.4); ax.grid(which="minor", alpha=0.12)
        ax.set_xlabel("time from kernel start (us)   [major grid 20 us, minor grid 1 us = one sample]"); ax.set_ylabel("12 V rail voltage (V)")
        if fq_t is not None:
            ax2 = ax.twinx(); fs = (fq_t >= -20e-6) & (fq_t <= 300e-6)
            ax2.plot(fq_t[fs] * 1e6, fq_f[fs], color="#9467bd", lw=1.0, marker="s", ms=3, label="freqmon SM clock (in-kernel, 10 us)")
            ax2.set_ylabel("SM clock (MHz)", color="#9467bd"); ax2.legend(loc="lower right", fontsize=8)
        ax.legend(loc="upper right", fontsize=8); ax.set_title(title + f"   first 300 us,  min {v[sel].min():.3f} V at {t[sel][v[sel].argmin()]*1e6:.0f} us")
        plt.tight_layout(); plt.savefig(f.replace(".csv", "_V_scope_0-300us.png"), dpi=120); plt.close(fig)

        # ---------- PMD for this burst
        tp_al = None
        if pmd is not None and m is not None:
            tm = pmd[:, c["t_mono"]]; i_sum = pmd[:, c["PCIE1_A"]] + pmd[:, c["PCIE2_A"]]
            sel = (tm >= m["t_start"] - 0.3) & (tm <= m["t_end"] + 0.3)
            if sel.sum() > 5:
                tp = tm[sel] - m["t_start"]; ip = i_sum[sel]
                idle_i = np.median(ip[tp < -0.05]) if (tp < -0.05).sum() else ip[:5].mean()
                rise = np.where(ip > idle_i + 2.0)[0]
                p0 = tp[rise[0]] if len(rise) else 0.0
                tp_al = tp - p0                                   # 0 = current edge (= kernel start, +/-1 ms)
                fig, ax = plt.subplots(figsize=(13, 4.2))
                for rail, col in zip(rails, ("#d62728", "#2ca02c")):
                    ax.step(tp_al * 1e3, pmd[sel][:, c[f"{rail}_V"]], where="post", lw=1.0, color=col, label=f"ElmorLabs PMD2X {rail}_V (1 ms poll, hold)")
                vlines(ax, 0.0, load_s, K)
                ax.set_xlim(t[0] * 1e3, t[-1] * 1e3); ax.set_xlabel("time from kernel start (ms)   [PMD aligned on its current edge, +/-1 ms]"); ax.set_ylabel("12 V rail voltage (V)")
                ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8); ax.set_title(title + "   PMD2X")
                plt.tight_layout(); plt.savefig(f.replace(".csv", "_V_pmd.png"), dpi=120); plt.close(fig)

        # ---------- overlay: scope V + PMD V | nvidia-smi clock + freqmon clock
        fig, ax = plt.subplots(figsize=(13, 5.2))
        ax.plot(t * 1e3, v, lw=0.5, color="#1f77b4", alpha=0.9, label=lab_scope)
        if v2 is not None: ax.plot(t * 1e3, v2, lw=0.5, color="#ff7f0e", alpha=0.8, label=lab_v2)
        if tp_al is not None:
            for rail, col in zip(rails, ("#d62728", "#2ca02c")):
                ax.step(tp_al * 1e3, pmd[sel][:, c[f"{rail}_V"]], where="post", lw=1.2, color=col, label=f"PMD2X {rail}_V (1 ms, hold)")
        vlines(ax, 0.0, load_s, K)
        ax.set_xlim(t[0] * 1e3, t[-1] * 1e3); ax.set_xlabel("time from kernel start (ms)"); ax.set_ylabel("12 V rail voltage (V)"); ax.grid(alpha=0.3)
        ax2 = ax.twinx(); have2 = False
        if fq_t is not None and len(fq_t):
            ax2.plot(fq_t * 1e3, fq_f, color="#9467bd", lw=0.4, alpha=0.5, label="freqmon SM clock (10 us raw)")
            nb = 50; nn = len(fq_f) // nb
            if nn > 2:
                ax2.plot(fq_t[:nn * nb].reshape(nn, nb).mean(1) * 1e3, np.median(fq_f[:nn * nb].reshape(nn, nb), 1), color="#5c2d91", lw=1.3, label="freqmon SM clock (0.5 ms median)")
            have2 = True
        if smi is not None and m is not None:
            ts = smi[0] - m["wall_start"]; ss = (ts > t[0] - 0.2) & (ts < t[-1] + 0.2)
            if ss.sum():
                ax2.step(ts[ss] * 1e3, smi[1][ss], where="post", color="#8c564b", lw=1.5, marker="o", ms=4, label="nvidia-smi clocks.current.sm (100 ms, wall-clock aligned)")
                have2 = True
        if have2:
            ax2.set_ylabel("SM clock (MHz)"); lo = min(np.nanmin(fq_f) if fq_t is not None and len(fq_f) else 3000, 2400); ax2.set_ylim(lo - 50, 2950)
            ax2.legend(loc="lower right", fontsize=8)
        ax.legend(loc="upper right", fontsize=8); ax.set_title(title + "   overlay: scope V, PMD2X V, nvidia-smi clock, freqmon clock")
        plt.tight_layout(); plt.savefig(f.replace(".csv", "_V_overlay.png"), dpi=120); plt.close(fig)

        on = (t > 0.3e-3) & (t < (load_s or t[-1]) - 0.3e-3)
        summary.append(dict(burst=kb, load_ms=load_s * 1e3 if load_s else None, host_ms=m["on_ms"] if m else None,
                            gpu_ms=m.get("gpu_ms") if m else None, scope_idle=v_idle, scope_min=float(v.min()),
                            scope_min_us=float(t[v.argmin()] * 1e6), scope_settled=float(np.median(v[on])) if on.sum() else None,
                            v2_idle=float(v2[t_raw < -0.5e-3].mean()) if v2 is not None else None, v2_min=float(v2.min()) if v2 is not None else None,
                            v2_minus_v_mV=float((v2 - v)[on].mean() * 1e3) if (v2 is not None and on.sum()) else None,
                            fq_min=float(fq_f.min()) if fq_t is not None and len(fq_f) else None,
                            fq_idle=float(np.median(fq_f[fq_t < 0])) if fq_t is not None and (fq_t < 0).sum() else None))
    has2 = any(s.get("v2_idle") is not None for s in summary)
    print(f"{'burst':>5}{'load ms':>8}{'host ms':>8}{'gpu ms':>7}{'V idle':>8}{'V min':>8}{'@us':>7}{'V settled':>10}{'clk idle':>9}{'clk min':>8}" + ("{:>9}{:>8}{:>12}".format("V2 idle", "V2 min", "V2-V1 mV") if has2 else ""))
    for s in summary:
        g = lambda x, w, fmt: (fmt % x) if x is not None else " " * (w - 3) + "n/a"
        print(f"{s['burst']:5d}{g(s['load_ms'],8,'%8.1f')}{g(s['host_ms'],8,'%8.1f')}{g(s['gpu_ms'],7,'%7.1f')}{s['scope_idle']:8.3f}{s['scope_min']:8.3f}"
              f"{s['scope_min_us']:7.0f}{g(s['scope_settled'],10,'%10.3f')}{g(s['fq_idle'],9,'%9.0f')}{g(s['fq_min'],8,'%8.0f')}"
              + (f"{g(s['v2_idle'],9,'%9.3f')}{g(s['v2_min'],8,'%8.3f')}{g(s['v2_minus_v_mV'],12,'%+12.1f')}" if has2 else ""))
    json.dump(summary, open(os.path.join(d, "voltage_plots_summary.json"), "w"), indent=1)
    print("plots ->", d)


if __name__ == "__main__":
    main()
