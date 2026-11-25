#!/usr/bin/env python
"""overlay_summary.py <result_dir> [--param clk|waves] [--out png]
One PNG: top row = CH(V) (PSU cable), bottom row = CH(V2) (PCIe slot); every cycle of every run in the folder is overlaid,
aligned on the voltage onset (t=0 = kernel start). Colour = the swept parameter (07: SM clock during the kernel from freqmon,
08: waves), opacity = cycle index (first cycle opaque, repeats lighter). Rows = the four x windows of burst_NN.png (1 us / 10 us / 100 us / 1 ms ticks), columns = the recorded signals: CH(V), CH(V2), current, P = V x I."""
import argparse, glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from onset import load_csv, onset_from_drop
_argv = sys.argv; sys.argv = [_argv[0]]; import spike_eval as se; sys.argv = _argv      # plot helpers (cap line, scope name)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib import cm, colors


def _nice(x):
    """1-2-5 tick step >= x (in the axis unit), so long windows get <= ~24 round labels."""
    import math
    if x <= 1: return 1
    m = 10 ** math.floor(math.log10(x)); return m * (1 if x <= m else 2 if x <= 2 * m else 5 if x <= 5 * m else 10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir"); ap.add_argument("--param", choices=["clk", "waves", "auto"], default="auto"); ap.add_argument("--out", default=None)
    ap.add_argument("--full-ms", type=float, default=None, help="x range of the last window (default: longest record)")
    ap.add_argument("--highlight", type=float, default=None, help="draw the setting with this value in red (default: 1.0 for waves)")
    ap.add_argument("--k", type=int, default=None, help="waves: only runs with this GEMM K (one figure per K; different K = different cycle length)")
    ap.add_argument("--signal", choices=["voltage", "freq", "combined"], default="voltage",
                    help="voltage: CH(V)/CH(V2)/I/P scope traces (4 windows); freq: the in-kernel freqmon SM clock (4 windows); "
                         "combined: ONE x window (--combined-window, default the 100 us-tick window = -0.2 .. +2 ms), freqmon clock on top and every scope signal stacked below (shared x)")
    ap.add_argument("--combined-window", choices=["1us", "10us", "100us", "1ms"], default="100us", help="which of the four burst_NN.png windows the combined figure uses")
    a = ap.parse_args(); d = a.result_dir.rstrip("/")
    runs = sorted(glob.glob(os.path.join(d, "spike_eval_*")))
    param = a.param
    if param == "auto": param = "waves" if any("_t08_" in r for r in runs) else "clk"
    if param == "clk" and any(("_t03_clk_" in r or "_t07_clk_" in r) for r in runs): runs = [r for r in runs if ("_t03_clk_" in r or "_t07_clk_" in r)]   # run_all folders hold every test
    items = []
    for run in runs:
        if not os.path.exists(os.path.join(run, "marks.json")): continue
        M = json.load(open(os.path.join(run, "marks.json"))); V = json.load(open(os.path.join(run, "voltage_plots_summary.json"))) if os.path.exists(os.path.join(run, "voltage_plots_summary.json")) else []
        if param == "waves":
            if not M.get("wave"): continue
            if re.search(r"cta_(square|row|column|rows2)$", run): continue        # 04b tile-layout runs are not sweep points
            if a.k is not None and int(M["gemm"]["K"]) != a.k: continue
            val = M["wave"]["waves"]; lab = f"{val:.2f} waves ({M['wave']['ctas']} CTA, M={M['gemm']['M']} N={M['gemm']['N']} K={M['gemm']['K']})"
        else:
            tag = re.split(r"_t0[37]_clk_", run)[-1] if ("_t03_clk_" in run or "_t07_clk_" in run) else os.path.basename(run)
            if os.path.exists(os.path.join(run, "freqmon.csv")) and M["marks"][0].get("gt_before_ns"):
                F = load_csv(os.path.join(run, "freqmon.csv")); m0 = M["marks"][0]
                rel = (F["t_gpu_ns"] - m0["gt_before_ns"]) / 1e9; sel = (rel > 0.002) & (rel < M["on_ms"] * 1e-3)
                val = float(np.median(F["sm_clock_MHz"][sel])) if sel.sum() else 0.0
                lab = f"{tag}: {val:.0f} MHz in kernel"
            else:                                                   # no freqmon (FREQMON=0): lock target from the tag, else nvidia-smi max
                mm = re.search(r"_(\d+)MHz", tag); val = float(mm.group(1)) if mm else None
                if val is None:
                    try:
                        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import summary_runs
                        val = summary_runs.smi_clock(run, M["marks"][0]) or 0.0
                    except Exception: val = 0.0
                lab = f"{tag}: {val:.0f} MHz ({'lock target' if mm else 'nvidia-smi max'}, no freqmon)"
        recs = []
        for f in sorted(glob.glob(os.path.join(run, "burst_[0-9][0-9].csv"))):
            r = load_csv(f)
            s0, idle, end = onset_from_drop(r["t_s"], r["v_V"])
            sig = {"V": r["v_V"]}
            if "v2_V" in r.dtype.names: sig["V2"] = r["v2_V"]
            if "i_A" in r.dtype.names and np.isfinite(r["i_A"]).all(): sig["I"] = r["i_A"]; sig["P"] = r["p_W"]
            recs.append((r["t_s"] - s0, sig, idle))
        meta = {}
        try: meta = json.loads(str(np.load(glob.glob(os.path.join(run, "burst_[0-9][0-9].npz"))[0])["meta"]))
        except Exception: pass
        fqs = []
        if a.signal in ("freq", "combined") and os.path.exists(os.path.join(run, "freqmon.csv")):
            F = load_csv(os.path.join(run, "freqmon.csv"))
            for m in M["marks"]:
                if m.get("gt_before_ns") is None: continue
                rel = (F["t_gpu_ns"] - m["gt_before_ns"]) / 1e9; sel = (rel > -0.02) & (rel < m.get("gpu_ms", M["on_ms"]) * 1e-3 + 0.02)
                if sel.sum(): fqs.append((rel[sel], F["sm_clock_MHz"][sel], 0.0))
        if a.signal == "freq": recs = fqs
        if recs: items.append(dict(val=val, lab=lab, recs=recs, fqs=fqs, meta=meta, run=os.path.basename(run), wave=M.get("wave")))
    if not items: sys.exit("no runs with two scope channels in " + d)
    # drop cycles whose pre-onset level is not idle (a forced trigger that landed inside the load: the onset detector has no idle reference)
    if a.signal in ("voltage", "combined"):
        g_idle = float(np.median([r[2] for x in items for r in x["recs"]])); dropped = 0
        for x in items:
            keep = [r for r in x["recs"] if r[2] > g_idle - 0.05]; dropped += len(x["recs"]) - len(keep); x["recs"] = [r[:2] for r in keep]
        items = [x for x in items if x["recs"]]
        if dropped: print(f"dropped {dropped} cycle(s) without an idle reference before the onset")
    items.sort(key=lambda x: x["val"])
    # continuous colour = swept value (viridis + colour bar). Settings whose values are (almost) the same, e.g. default / max / 0.9
    # that all run at ~2650 MHz, get the same hue but different line styles and a lighter/darker shade so they stay apart.
    vals = [x["val"] for x in items]
    norm = colors.Normalize(min(vals), max(vals)) if param == "clk" else colors.LogNorm(min(vals), max(vals))
    cmap = plt.get_cmap("viridis"); styles = ["-", "--", ":", "-."]; grp = []
    for x in items:
        same = grp and abs(x["val"] - grp[-1]["val"]) <= (0.02 * max(vals) if param == "clk" else 0.02 * grp[-1]["val"])
        grp = grp + [x] if same else [x]
        j = len(grp) - 1; base = np.array(cmap(norm(x["val"])))
        x["ls"] = styles[j % len(styles)]
        x["color"] = tuple(np.clip(base[:3] * (1 - 0.18 * j) + np.array([1, 1, 1]) * 0.0, 0, 1)) + (1.0,) if j else tuple(base)
        if j: x["lab"] += f"  [{['solid', 'dashed', 'dotted', 'dash-dot'][j % 4]}, same clock as above]"
    hl = a.highlight if a.highlight is not None else (1.0 if param == "waves" else None)
    for x in items:
        x["hl"] = hl is not None and abs(x["val"] - hl) < 1e-3
        if x["hl"]:
            x["color"] = (0.85, 0.0, 0.0, 1.0); x["ls"] = "-"
            w = x.get("wave") or {}
            x["lab"] = "RED: " + x["lab"] + (f"  = exactly 1 wave: {w['ctas']} CTAs on {w['usable_sms']} usable SMs x {w['ctas_per_sm']} CTA/SM, no tail" if w else "")
    meta0 = items[0]["meta"]; n1 = f"CH{meta0.get('v_ch', 1)} {meta0.get('v_name', 'rail voltage')}"; n2 = f"CH{meta0.get('v2_ch', 2)} {meta0.get('v2_name', 'rail voltage (2nd)')}"
    try: i_ch = meta0.get("i_ch") or (json.load(open(os.path.join(d, items[0]["run"], "marks.json")))["args"].get("i_ch") or "?")
    except Exception: i_ch = "?"
    names = {"V": n1 + " (V)", "V2": n2 + " (V)", "I": f"current CH{i_ch} (A)", "P": "P = V x I (W)"}
    cap_w = se.power_cap_w(os.path.join(d, items[0]["run"]))                                    # board power limit for the P panels
    idn = str(meta0.get("idn", "")); scope_name = ("LeCroy HDO4034A" if "LECROY" in idn.upper() or "HDO" in idn.upper() else "Keysight DSOX1202A")
    cols = [k for k in ("V", "V2", "I", "P") if k in items[0]["recs"][0][1]]
    # layout like burst_NN.png: rows = x windows (1 us / 10 us / 100 us / 1 ms ticks), cols = CH(V), CH(V2)
    from matplotlib.ticker import MultipleLocator
    WINDOWS = [(1e-6, 20e-6), (10e-6, 200e-6), (100e-6, 2e-3), (1e-3, None)]
    ulab = {1e-6: "1 us", 10e-6: "10 us", 100e-6: "100 us", 1e-3: "1 ms"}
    tmax = max(r[0][-1] for x in items for r in x["recs"]) if a.full_ms is None else a.full_ms * 1e-3
    items.sort(key=lambda x: (x["hl"], x["val"]))          # highlighted setting drawn last = on top
    if a.signal == "combined":
        rows_ = ["F"] + cols; nr = len(rows_)
        fig, axs = plt.subplots(nr, 1, figsize=(15, 2.9 * nr), sharex=True)
        unit, span = {"1us": (1e-6, 20e-6), "10us": (10e-6, 200e-6), "100us": (100e-6, 2e-3), "1ms": (1e-3, None)}[a.combined_window]
        hi = tmax if span is None else span; lo = -2 * unit; scale = 1e6 if unit < 100e-6 else 1e3; xl = "us" if unit < 100e-6 else "ms"
        for x in items:
            c = x["color"]; n = len(x["recs"])
            for k, (t, sig) in enumerate(x["recs"]):
                al = 0.9 if n == 1 else max(0.25, 0.9 - 0.6 * k / max(1, n - 1)); lw = (0.6 if unit >= 100e-6 else 0.9) * (2.2 if x["hl"] else 1)
                sel = (t >= lo) & (t <= hi); step = max(1, int(sel.sum() / 20000)); tt = t[sel][::step] * scale
                for ri, key in enumerate(cols):
                    if key in sig: axs[ri + 1].plot(tt, sig[key][sel][::step], color=c, lw=lw, alpha=al, ls=x["ls"], label=x["lab"] if (k == 0 and ri == 0) else None)
            for k, (t, f, _) in enumerate(x.get("fqs", [])):
                al = 0.9 if n == 1 else max(0.25, 0.9 - 0.6 * k / max(1, n - 1)); sel = (t >= lo) & (t <= hi)
                if sel.sum(): axs[0].step(t[sel] * scale, f[sel], where="post", color=c, lw=(0.5 if unit >= 1e-3 else 0.9) * (2.2 if x["hl"] else 1), alpha=al, ls=x["ls"], marker="s" if unit <= 10e-6 else None, ms=3)
        axs[0].set_ylabel(f"freqmon SM clock (MHz)\n[{ulab[unit]} ticks]", fontsize=8); axs[0].set_title("in-kernel SM clock (freqmon, 10 us; t = 0 at the first-launch GPU stamp)", fontsize=10)
        for ri, key in enumerate(cols):
            axs[ri + 1].set_ylabel(f"{names[key]}\n[{ulab[unit]} ticks]", fontsize=8); axs[ri + 1].set_title(names[key] + " (t = 0 at the voltage onset)", fontsize=10)
            if key == "P": se._cap_line(axs[ri + 1], cap_w)
        for A in axs:
            A.axvline(0, color="k", ls="--", lw=0.7); A.grid(alpha=0.3); A.tick_params(labelsize=7); A.set_xlim(lo * scale, hi * scale)
            A.xaxis.set_major_locator(MultipleLocator(unit * scale if span is not None else _nice((hi - lo) * scale / 24)))
        axs[-1].set_xlabel(f"time from kernel start ({xl}); grid = {ulab[unit]}; scope 1 us samples, freqmon 10 us", fontsize=8)
        h, l = axs[1].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3 if len(items) > 8 else 2, fontsize=7.5, frameon=True, bbox_to_anchor=(0.45, 0.0))
        fig.subplots_adjust(top=0.90, bottom=0.11 + 0.014 * ((len(items) + 2) // 3 if len(items) > 8 else (len(items) + 1) // 2), left=0.07, right=0.90, hspace=0.35)
        cax = fig.add_axes([0.925, 0.25, 0.012, 0.5])
        sm = cm.ScalarMappable(norm=norm, cmap=cmap); cb = fig.colorbar(sm, cax=cax); cb.set_label("SM clock during the kernel (MHz)" if param == "clk" else "waves")
        fig.suptitle(f"{os.path.basename(d)}{(' K=%d' % a.k) if getattr(a, 'k', None) else ''}: {len(items)} settings x {sum(len(x['recs']) for x in items)} cycles, {ulab[unit]}-tick window: freqmon clock + scope signals\n"
                     f"colour = {'SM clock during the kernel' if param == 'clk' else 'waves'} (equal values -> solid / dashed / dotted), opacity = cycle index (first cycle opaque)", fontsize=11)
        out = a.out or os.path.join(d, f"overlay_{param}_combined.png"); plt.savefig(out, dpi=130); print("->", out); return
    if a.signal == "freq":
        fig, axs = plt.subplots(len(WINDOWS), 1, figsize=(15, 3.3 * len(WINDOWS)))
        for ri, (unit, span) in enumerate(WINDOWS):
            lo = -2 * unit; hi = tmax if span is None else span
            scale = 1e6 if unit < 100e-6 else 1e3; xl = "us" if unit < 100e-6 else "ms"
            A = axs[ri]
            for x in items:
                c = x["color"]; n = len(x["recs"])
                for k, (t, f, _) in enumerate(x["recs"]):
                    al = 0.9 if n == 1 else max(0.25, 0.9 - 0.6 * k / max(1, n - 1))
                    sel = (t >= lo - 10e-6) & (t <= hi)
                    if sel.sum() < 1: continue
                    A.step(t[sel] * scale, f[sel], where="post", color=c, lw=(0.9 if unit < 1e-3 else 0.5) * (2.2 if x["hl"] else 1), alpha=al, ls=x["ls"],
                           marker="s" if unit <= 10e-6 else None, ms=3, label=x["lab"] if (k == 0 and ri == 0) else None)
            A.axvline(0, color="k", ls="--", lw=0.7); A.grid(alpha=0.3); A.tick_params(labelsize=7)
            A.xaxis.set_major_locator(MultipleLocator(unit * scale if span is not None else _nice((hi - lo) * scale / 24)))
            A.set_xlim(lo * scale, hi * scale); A.set_ylabel(f"freqmon SM clock (MHz)\n[{ulab[unit]} ticks]", fontsize=8)
            A.set_xlabel(f"time from the first-launch GPU stamp ({xl}); freqmon sample = 10 us", fontsize=8)
            if ri == 0: A.set_title("in-kernel SM clock (freqmon, 10 us samples)", fontsize=11)
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3 if len(items) > 8 else 2, fontsize=7.5, frameon=True, bbox_to_anchor=(0.45, 0.0))
        fig.subplots_adjust(top=0.93, bottom=0.10 + 0.012 * ((len(items) + 2) // 3 if len(items) > 8 else (len(items) + 1) // 2), left=0.07, right=0.90, hspace=0.38)
        cax = fig.add_axes([0.925, 0.25, 0.012, 0.5])
        sm = cm.ScalarMappable(norm=norm, cmap=cmap); cb = fig.colorbar(sm, cax=cax); cb.set_label("SM clock during the kernel (MHz)" if param == "clk" else "waves")
        fig.suptitle(f"{os.path.basename(d)}{(' K=%d' % a.k) if getattr(a, 'k', None) else ''}: {len(items)} settings x {sum(len(x['recs']) for x in items)} cycles, freqmon in-kernel clock monitor (10 us), stamp-aligned\n"
                     f"colour = {'SM clock during the kernel' if param == 'clk' else 'waves'} (equal values -> solid / dashed / dotted), opacity = cycle index (first cycle opaque)", fontsize=11)
        out = a.out or os.path.join(d, f"overlay_{param}_freqmon.png"); plt.savefig(out, dpi=130); print("->", out); return
    # voltage / current / power: rows = x windows, one column per recorded signal (V, V2, I, P)
    nc = len(cols)
    fig, ax = plt.subplots(len(WINDOWS), nc, figsize=(8.5 * nc, 3.3 * len(WINDOWS)), squeeze=False)
    for ri, (unit, span) in enumerate(WINDOWS):
        lo = -2 * unit; hi = tmax if span is None else span
        scale = 1e6 if unit < 100e-6 else 1e3; xl = "us" if unit < 100e-6 else "ms"
        for x in items:
            c = x["color"]; n = len(x["recs"])
            for k, (t, sig) in enumerate(x["recs"]):
                al = 0.9 if n == 1 else max(0.25, 0.9 - 0.6 * k / max(1, n - 1))      # first cycle opaque, repeats lighter
                sel = (t >= lo) & (t <= hi)
                if sel.sum() < 2: continue
                step = max(1, int(sel.sum() / 20000)); tt = t[sel][::step] * scale
                mk = "." if unit == 1e-6 else None; lw = (0.6 if unit >= 100e-6 else 0.9) * (2.2 if x["hl"] else 1)
                for ci, key in enumerate(cols):
                    if key not in sig: continue
                    ax[ri, ci].plot(tt, sig[key][sel][::step], color=c, lw=lw, alpha=al, marker=mk, ms=2.5, ls=x["ls"], label=x["lab"] if (k == 0 and ri == 0 and ci == 0) else None)
        for ci, key in enumerate(cols):
            A = ax[ri, ci]; A.axvline(0, color="k", ls="--", lw=0.7); A.grid(alpha=0.3); A.tick_params(labelsize=7)
            if key == "P": se._cap_line(A, cap_w)
            A.xaxis.set_major_locator(MultipleLocator(unit * scale if span is not None else _nice((hi - lo) * scale / 24)))
            A.set_xlim(lo * scale, hi * scale); A.set_ylabel(f"{names[key]}\n[{ulab[unit]} ticks]", fontsize=8)
            A.set_xlabel(f"time from kernel start (voltage onset) ({xl}); 1 sample = 1 us", fontsize=8)
            if ri == 0: A.set_title(names[key], fontsize=11)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3 if len(items) > 8 else 2, fontsize=7.5, frameon=True, bbox_to_anchor=(0.45, 0.0))
    fig.subplots_adjust(top=0.93, bottom=0.10 + 0.012 * ((len(items) + 2) // 3 if len(items) > 8 else (len(items) + 1) // 2), left=0.06, right=0.90, hspace=0.38, wspace=0.22)
    cax = fig.add_axes([0.925, 0.25, 0.012, 0.5])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); cb = fig.colorbar(sm, cax=cax); cb.set_label("SM clock during the kernel (MHz)" if param == "clk" else "waves")
    fig.suptitle(f"{os.path.basename(d)}{(' K=%d' % a.k) if getattr(a, 'k', None) else ''}: {len(items)} settings x {sum(len(x['recs']) for x in items)} cycles, {scope_name} 1 us/sample, onset-aligned\n"
                 f"colour = {'SM clock during the kernel' if param == 'clk' else 'waves'} (equal values -> solid / dashed / dotted), opacity = cycle index (first cycle opaque)", fontsize=11)
    out = a.out or os.path.join(d, f"overlay_{param}.png"); plt.savefig(out, dpi=130); print("->", out)


if __name__ == "__main__":
    main()
