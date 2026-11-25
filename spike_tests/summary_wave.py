#!/usr/bin/env python
"""summary_wave.py <result_dir> — collect the t04_* runs (test 04; legacy t08): per wave target the di/dt drop (scope, idle - min in the
first 300 us), the time of that minimum, the whole-cycle minimum, the kernel time, PMD2X peak and the freqmon clock."""
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # summary_runs.smi_clock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from onset import load_csv, onset_from_drop


def main():
    d = sys.argv[1]; rows = []
    runs = [r for r in glob.glob(os.path.join(d, "spike_eval_*_t0[48]_*")) if os.path.exists(os.path.join(r, "marks.json"))]
    usable = None
    for r in runs:                                                         # usable SMs from any wavegemm run (for the cuBLAS reference rows)
        w_ = json.load(open(os.path.join(r, "marks.json"))).get("wave")
        if w_: usable = w_.get("usable_sms"); break
    usable = usable or int(os.environ.get("AUTO_USABLE_SMS") or 35)
    def wave_of(M):
        if M.get("wave"): return M["wave"]
        g = M["gemm"]; ctas = (g["M"] // 256) * (g["N"] // 128)             # cuBLAS fp32 8192^3 uses the same 256x128 tile, 1 CTA/SM
        return dict(waves=ctas / usable, ctas=ctas, tile=[256, 128], usable_sms=usable, kernel="cublas")
    for run in sorted(runs, key=lambda r: (json.load(open(os.path.join(r, "marks.json")))["gemm"]["K"], wave_of(json.load(open(os.path.join(r, "marks.json"))))["waves"])):
        try:
            M = json.load(open(os.path.join(run, "marks.json"))); S = json.load(open(os.path.join(run, "summary.json")))
        except Exception:
            continue
        wv = wave_of(M); g = M["gemm"]; Mm, Nn, Kk = g["M"], g["N"], g["K"]; kern = "cublas" if not M.get("wave") else "wave"
        if re.search(r"cta_(square|row|column|rows2)$", run): kern = "layout"     # 04b tile-layout runs: csv only, not part of the sweep statistics
        flops = 2.0 * Mm * Nn * Kk
        ai_ideal = flops / (4.0 * (Mm * Kk + Kk * Nn + Mm * Nn))                        # each operand touched once (perfect reuse)
        tmx, tnx = (wv.get("tile") or [128, 128])
        ai_tile = flops / (4.0 * (Mm * Kk * (Nn / tnx) + Kk * Nn * (Mm / tmx) + Mm * Nn))  # one tile per CTA, no L2 reuse (every CTA streams its A/B panels)
        for f in sorted(glob.glob(os.path.join(run, "burst_[0-9][0-9].csv"))):
            kb = int(os.path.basename(f)[6:8])
            rec = load_csv(f); t, v = rec["t_s"], rec["v_V"]
            v2 = rec["v2_V"] if "v2_V" in rec.dtype.names else None
            pre = t < -0.5e-3; s0, idle, _ = onset_from_drop(t, v); tt = t - s0
            w = (tt >= 0) & (tt <= 300e-6); cyc = (tt >= 0) & (tt <= M["on_ms"] * 1e-3)
            pm = (S.get("pmd") or {}).get("per_burst") or []; pb = pm[kb] if kb < len(pm) and pm[kb] else {}
            fq = None
            if os.path.exists(os.path.join(run, "freqmon.csv")) and M["marks"][kb].get("gt_before_ns"):
                F = load_csv(os.path.join(run, "freqmon.csv"))
                rel = (F["t_gpu_ns"] - M["marks"][kb]["gt_before_ns"]) / 1e9; sel = (rel > 0) & (rel < M["on_ms"] * 1e-3)
                fq = float(np.median(F["sm_clock_MHz"][sel])) if sel.sum() else None
            mk = M["marks"][kb]; kms = mk.get("gpu_ms") or ((mk["t_end"] - mk["t_start"]) * 1e3 if mk.get("t_end") and mk.get("t_start") else None)
            if fq is None:
                try:
                    import summary_runs; fq = summary_runs.smi_clock(run, mk)
                except Exception: pass
            rows.append(dict(waves=wv["waves"], ctas=wv["ctas"], M=g["M"], N=g["N"], K=g["K"], kernel=kern, cycle=kb, kernel_ms=kms,
                             gflop=flops / 1e9, tflops=(flops / (kms * 1e-3) / 1e12) if kms else None,
                             ai_ideal_flop_per_byte=ai_ideal, ai_tile_flop_per_byte=ai_tile,
                             v_idle=float(idle), didt_drop_300us=float(idle - v[w].min()) if w.sum() else None,
                             t_min_us=float(tt[w][v[w].argmin()] * 1e6) if w.sum() else None,
                             drop_cycle_min=float(idle - v[cyc].min()) if cyc.sum() else None,
                             v2_didt_drop_300us=float(v2[pre].mean() - v2[w].min()) if (v2 is not None and w.sum()) else None,
                             pmd_p_max=pb.get("p_max"), pmd_i_max=((pb.get("i1_max") or 0) + (pb.get("i2_max") or 0)) if pb else None,
                             pmd_p_idle=(S.get("pmd") or {}).get("idle_p"), clk_kernel=fq, run=os.path.basename(run)))
    if not rows:
        print("no t04 wave runs in", d); return
    # sanity: a cycle whose "idle" is far below the folder's idle level has the load inside its pre-onset window (record crop or
    # onset failure); keep it in the csv (flag bad_record) but drop it from medians, table and plot
    v_idle_ref = float(np.median([r["v_idle"] for r in rows])); good = []
    for r in rows:
        r["bad_record"] = bool(r["v_idle"] < v_idle_ref - 0.08)
        if r["bad_record"]: print(f"  ! {r['run']} cycle {r['cycle']}: idle {r['v_idle']:.3f} V vs folder {v_idle_ref:.3f} V -> excluded (load inside the pre-onset window)")
        else: good.append(r)
    rows_all = rows; rows = [r for r in good if r["kernel"] != "layout"]
    keys = ["waves", "ctas", "M", "N", "K", "kernel", "cycle", "bad_record", "kernel_ms", "gflop", "tflops", "ai_ideal_flop_per_byte", "ai_tile_flop_per_byte", "v_idle", "didt_drop_300us", "t_min_us", "drop_cycle_min", "v2_didt_drop_300us", "pmd_p_max", "pmd_i_max", "pmd_p_idle", "clk_kernel", "run"]
    with open(os.path.join(d, "wave_sweep_summary.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows_all: f.write(",".join("" if r.get(k) is None else str(r[k]) for k in keys) + "\n")
    json.dump(rows, open(os.path.join(d, "wave_sweep_summary.json"), "w"), indent=1)        # json = the cycles used for medians
    gfmt = lambda x, fmt: (fmt % x) if x is not None else "    n/a"
    print(f"{'waves':>7}{'CTAs':>6}{'M':>6}{'N':>6}{'K':>6}{'kern':>7}{'cyc':>4}{'kern ms':>9}{'V idle':>8}{'dV 300us':>9}{'@us':>6}{'dV cycle':>9}{'CH1 dV':>8}{'PMD W':>7}{'PMD A':>7}{'clk':>6}")
    for r in rows:
        print(f"{r['waves']:7.2f}{r['ctas']:6d}{r['M']:6d}{r['N']:6d}{r['K']:6d}{r['kernel']:>7}{r['cycle']:4d}{gfmt(r['kernel_ms'],'%9.2f')}{r['v_idle']:8.3f}{gfmt(r['didt_drop_300us'],'%9.3f')}{gfmt(r['t_min_us'],'%6.0f')}{gfmt(r['drop_cycle_min'],'%9.3f')}{gfmt(r['v2_didt_drop_300us'],'%8.3f')}{gfmt(r['pmd_p_max'],'%7.1f')}{gfmt(r['pmd_i_max'],'%7.2f')}{gfmt(r['clk_kernel'],'%6.0f')}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        groups = sorted({(r["kernel"], r["K"]) for r in rows}, key=lambda g: (g[0] != "wave", g[1]))
        mean = lambda key, kern, K, w: np.median([r[key] for r in rows if r["kernel"] == kern and r["K"] == K and r["waves"] == w and r.get(key) is not None] or [np.nan])
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        for kern, K in groups:
            wvs = sorted({r["waves"] for r in rows if r["kernel"] == kern and r["K"] == K}); lab = f"K={K}" + (" cuBLAS ref" if kern == "cublas" else "")
            mk = "D" if kern == "cublas" else "o"
            ax[0].plot(wvs, [mean("didt_drop_300us", kern, K, w) for w in wvs], mk + "-", label=f"di/dt drop 300 us, {lab}")
            ax[0].plot(wvs, [mean("drop_cycle_min", kern, K, w) for w in wvs], mk + ":", alpha=0.6, label=f"whole-cycle drop, {lab}")
            ax[1].plot(wvs, [mean("kernel_ms", kern, K, w) for w in wvs], mk + "-", label=f"kernel time, {lab}")
        ax[0].set_xscale("log"); ax[0].set_xlabel("waves = CTAs / usable SMs (1 CTA/SM, one 256x128 tile per CTA)"); ax[0].set_ylabel("12 V rail drop (V)"); ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=7)
        ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("waves"); ax[1].set_ylabel("ms"); ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=7)
        fig.suptitle(f"wave x size sweep: {os.path.basename(d)}  ({len(groups)} K values; median per point)"); plt.tight_layout(); plt.savefig(os.path.join(d, "wave_sweep_summary.png"), dpi=120)
        print("->", os.path.join(d, "wave_sweep_summary.{csv,json,png}"))
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
