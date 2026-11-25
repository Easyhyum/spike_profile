#!/usr/bin/env python
"""make_report.py <result_dir> — report.md + report.json for one run_all folder: GPU, chosen workload, hardware, then per test
the key numbers (medians over cycles) and the figure list. Reads runs_summary.json (summary_runs.py), clock_sweep_summary.json
and wave_sweep_summary.json when present; (re)generates runs_summary first."""
import glob, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))


def med(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return float(np.median(v)) if v else None


def f(x, fmt="%.3f", unit=""):
    return (fmt % x + unit) if x is not None else "n/a"


def _pcie_link(d):
    """PCIe generation / width seen by nvidia-smi during the runs (idle links down-train, so report idle and under-load values)."""
    import csv as _csv, collections
    gens, widths = collections.Counter(), collections.Counter()
    for f in glob.glob(os.path.join(d, "spike_eval_*", "gpu_*.csv"))[:5]:
        for row in _csv.reader(open(f)):
            if row[0].startswith("timestamp") or len(row) < 10: continue
            try: gens[row[-2].strip()] += 1; widths[row[-1].strip()] += 1
            except Exception: pass
    if not gens: return "not recorded"
    return "gen " + "/".join(sorted(gens)) + " x" + "/".join(sorted(widths)) + f" (samples: {dict(gens)} ; {dict(widths)})"


def _scope_dt_ns(d):
    import numpy as _np
    for f in glob.glob(os.path.join(d, "spike_eval_*", "burst_00.npz"))[:1]:
        try:
            m = json.loads(str(_np.load(f)["meta"])); return f"{m['dt']*1e9:.0f} ns"
        except Exception: pass
    return "1 us"


def main():
    d = sys.argv[1].rstrip("/")
    subprocess.run([sys.executable, os.path.join(HERE, "summary_runs.py"), d], stdout=subprocess.DEVNULL)
    rows = json.load(open(os.path.join(d, "runs_summary.json"))) if os.path.exists(os.path.join(d, "runs_summary.json")) else []
    auto = {}
    for cand in (os.path.join(d, "auto_shape.env.used"), os.path.join(HERE, "logs", "auto_shape.env")):
        if os.path.exists(cand):
            for line in open(cand):
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1); auto[k] = v.strip('"')
            break
    hw = {}
    hws = sorted(glob.glob(os.path.join(HERE, "logs", "hw_*.json")))
    if hws:
        try: hw = json.load(open(hws[-1]))
        except Exception: pass
    R = dict(result_dir=d, gpu=hw.get("gpu_info"), scope=hw.get("idn"), workload=auto, tests={})
    L = [f"# Spike test report: {os.path.basename(d)}", ""]
    fm_on = any(json.load(open(f)).get("freqmon_us") for f in glob.glob(os.path.join(d, "spike_eval_*", "marks.json"))[:3]) if glob.glob(os.path.join(d, "spike_eval_*", "marks.json")) else True
    mode = (f"{auto.get('AUTO_USABLE_SMS', '?')} usable with freqmon (one SM holds the 10 us clock monitor)" if fm_on
            else "ALL SMs run the GEMM (FREQMON=0: no in-kernel clock monitor; clock = nvidia-smi 100 ms)")
    R["freqmon"] = bool(fm_on)
    pls = [json.load(open(f)).get("power_limits") for f in glob.glob(os.path.join(d, "spike_eval_*", "marks.json"))[:3]]
    pl = next((x for x in pls if x), None); R["power_limits"] = pl
    pw = next((json.load(open(f)).get("args", {}).get("prewarm_gap_ms") for f in glob.glob(os.path.join(d, "spike_eval_*", "marks.json"))[:1]), 0) or 0
    warm_line = (f"- Warm clock: a GEMM of the same shape ended {pw:g} ms before every measured launch (clock at its settled level from t = 0)" if pw else "- Cold launch: no warm-up GEMM (the clock sits ~30 ms at ~2640 MHz after the load step)")
    pl_line = (f"- Board power limit during the runs: {pl['current']:.0f} W (default {pl.get('default', float('nan')):.0f} W, max {pl.get('max', float('nan')):.0f} W; nvidia-smi -q -d POWER)"
               if pl else "- Board power limit during the runs: not recorded (older run)")
    L += [f"- GPU: {hw.get('gpu_info', '?')}  ({auto.get('AUTO_SMS', '?')} SMs; {mode}); PCIe link during the runs: {_pcie_link(d)}",
          f"- Scope: {hw.get('idn', '?')}; PMD2X: {hw.get('pmd_info', '?')}",
          f"- Workload (auto, find_max_shape.py): {auto.get('AUTO_DTYPE', '?')} M=N=K={auto.get('AUTO_SIZE', '?')}, 1 launch = {auto.get('AUTO_LAUNCH_MS', '?')} ms, "
          f"{auto.get('AUTO_TFLOPS', '?')} TFLOP/s, {auto.get('AUTO_PMEAN_W', '?')} W mean (nvidia-smi); every measurement is a single cycle = 1 launch unless noted",
          pl_line, warm_line,
          f"- Alignment: t = 0 at the rail-voltage onset (scope, {_scope_dt_ns(d)} per sample; csv decimated to >= 1 us, npz full rate); PMD2X 1 ms; freqmon in-kernel clock 10 us", ""]
    if rows:
        n1 = "CH(V)"; n2 = "CH(V2)"
        first = glob.glob(os.path.join(d, "spike_eval_*", "burst_00.npz"))
        if first:
            try:
                meta = json.loads(str(np.load(first[0])["meta"])); n1 = f"CH{meta.get('v_ch')} {meta.get('v_name', '')}".strip(); n2 = f"CH{meta.get('v2_ch')} {meta.get('v2_name', '')}".strip()
            except Exception: pass
        # ---- 01 baseline
        b = [r for r in rows if r["tag"].startswith("t01_")]
        if b:
            for r in rows:                                                   # no freqmon -> nvidia-smi clock stands in
                if r.get("clk_kernel") is None and r.get("clk_smi") is not None: r["clk_kernel"] = r["clk_smi"]
            R["tests"]["01_baseline"] = {k: med(b, k) for k in ("kernel_ms", "v_idle", "didt_drop_300us", "t_min_us", "droop_settled", "drop_cycle_min", "v2_idle", "v2_didt_drop_300us", "v2_droop_settled", "lane_idle_mV", "lane_load_mV", "lane_max_mV", "pmd_p_max", "pmd_i_max", "clk_kernel")}
            R["tests"]["01_baseline"]["cycles"] = len(b); R["tests"]["01_baseline"]["edge_triggered"] = sum(1 for r in b if r.get("edge_triggered"))
            t = R["tests"]["01_baseline"]
            L += ["## 01 baseline (strongest-spike GEMM, single cycles)", "",
                  f"- {len(b)} cycles, kernel {f(t['kernel_ms'], '%.1f', ' ms')}, {'in-kernel' if R.get('freqmon', True) else 'nvidia-smi'} clock {f(t['clk_kernel'], '%.0f', ' MHz')}, PMD2X peak {f(t['pmd_p_max'], '%.0f', ' W')} / {f(t['pmd_i_max'], '%.1f', ' A')}",
                  f"- {n1}: idle {f(t['v_idle'])} V, di/dt drop {f(t['didt_drop_300us'])} V at {f(t['t_min_us'], '%.0f', ' us')}, settled droop {f(t['droop_settled'])} V, cycle min {f(t['drop_cycle_min'])} V below idle",
                  f"- {n2}: idle {f(t['v2_idle'])} V, di/dt drop {f(t['v2_didt_drop_300us'])} V, settled droop {f(t['v2_droop_settled'])} V",
                  f"- lane difference {n2} - {n1}: idle {f(t['lane_idle_mV'], '%+.0f', ' mV')}, during load {f(t['lane_load_mV'], '%+.0f', ' mV')}, max instantaneous {f(t['lane_max_mV'], '%+.0f', ' mV')}",
                  f"- scope edge trigger fired on {t['edge_triggered']}/{len(b)} cycles (the rest used the software force; all onset-aligned)"]
            if med(b, "scope_i_max") is not None:
                for k in ("scope_i_idle", "scope_didt_A_300us", "scope_i_max", "scope_i_load", "scope_p_max", "scope_p_load", "scope_p_idle", "scope_i_level_A", "scope_i_rise_start_us", "scope_i_t10_us", "scope_i_t90_us", "scope_i_rise_10_90_us", "scope_didt_A_per_us", "scope_didt_max_A_per_us", "scope_didt_max_at_us"): t[k] = med(b, k)
                L += [f"- scope current: idle {f(t['scope_i_idle'], '%.1f', ' A')}, di/dt step {f(t['scope_didt_A_300us'], '%+.1f', ' A')} in the first 300 us, cycle max {f(t['scope_i_max'], '%.1f', ' A')}, load {f(t['scope_i_load'], '%.1f', ' A')}",
                      f"- current rise at the kernel start: leaves idle at {f(t['scope_i_rise_start_us'], '%+.0f', ' us')}, 10 % at {f(t['scope_i_t10_us'], '%+.0f', ' us')}, 90 % at {f(t['scope_i_t90_us'], '%+.0f', ' us')} -> rise 10-90 % = {f(t['scope_i_rise_10_90_us'], '%.0f', ' us')} "
                      f"({f(t['scope_i_idle'], '%.1f')} -> {f(t['scope_i_level_A'], '%.1f', ' A')} plateau, {f(t['scope_didt_A_per_us'], '%.3f', ' A/us')} between the 10 % and 90 % points; steepest 10 us slope {f(t['scope_didt_max_A_per_us'], '%.3f', ' A/us')} at {f(t['scope_didt_max_at_us'], '%+.0f', ' us')}; t = 0 at the voltage onset)",
                      f"- scope power P = V x I: idle {f(t['scope_p_idle'], '%.0f', ' W')}, load {f(t['scope_p_load'], '%.0f', ' W')}, peak {f(t['scope_p_max'], '%.0f', ' W')}",
                      f"- peak vs averaging window (scope): P {f(med(b, 'scope_p_max'), '%.0f')} W raw / {f(med(b, 'scope_p_max_10us'), '%.0f')} W @10 us / {f(med(b, 'scope_p_max_100us'), '%.0f')} W @100 us / {f(med(b, 'scope_p_max_1ms'), '%.0f')} W @1 ms; "
                      f"I {f(med(b, 'scope_i_max'), '%.1f')} / {f(med(b, 'scope_i_max_10us'), '%.1f')} / {f(med(b, 'scope_i_max_100us'), '%.1f')} / {f(med(b, 'scope_i_max_1ms'), '%.1f')} A; PMD2X (1 ms samples) peak {f(t['pmd_p_max'], '%.0f', ' W')} / {f(t['pmd_i_max'], '%.1f', ' A')}"]
            L.append("")
        # ---- 02 duration
        du = [r for r in rows if r["tag"].startswith("t02_")]
        if du:
            tags = []; [tags.append(r["tag"]) for r in du if r["tag"] not in tags]
            L += ["## 02 cycle-length sweep", "", "| run | M | launches | kernel ms | di/dt drop V | settled droop V | cycle min V | PMD W | clock MHz |", "|---|---|---|---|---|---|---|---|---|"]
            R["tests"]["02_duration"] = []
            for tg in tags:
                rs = [r for r in du if r["tag"] == tg]; e = dict(tag=tg, M=rs[0]["M"], launches=rs[0]["launches"], **{k: med(rs, k) for k in ("kernel_ms", "didt_drop_300us", "droop_settled", "drop_cycle_min", "pmd_p_max", "clk_kernel")})
                R["tests"]["02_duration"].append(e)
                L.append(f"| {tg} | {e['M']} | {e['launches']} | {f(e['kernel_ms'], '%.1f')} | {f(e['didt_drop_300us'])} | {f(e['droop_settled'])} | {f(e['drop_cycle_min'])} | {f(e['pmd_p_max'], '%.0f')} | {f(e['clk_kernel'], '%.0f')} |")
            L.append("")
    # ---- 03 clock
    cs = os.path.join(d, "clock_sweep_summary.json")
    if os.path.exists(cs):
        C = json.load(open(cs)); R["tests"]["03_clock"] = C
        L += ["## 03 clock sweep (single cycles, locked graphics clock)", "", "| setting | clock in kernel MHz | di/dt drop V | settled droop V | cycle min V | PMD W | kernel ms |", "|---|---|---|---|---|---|---|"]
        seen = []
        for r in C:
            if r["setting"] in seen: continue
            seen.append(r["setting"]); rs = [x for x in C if x["setting"] == r["setting"]]
            L.append(f"| {r['setting']} | {f(med(rs, 'clk_kernel'), '%.0f')} | {f(med(rs, 'didt_drop_300us'))} | {f(med(rs, 'droop_settled'))} | {f(med(rs, 'v_min_burst'))} | {f(med(rs, 'pmd_p_max'), '%.0f')} | {f(med(rs, 'host_ms'), '%.1f')} |")
        L.append("")
    # ---- 04 wave
    ws = os.path.join(d, "wave_sweep_summary.json")
    if os.path.exists(ws):
        W = json.load(open(ws)); R["tests"]["04_wave"] = W
        L += ["## 04 wave x size sweep (wavegemm single cycles; cuBLAS max shape as reference)", "", "| kernel | K | waves | CTAs | M | N | kernel ms | TFLOP/s | di/dt drop V | CH(V2) drop V | PMD W | PMD A |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for kern, K, w in sorted({(r.get("kernel", "wave"), r["K"], r["waves"]) for r in W}, key=lambda x: (x[0] != "wave", x[1], x[2])):
            rs = [r for r in W if r.get("kernel", "wave") == kern and r["K"] == K and r["waves"] == w]
            L.append(f"| {kern} | {K} | {w:.2f} | {rs[0]['ctas']} | {rs[0]['M']} | {rs[0]['N']} | {f(med(rs, 'kernel_ms'), '%.2f')} | {f(med(rs, 'tflops'), '%.1f')} | {f(med(rs, 'didt_drop_300us'))} | {f(med(rs, 'v2_didt_drop_300us'))} | {f(med(rs, 'pmd_p_max'), '%.0f')} | {f(med(rs, 'pmd_i_max'), '%.1f')} |")
        L.append("")
    figs = sorted(glob.glob(os.path.join(d, "*.png")))
    L += ["## Figures (folder level)", ""] + [f"- `{os.path.basename(p)}`" for p in figs] + ["",
          "Per-run folders hold `burst_NN.png` (4 x windows, both channels), `burst_NN_freq.png` (+ in-kernel clock), `burst_NN_V_*.png`, csv/npz raw data, `marks.json`, `summary.json`.", ""]
    open(os.path.join(d, "report.md"), "w").write("\n".join(L)); json.dump(R, open(os.path.join(d, "report.json"), "w"), indent=1)
    print("\n".join(L)); print("->", os.path.join(d, "report.md"))


if __name__ == "__main__":
    main()
