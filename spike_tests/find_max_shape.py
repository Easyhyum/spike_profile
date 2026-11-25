#!/usr/bin/env python
"""find_max_shape.py — pick, on WHATEVER GPU is attached, the GEMM (dtype, M=N=K) that draws the most power per launch
("strongest spike" workload) and derive the shapes the test sequence needs from it. Prints a shell fragment and writes
logs/auto_shape.json + logs/auto_shape.env (sourced by common.sh).

How: for every candidate dtype x size, warm up, time one launch (CUDA events), then run the GEMM back-to-back for
--probe-s seconds while nvidia-smi samples power at 50 ms; score = MEAN power.draw over the probe (the 50 ms max is too noisy
to rank by). Candidates whose single launch is longer than --max-launch-ms (the scope holds 2 s at 1 us; 15 ms pre +
15 ms post around it) or that do not fit in memory are skipped. Among candidates within 3 % of the best mean power the
one whose single launch is closest to --target-launch-ms wins (long enough to show the settled droop and the DVFS
step, short enough for a comfortable scope window).

    python find_max_shape.py                # defaults: fp32,bf16 x 4096,8192,12288,16384
    python find_max_shape.py --dtypes fp32 --sizes 8192        # pin it
Usable SMs for the wave sweep = SM count - 1 (freqmon holds one SM); the wave targets are derived in wave_targets.py.
"""
import argparse, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtypes", default="fp32,bf16"); ap.add_argument("--sizes", default="4096,8192,12288,16384")
    ap.add_argument("--probe-s", type=float, default=1.5, help="seconds of back-to-back launches per candidate")
    ap.add_argument("--max-launch-ms", type=float, default=400.0, help="skip candidates with a longer single launch")
    ap.add_argument("--target-launch-ms", type=float, default=60.0, help="tie-break: prefer a single launch near this length")
    ap.add_argument("--out", default=os.path.join(HERE, "logs", "auto_shape"))
    a = ap.parse_args()
    import torch, pmd2_fast as pf
    dev = torch.device("cuda"); p = torch.cuda.get_device_properties(0)
    smi = pf.find_nvidia_smi() or "nvidia-smi"
    res = []
    for dts in a.dtypes.split(","):
        dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dts]
        for n in (int(x) for x in a.sizes.split(",")):
            need = 3 * n * n * (4 if dt == torch.float32 else 2)
            if need > p.total_memory * 0.8:
                res.append(dict(dtype=dts, size=n, skipped="memory")); continue
            try:
                A = torch.randn(n, n, device=dev, dtype=dt); B = torch.randn(n, n, device=dev, dtype=dt); C = torch.empty(n, n, device=dev, dtype=dt)
            except RuntimeError:
                res.append(dict(dtype=dts, size=n, skipped="memory")); torch.cuda.empty_cache(); continue
            t0 = time.perf_counter(); k = 0
            while time.perf_counter() - t0 < 1.0 or k < 3:
                torch.matmul(A, B, out=C); k += 1
                if k % 2 == 0: torch.cuda.synchronize()
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record(); torch.matmul(A, B, out=C); e1.record(); torch.cuda.synchronize()
            ms = e0.elapsed_time(e1)
            if ms > a.max_launch_ms:
                res.append(dict(dtype=dts, size=n, launch_ms=ms, skipped=f"launch > {a.max_launch_ms:.0f} ms")); del A, B, C; torch.cuda.empty_cache(); continue
            proc = subprocess.Popen([smi, "-i", "0", "--query-gpu=power.draw.instant,power.draw,clocks.sm", "--format=csv,noheader,nounits", "-lms", "50"],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            t0 = time.perf_counter(); k = 0
            while time.perf_counter() - t0 < a.probe_s:
                torch.matmul(A, B, out=C); k += 1
                if k % 4 == 0: torch.cuda.synchronize()
            torch.cuda.synchronize(); time.sleep(0.15); proc.terminate(); out = proc.communicate(timeout=5)[0]
            pw = []; clk = []
            for line in out.splitlines():
                f = [x.strip() for x in line.split(",")]
                try:
                    pi = float(f[0]) if f[0] not in ("[N/A]", "") else float(f[1]); pw.append(pi); clk.append(float(f[2]))
                except Exception: pass
            tfl = 2 * n ** 3 / (ms * 1e-3) / 1e12
            res.append(dict(dtype=dts, size=n, launch_ms=ms, tflops=tfl, p_max=max(pw) if pw else None, p_mean=sum(pw) / len(pw) if pw else None,
                            clk_max=max(clk) if clk else None, samples=len(pw)))
            print(f"{dts:5} {n:6}: {ms:8.2f} ms/launch {tfl:6.1f} TFLOP/s  power max {max(pw) if pw else float('nan'):6.1f} W (mean {sum(pw)/len(pw) if pw else float('nan'):5.1f}, {len(pw)} samples)", flush=True)
            del A, B, C; torch.cuda.empty_cache()
    ok = [r for r in res if r.get("p_mean")]
    if not ok: sys.exit("no candidate could be measured")
    best = max(ok, key=lambda r: r["p_mean"]); near = [r for r in ok if r["p_mean"] >= 0.97 * best["p_mean"]]
    best = min(near, key=lambda r: abs(r["launch_ms"] - a.target_launch_ms))
    n = best["size"]
    # duration sweep rows: ~1/16, 1/4 and 1/1 of the max shape (short / mid / long cycle), multiples of 256
    rows = sorted({max(256, (n // 16) // 256 * 256), max(256, (n // 4) // 256 * 256), n})
    auto = dict(gpu=p.name, sms=p.multi_processor_count, usable_sms=p.multi_processor_count - 1, mem_gb=p.total_memory / 2**30,
                dtype=best["dtype"], size=n, launch_ms=best["launch_ms"], tflops=best["tflops"], p_max_w=best["p_max"], p_mean_w=best["p_mean"],
                duration_rows=rows, candidates=res, chosen_by=f"max mean nvidia-smi power.draw over {a.probe_s} s; within 3 % -> single launch closest to {a.target_launch_ms:.0f} ms")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(auto, open(a.out + ".json", "w"), indent=1)
    with open(a.out + ".env", "w") as f:
        f.write(f"# written by find_max_shape.py on {time.strftime('%Y-%m-%d %H:%M:%S')} for {p.name}\n")
        f.write(f"AUTO_DTYPE={best['dtype']}\nAUTO_SIZE={n}\nAUTO_LAUNCH_MS={best['launch_ms']:.2f}\nAUTO_TFLOPS={best['tflops']:.1f}\nAUTO_PMAX_W={best['p_max']:.1f}\nAUTO_PMEAN_W={best['p_mean']:.1f}\n")
        f.write(f"AUTO_DURATION_ROWS=\"{' '.join(map(str, rows))}\"\nAUTO_SMS={p.multi_processor_count}\nAUTO_USABLE_SMS={p.multi_processor_count - 1}\n")
    print(f"\nchosen: {best['dtype']} M=N=K={n}  {best['launch_ms']:.1f} ms/launch  {best['tflops']:.1f} TFLOP/s  {best['p_max']:.0f} W  on {p.name} ({p.multi_processor_count} SMs)")
    print(f"-> {a.out}.env / .json")


if __name__ == "__main__":
    main()
