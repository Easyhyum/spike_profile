# GPU 12 V rail di/dt measurement rig — handoff

Written for another engineer or an AI assistant that has to run, port or extend this measurement setup without the
original conversation. Everything below is in `spike_tests_bundle_<date>.zip` (code only; results are not included).
Last updated 2026-11-24 (sections renumbered; RESUME_DIR; model_run op hooks / vLLM; `BUGS_AND_FIXES.md` consolidates every bug and fix).

Three campaign folders share one measurement engine (`spike_eval.py`):

| folder | campaign | workload | state |
|---|---|---|---|
| `spike_tests/` | voltage + voltage (CH1 PSU cable, CH2 PCIe slot) | cuBLAS max-power shape, clocks, `wavegemm.cu` waves | complete; original files only |
| `spike_tests_vi/` | current + voltage (CH1 CP500 = all 12 V current, CH2 PSU 12 V, CH4 PCIe slot 12 V) on the LeCroy HDO4034A | same tests, symlinked scripts | running since 2026-11-17 (01 verified) |
| `ws_gemm/` | warp-specialized bf16 GEMM (`wsgemm.cu`): loader pacing, soft starts, kernel variants | own scripts `pacing.sh` / `variants.sh` / `run_all.sh` | active (2026-11-17) |

## 1. What this measures

One GEMM launch on a GPU is a current step on the 12 V supply. The rig records, **on the same launch**, four signals:

| signal | instrument | resolution | what it gives |
|---|---|---|---|
| 12 V rail voltage (PSU cable, PCIe slot) and/or cable current | oscilloscope (Keysight DSOX1202A or LeCroy HDO4034A + CP500) | 1 us | di/dt drop in the first 300 us, settled droop, ringing, lane differences, P = V x I |
| cable voltage / current / power | ElmorLabs PMD2X (two PCIe cables) | 1 ms | peak and mean cable power, current |
| board power, SM clock | nvidia-smi | 100 ms | coarse context |
| SM clock inside the GPU | `freqmon_excl.cu` (a kernel that owns one SM and samples the clock) | 10 us | DVFS behaviour during the kernel, aligned to the launch via a `%globaltimer` stamp |

Workloads: the cuBLAS GEMM that draws the most power on the attached GPU (found automatically), shorter variants,
locked graphics clocks, `wavegemm.cu` (a GEMM with an explicit grid) to sweep the number of waves, and `ws_gemm/wsgemm.cu`
(a warp-specialized bf16 GEMM whose global-memory loader can be paced: burst depth, trickle, duty ramp, SM stagger) to
see how the shape of the current step itself changes the di/dt drop.

## 2. Hardware / software topology

- Host: Windows 11 + WSL2 (Ubuntu). GPU in WSL via the NVIDIA WSL driver. The scope and the PMD2X are USB devices forwarded
  into WSL with **usbipd-win** (`usbipd bind --force` once as admin, `usbipd attach --wsl --busid <id>`; `spike_tests/usb_attach.sh`
  automates attach and recovery). The HDO4034A can also be reached over LAN (`TCPIP::<ip>::INSTR`, scope remote control = VXI11);
  over USB the scope's remote control must be set to USBTMC (Utilities > Remote), otherwise commands are never answered.
- Python venv in `matmul_test/venv` (see `requirements.txt`). CUDA toolkit for JIT-building the `.cu` extensions
  (`torch.utils.cpp_extension.load`): found via `CUDA_HOME`, `../cuda*`, `~/cuda*` or `/usr/local/cuda` (`spike_eval._cuda_env`).
  The nvcc arch flag is derived from the attached GPU (`_arch_flags`).
- Everything runs as root (`sudo`): the PMD2X serial port and USBTMC need it. The sudo password is never stored in the scripts.
- Clock locking / power limit: `nvidia-smi -lgc` and `-pl` do not work inside WSL. `spike_tests/clockctl.py` starts one
  elevated PowerShell "server" on Windows (one UAC prompt per sequence) that runs the Windows `nvidia-smi -lgc/-rgc/-pl` on
  request through files in `C:\Users\<user>\.spike_clockctl\`; `clockctl.py limits / pl <W> / plreset` handle the board power
  limit (5060 Ti: default 180 W, range 150-200 W), and the server restores `-rgc` and the default limit when it exits.
- Wiring used so far: scope CH1 on the PSU 12 V cable, CH2 on the PCIe slot 12 V (voltage + voltage campaign); next: CH1 =
  current probe on the cable (1..90 A), CH2 = PSU 12 V (current + voltage campaign). PMD2X in line with both PCIe cables.

## 3. Folder map

```
matmul_test/
  spike_eval.py          the measurement program (one run = N cycles of one workload, all instruments)
  onset.py               kernel-start detection on a voltage record (shared by every plot / summary)
  plot_spike_csv.py      per-run voltage figures (whole cycle, 0-300 us, PMD, overlay, second channel)
  wavegemm.cu            fp32 GEMM, one 256x128 tile per CTA, 1 CTA/SM -> exact wave count (81-92 % of cuBLAS)
  freqmon_excl.cu        in-kernel SM clock monitor (holds one SM with 96 KB smem)
  gtstamp.cu             writes %globaltimer on the workload stream (launch-time stamps)
  pmd2_fast.py           PMD2X serial helpers + nvidia-smi logger helpers
  wavegemm_cutlass.cu.txt  CUTLASS version of wavegemm (same kernel as cuBLAS); needs CUTLASS headers, not built here
  requirements.txt, HANDOFF.md
  ws_gemm/               warp-specialized GEMM campaign (own config.env, logs/, results/; tags t05_* / t06_*)
    wsgemm.cu            bf16 GEMM, producer warps (cp.async -> smem ring, mbarriers) + consumer warps (ldmatrix + mma.sync);
                         variant table (tile / stages / warp grid / producer warps) + runtime pacing knobs; built by
                         spike_eval.py from this path (`--kernel ws`), so the folder name must stay `ws_gemm`
    ws_bench.py          builds wsgemm, lists `variants()`, compares variants / knobs with cuBLAS bf16 (no instruments)
    stage_timing.py      in-kernel time per smem stage of both warp groups (issue / wait / compute / starve)
    pacing.sh            t05: single cycles + bursts of cuBLAS vs ws configs (burst depth, trickle, ramp, stagger, noload, loadonly)
    variants.sh          t06: bursts of every ws variant vs cuBLAS (power, clock, GFLOP/J)
    run_all.sh           stages A (decomposition) B1 B2 (variants, two shapes) C (soft start / trickle) D (compute-only / loader-only)
    summary_ws.py        ws_summary.csv/md + ws_overlay_s<i>.png per shape
    common.sh check_hw.py usb_attach.sh summary_runs.py   symlinks to ../spike_tests (recreate them when a copy drops symlinks)
  spike_tests/           voltage + voltage campaign (CH1 PSU cable, CH2 PCIe slot)
    config.env           hardware / channel settings (no workload settings)
    common.sh            shared shell: auto shape, hardware gating, argument builders, run_eval
    00_preflight.sh      GPU / PMD / scope check + find_max_shape.py
    01_baseline.sh 02_duration.sh 03_clock.sh 04_wave.sh run_all.sh
    find_max_shape.py    probes dtype x size, picks the strongest-spike GEMM -> logs/auto_shape.env
    wave_targets.py      wave -> (CTAs, M, N) for the attached GPU
    check_hw.py          preflight JSON (per-channel verdicts), Keysight and LeCroy
    clockctl.py          Windows-side clock lock server client
    usb_attach.sh        usbipd attach with vhci-port recovery
    summary_runs.py summary_clock.py summary_wave.py make_report.py overlay_summary.py replot_burst.py
    lecroy_check.py/.sh  dry run of the LeCroy driver
    launch_hook.c        LD_PRELOAD hook printing every CUDA launch's grid/block/smem/CTAs-per-SM (cuBLAS tile discovery)
    legacy/              retired tests (PMD-only, two-cable, separate V x I, bf16 vs fp32)
    logs/  results/      per-machine state and outputs (not in the bundle)
  spike_tests_vi/        current + voltage campaign: every script is a symlink to ../spike_tests; own config.env, logs/, results/
  model_run/             LLM inference profiles (HF + vLLM), op hooks, shapes tables, sustained prefill
  hdo_capture/           HDO-only tools: fixed-time 2-channel capture, CP500 step response, ramp analysis, fast spike capture
  load_tools/            sustained_load.py: steady GEMM load with per-cable PMD readout (wiring checks)
  ncu_profile/           Nsight Compute profiles of the wave x size shapes (bandwidth, hit rates, throughput); ncu extracted in ~/nsight-compute
  BUGS_AND_FIXES.md      every bug found during the campaign, cause and fix, with the date from which results are trustworthy
```

`spike_tests_vi/setup_links.sh` recreates the symlinks after unpacking the bundle on a filesystem that dropped them. For
`ws_gemm/` the four links are `common.sh check_hw.py usb_attach.sh summary_runs.py -> ../spike_tests/<same name>`
(`common.sh` resolves `HERE` from the link's own directory, so the folder's config.env, logs/ and results/ are used).
The bundle stores them as symlink entries; `unzip` restores them, Python's `zipfile.extractall` writes the target path as
file content instead, so after a Python extraction run `cd ws_gemm && for f in common.sh check_hw.py usb_attach.sh
summary_runs.py; do ln -sf ../spike_tests/$f $f; done`.

## 4. How to run

```bash
cd matmul_test/spike_tests            # or spike_tests_vi
vi config.env                         # channels, probe scale, usbipd bus ids, scope model
sudo ./run_all.sh                     # 00 preflight (+ shape probe) -> 01 -> 02 -> 03 (one UAC "yes") -> 04 -> report.md
sudo ./run_all.sh 01 04               # subset
sudo ./01_baseline.sh                 # any test alone; each invocation gets results/<stamp>_<name>/
sudo env FREQMON=0 ./run_all.sh       # all-SM mode: no freqmon, every SM runs the GEMM (results/<stamp>_run_all_allsm, clocks from nvidia-smi)
sudo env POWER_LIMIT_W=200 PREWARM_GAP_MS=20 SCOPE_SAMPLE_NS=20 ./run_all.sh      # 200 W cap, warm clock, 20 ns samples (folder suffix _pl200_warm20_20ns)
sudo env RESUME_DIR=results/<stamp>_run_all_... <same env as the original> ./run_all.sh 02 03 04   # continue an interrupted sequence in its folder
sudo env POWER_LIMIT_W=200 ./run_all.sh   # whole sequence at a raised board power limit via clockctl.py pl (UAC once at the start; default restored at the end)
sudo env PREWARM_GAP_MS=20 ./run_all.sh   # warm clock: GEMM ends 20 ms before each measured launch -> ~2750 MHz from t = 0 (LeCroy stop-capture only)
sudo env FREQMON=0 PREWARM_GAP_MS=20 PREWARM_MS=500 SCOPE_SAMPLE_NS=20 ./run_all.sh   # all-SM + warm clock + 20 ns scope samples (started 2026-11-20)

cd matmul_test/ws_gemm                # warp-specialized GEMM campaign (bf16; shapes are fixed in pacing.sh / variants.sh)
sudo ./run_all.sh                     # A B1 B2 C D; each stage writes results/<stamp>_pacing|variants/ + ws_summary.md
sudo ./run_all.sh C                   # one stage; or directly:  sudo env WS_CONFIGS="cublas burst4 stag50" WS_BURST_CONFIGS= ./pacing.sh
venv/bin/python ws_gemm/ws_bench.py   # kernel-only comparison of the variants / knobs with cuBLAS (no instruments)
```

`spike_eval.py --kernel ws` arguments (all forwarded by `pacing.sh`'s `cfg_args`): `--ws-variant N` (id from `ws_bench.py`),
`--ws-depth` (loader run-ahead in stages, 0 = all), `--ws-spread/--ws-groups/--ws-slack-ns` (trickle: the loader's idle time
becomes pauses between chunk groups), `--ws-ramp-us/--ws-duty0/--ws-slot-us` (PWM duty ramp, first launch of a cycle only),
`--ws-stagger-us` (i-th CTA of the first wave starts i x us late, first launch only), `--ws-noload` (compute only),
`--ws-nocompute 1|N` (loader only: free-running, or consumers sleep N ns per tile). bf16 only; M, N, K must be multiples of
the variant's tile. `marks.json` gets a `ws` block (variant, stages, warp counts, smem, CTAs, every knob).

Outputs: `results/<stamp>_run_all/report.md` + `report.json`, `runs_summary.csv/json` (one row per cycle),
`clock_sweep_summary.*`, `wave_sweep_summary.*`, overlay PNGs, and one `spike_eval_<stamp>_<tag>/` folder per run with
`burst_NN.csv/npz` (raw 1 us samples), `pmd.csv`, `freqmon.csv`, `gpu_*.csv`, `marks.json`, `summary.json` and figures.

## 5. Measurement logic (spike_eval.py)

1. Workload tensors on their own CUDA stream (the legacy default stream would serialize with freqmon). Warm-up >= 20
   launches / 2 s, then 10 timed launches -> `t_one`. `K` launches per cycle = `--launches` (1 = single cycle) or
   round(`--on-ms` / `t_one`).
2. Processes: PMD2X poller (multiprocessing, ~1 kHz), nvidia-smi logger (subprocess), scope worker (multiprocessing).
   freqmon runs in-process on its own stream after the stamp kernel is warm.
3. (DSOX) Per cycle: wait PMD ready -> wait `armed` from the scope worker (Arm Event Register confirmed on the DSOX) -> main
   process clears `armed` and waits 50 ms -> stores the launch time (`launch_t`, shared double) and sets the `launch` event
   -> stamp -> K launches -> stamp -> sync. The scope worker forces the trigger (`:TRIGger:FORCe` / `FRTR`) as soon as it
   sees the launch event; if the hardware edge trigger already fired, the force is a no-op. Reason: at 0.5 V/div the DSOX
   edge trigger misses randomly (hysteresis ~0.1-0.2 V), and small-load drops (< 0.15 V) never reach the edge level.
   **Force timing (fixed 2026-11-17):** while `armed` is clear and the launch has not happened yet, the worker sends no
   status query at all (`done()` is a ~10 ms USB round trip) and blocks on the launch event; after the launch it polls
   `done()` every 30 ms. The force now goes out 0.5-0.7 ms after the launch, measured against the main process's launch
   time and logged as `force_cmd_ms` (previously measured inside the worker from the moment it noticed the event, which
   hid any delay).
   **(HDO4034A) stop-capture:** the LeCroy never completes a SINGLE acquisition by itself and `FRTR` ends it at once, with the
   record = the window BEFORE the force (TRDL only changes the nominal time axis). `LecroyHDO.setup` therefore parks the edge
   level out of range and sets `force_delay_s = window - pre`; the worker sends `FRTR` that long after the launch -> record
   [-pre, window - pre] around the launch (measured: load at +1.0 .. +62.4 ms, onset +0.6 .. +0.8 ms = force jitter).
   Records carry `capture = "stop"`; the collector prints "stop-captured". Measured stop latency (record end after the
   FRTR): ~1-6 ms at a 100 ms window, ~13 ms at 200 ms, 25-45 ms at 0.5-1 s, and the first SINGLE after a timebase change
   once ended ~150 ms early. The driver therefore asks for pre + load + post + 60 ms of window (TDIV then snaps up: 63 ms
   GEMM -> 200 ms record, 5-launch burst -> 500 ms), sends FRTR at launch + load + post, takes one warm-up acquisition after
   setup, and labels the csv time axis as if the record ended exactly at the force (`t0_nominal`); the plots align on the
   onset anyway. A record whose load begins before its first sample (2026-11-18 first x5 run) is invalid: check
   `[scope] burst N: ... record -A .. +B ms` covers the load.
4. Scope window = 15 ms pre-trigger + K x t_one + 15 ms post, 1 us/sample. Records are matched to cycles by completion time;
   a record that completed before cycle 0 (spurious trigger) or a second record for one cycle (smaller drop) is dropped.
5. Alignment: every plot and metric uses t = 0 at the **voltage onset** (`onset.py`): 20-sample smoothing, robust idle
   (median / MAD of the pre-trigger part), threshold = idle - max(6 sigma, 10 mV, 30 % of the record's deepest excursion),
   walked back to 2 sigma. freqmon is aligned with the GPU stamp of the first launch; PMD on its own current edge.
   `edge_triggered` = onset within 0.3 ms of the trigger.
6. Outputs per cycle: `burst_NN.png` (rows = x windows with 1 us / 10 us / 100 us / 1 ms grid, columns = P, I, V as
   available; both voltage channels on the V axes; max/min labels per panel; power axes show the GPU's power limits from
   `nvidia-smi -q -d POWER` saved in `marks.json["power_limits"]`: dashed default/current cap, dotted max limit), `burst_NN_freq.png` (+ freqmon on a right axis), `plot_spike_csv.py`
   figures, csv/npz. The csv keeps the scope-trigger frame (`t_s`), `marks.json` has host/GPU times, `summary.json` PMD/smi.

## 6. Tests

| test | workload | sweep | cycles |
|---|---|---|---|
| 01 baseline | auto shape (fp32 8192^3 on the 5060 Ti) | - | CYCLES=10 single cycles |
| 02 duration | M = size/16, size/4, size, and bursts of 2 and 5 launches (BURSTS; ~125 / ~315 ms: does the 180 W limiter pull the ~200 W PMD level down?) | cycle length | CYCLES each |
| 03 clock | auto shape | default, max, max x 0.9..0.5 from the supported-clock list | CLOCK_CYCLES=2 |
| 04 wave | wavegemm, K = size | 0.5..0.9, 1.0, 1.5..2.0, 3, 4, 8, 16, 32, max shape | WAVE_CYCLES=3 |
| ws 05 pacing (`ws_gemm/pacing.sh`, tags `t05_s<i>_<config>[_x12]`) | bf16 M x K x N = 4096x12288x4096 and 4096x4096x12288 (high arithmetic intensity, ~9 ms per launch) | configs cublas, burst4/2/1, trk<slack>g<groups>, ramp<us>, stag<us>, noload, loadonly, loadmax, base; single cycles (di/dt) and bursts of WS_BURST=12 (sustained power) | WS_CYCLES=5 / WS_BURST_CYCLES=3 |
| ws 06 variants (`ws_gemm/variants.sh`, tags `t06_s<i>_v<id>`) | same shapes | every `wsgemm` variant vs cuBLAS, bursts of 12; `WS_EXTRA` adds knobs (D: `--ws-noload`, `--ws-nocompute 1`) | WS_BURST_CYCLES=3 |

`ws_gemm/run_all.sh` stages: A = power decomposition (cuBLAS, normal, compute only, loader only at 0.74x and max rate,
awake-GPU base, all bursts), B1/B2 = variants on shape 1 / a subset on shape 2, C = soft start and trickle single cycles
with the scope, D = compute-only and loader-only power per variant. The ws tests need the GPU and the PMD; the scope is
added when a voltage channel is usable (`-m 0` otherwise). Shapes are fixed in the scripts (no shape probe in `ws_gemm`).

`find_max_shape.py`: candidates fp32/bf16 x 4096..16384 (skips > 400 ms launches or > 80 % memory), 1.5 s of launches each
with nvidia-smi at 50 ms, score = mean power, ties within 3 % -> single launch closest to 60 ms. Writes `logs/auto_shape.env`
(`AUTO_DTYPE AUTO_SIZE AUTO_LAUNCH_MS AUTO_DURATION_ROWS AUTO_SMS AUTO_USABLE_SMS`). Re-probed when older than a day or `PROBE=1`.

Waves: `wavegemm` grid = (M/256)(N/128) CTAs, 1 CTA/SM (72 KB dynamic smem, checked with the occupancy API), usable SMs =
device SMs - 1 (freqmon), or all SMs with `FREQMON=0` (common.sh sets AUTO_USABLE_SMS = AUTO_SMS and adds `--no-freqmon`). `wave_targets.py` rounds wave x usable SMs to CTAs and factorizes into M = 256a, N = 128b.
cuBLAS cannot be used for this: its heuristic changes tile and split-K with the shape (on the 5060 Ti: 8192^3 fp32 ->
256x128 tile, 2048 CTAs, 58.5 waves; 2048 rows -> 128x128, 2 CTA/SM; 512 rows -> split-K 3).

## 7. Metrics (summary_runs.py, medians in report.md)

`summary_ws.py` (ws campaign, one row per run): ms/launch, TFLOP/s, % of cuBLAS, PMD peak/mean W, peak A, J/launch above
idle, GFLOP/J, freqmon clock, scope drop at 100 us / 300 us / 1 ms / 3 ms after the onset, cycle minimum, settled droop,
fastest fall (V/ms over 50 us), release overshoot, CH2 minimum drop.

`didt_drop_300us` (idle - min within 300 us), `t_min_us`, `droop_settled` (idle - median during load), `drop_cycle_min`,
second channel `v2_*`, lane difference `lane_idle_mV / lane_load_mV / lane_max_mV` (CH2 - CH1), current `scope_i_idle /
scope_didt_A_300us / scope_i_max / scope_i_load`, power `scope_p_max / scope_p_load / scope_p_idle`, `pmd_p_max / pmd_i_max`,
`clk_kernel` (freqmon median), `edge_triggered`. `summary_wave.py` adds GFLOP, TFLOP/s, arithmetic intensity (ideal and per
tile). Overlays (`overlay_summary.py --param clk|waves --signal voltage|freq|combined`): every setting overlaid, colour =
swept value (equal values -> solid/dashed/dotted), opacity = cycle index, legend below, colour bar right, 1 wave in red.

## 8. Reference numbers (RTX 5060 Ti 16 GB, 36 SMs, 2026-11-13)

- fp32 8192^3 cuBLAS: 63 ms, 17.3 TFLOP/s, PMD cables 226 W peak / 19 A; cable di/dt drop 0.24 V at ~150 us, settled 0.21 V,
  cycle min -0.31 V; slot 0.15 / 0.12 V; slot - cable +17 mV idle, +120 mV load, +156 mV max.
- Clock: locks above ~2470 MHz are not held under load (kernel runs at ~2550-2650 MHz); 0.5 x max -> 0.08 V drop, 73 W.
- Waves: drop grows with active SMs up to 1 wave (0.116 -> 0.221 V), then saturates; the partial last wave shows as a
  second step in the recovery. 1 wave = 35 CTAs on 35 usable SMs exactly.
- Duration: di/dt drop independent of cycle length (4 ms .. 125 ms all 0.24 V); settled droop grows 0.20 -> 0.25 V with the
  DVFS step (2630 -> 2780 MHz at ~35 ms).
- Measuring the PSU voltage only with summed current costs < 1 W / < 1 % in power (lane difference x slot current).

## 8b. freqmon vs all-SM (FREQMON=0) on the HDO4034A + CP500, 2026-11-17 (`spike_tests_vi/results/20261117_140046_run_all`
vs `20261117_143523_run_all_allsm`)

- With freqmon the GPU sits at ~2850 MHz / 3.1 A before every launch (the monitor keeps it awake); without it the GPU idles
  at ~180-900 MHz / 1.4 A. The kernel therefore starts from a cold DVFS state in all-SM mode.
- 01 (fp32 8192^3): di/dt drop 0.241 V @162 us (freqmon) vs 0.288 V @142 us (all-SM); settled 0.218 vs 0.245 V; current step
  13.0 vs 14.4 A (idle 3.1 vs 1.4 A, peak 19.5 A both). Drop / current step = 18.5 vs 20 mOhm: the larger drop is mostly the
  larger current step from the colder idle.
- Short kernels from a cold GPU run slowly: the 512-row GEMM takes 30.8 ms (all-SM) instead of 4.3 ms (freqmon) because the
  clock ramps from idle over ~20-30 ms; its drop is only 0.124 V / 7.2 A. Any kernel shorter than the DVFS ramp never reaches
  full power unless the GPU is pre-warmed.
- Locked clocks (03): both modes agree within 0.005 V at every lock (the lock pins the idle clock too), which confirms the
  pre-launch clock state, not the monitor's SM, is what differs.
- Waves (04): 1 wave = 36 CTA 0.244 V (all-SM) vs 35 CTA 0.220 V (freqmon); 0.5 wave 0.170 vs 0.117 V; saturation above 1
  wave in both modes. Kernel times are 20-30 % longer in all-SM mode for the short wave kernels (cold clock).
- Without freqmon the clock column is nvidia-smi's 100 ms max around the cycle (`clk_smi`), which is coarse and often misses a
  63 ms kernel: treat it as indicative only.

## 8c. `model_run/` — LLM inference power profile (2026-11-17 .. 11-20)

`sudo ./model_run/run_all.sh` profiles bf16 inference by phase (prefill N tok, decode M tok at batch 1 / B, sustained
back-to-back prefills) with the PMD2X poller started OUTSIDE the model process, nvidia-smi, the HDO stop-capture (whole phase
at 1 us, several 10 s captures for long phases) and host stamps per token / per engine step. Engines: HF transformers eager
(`llm_power_profile.py`) and vLLM 0.34 (`ENGINE=vllm`, `vllm_power_profile.py`, venv_vllm). `--op-hooks` records every
module's GPU interval (CUDA events in forward hooks), `--op-marker <layer>` adds a 10 ms GPU sleep as a scope alignment
notch; `plot_ops.py` attributes the 1 us scope power to q/k/v/o, attention core, gate/up/down, norms, lm_head.
Results (see `model_run/results/compare.md`, `*_summary.md`): 0.6B / 1.7B / 4B HF profiles; 1.7B prefill 16384 tok op
breakdown (attention core 35 % of the time, unfused RMSNorm 19 %, GEMMs 125-145 W, RMSNorm/o_proj 95-115 W, no op spike above
+4 A/100 us inside a prefill); vLLM decode b1 107 tok/s vs HF 37 tok/s. See `model_run/README.md`.

## 8d. 200 W power limit (all-SM, 2026-11-18, `spike_tests_vi/results/20261118_132537_run_all_allsm_pl200` + `..._134924_...` for 02)

- `POWER_LIMIT_W=200` raises the cap from the default 180 W to the GPU's max. 63 ms single cycle: PMD peak 234 W vs 221 W,
  CP500 peak 20.1 vs 19.5 A, di/dt drop 0.273 vs 0.288 V (within the cold-start scatter), settled 0.244 vs 0.245 V; locked
  clocks (03) identical to the 180 W run. 2-launch burst: PMD peak 249 vs 228 W.
- 5-launch burst (304 ms) at 200 W: PMD 25 ms bins 176 213 238 223 204 199 202 208 211 213 W; scope P 184 -> 202-225 W;
  nvidia-smi during the burst: clock 2700-2850 MHz (no pull-down), power.draw.instant 193-209 W, power.draw.average only
  45-78 W because the averaging window (~1 s) has not filled. So the cap acts on a ~1 s average: sub-second bursts run
  20-50 W (PMD) above the cap; only the multi-second sustained load earlier settled at nvidia-smi 180.0 W / PMD 190 W /
  2540 MHz. Cable PMD power also reads ~10 W above the GPU's own sensor.
- Cold-start bimodality in all-SM mode (no freqmon keeping the GPU awake): some cycles start from P8 (180 MHz) and the clock
  does not ramp for ~60-80 ms, so a 17 ms kernel takes 79 ms at ~600 MHz / 90 W (4-6 of 10 cycles in both the 180 W and
  200 W runs), a 1.6 ms wave kernel 4.4-5.5 ms, the 78 ms wave-56.9 kernel once 271 ms. Medians of short kernels in all-SM
  mode are therefore unreliable; compare per-cycle `kernel_ms` or pre-warm the GPU.

## 8e. Clock after a load step and the warm-clock option (2026-11-19, `spike_tests_vi/results/20261119_154758_lockflat`, `..._155520_01_baseline`)

- In-kernel clock after an idle -> load step (freqmon, 63 ms fp32 8192^3): ~2640 MHz for the first 30-35 ms, then 2750-2780
  MHz. Locks do not remove the step (lock 3090 / 2775 / 2700: still 2640 for 30 ms, then the lock value); locks <= 2632 MHz
  are flat from t = 0 (2550: 2550 throughout). The power limit is irrelevant for it (183 W < cap). It is a controller
  guardband after the current step, not exposed by nvidia-smi.
- The settled state survives an idle gap: a warm-up GEMM ending 0-20 ms before the measured launch gives 2750-2780 MHz from
  the first sample; at 50 ms the step partly returns, at >= 100 ms fully. Hence `--prewarm-gap-ms 20` / `PREWARM_GAP_MS=20`.
- Warm vs cold, 01 (freqmon on, 180 W): clock 2752 vs 2651 MHz, kernel 61.8 vs 63.6 ms, di/dt drop 0.268 vs 0.241 V, settled
  droop 0.269 vs 0.218 V (the whole kernel now runs at the higher V/F point), CP500 peak 19.8 vs 19.5 A, PMD peak 226 W both.

## 8f. Sample rate, rise time, HDO capture tools (2026-11-20)

- `--scope-sample-ns` / `SCOPE_SAMPLE_NS`: HDO 1-2-5 rates; 200 ms record -> 20 ns max (10 Mpts), 100 ms -> 8 ns; the driver lowers
  the rate to fit 12.5 Mpts. Transfer ~20 MB/s (2.5 s for two channels at 20 ns). csv decimated to >= 1 us, npz full rate float32,
  plots use min/max envelopes, onset / rise / slope smoothing is defined in time (20 us / 10 us), not samples.
- Current rise metric (`onset.rise_time`): idle -> plateau (1-5 ms after the onset), 10-90 %: 63 ms GEMM, warm clock: ~200 us
  (2.4 -> 18.6 A, max di/dt ~0.1 A/us); the ramp has a fast part (~16.8 A at 200 us) and a slow tail to the plateau at ~1 ms.
  `burst_NN_rise.png` = one panel with the value. The CP500 itself is <= 175 ns, so this is the board's input-current ramp.
- `hdo_capture/capture2s.py` (fixed-time 2-channel capture, optional GEMM inside, rise of the largest step) and
  `hdo_capture/clamp_step.py` (hardware edge trigger at full rate for an external PSU current step: 10-90 %, settle time,
  `rise.png`). The HDO edge trigger works at normal timebases (tested 2 ms .. 1 s records with HFREJ coupling); only the
  1 MSa/s long-memory mode of the campaign needed the stop-capture.
- all-SM warm-up length: 200-350 ms leaves a 20 ms current dip at +20..40 ms in ~half the cycles; >= 500 ms removes it but the
  first 5-10 ms start ~10 % low. Chosen: PREWARM_MS=500. In freqmon mode one launch (63 ms) is enough.
- A power-cycled HDO comes back with CFMT BYTE: 8-bit data decoded with WORD gains looked like 19.9 A at idle. Every tool now
  sets `CFMT DEF9,WORD,BIN` itself.

## 8g. 2026-11-20/23: the ~200 us current ramp, sample-interval study, 30-launch burst, WSL memory

- Current ramp at a kernel start (`hdo_capture/ramp_analysis.py`, `spike_tests_vi/ramp_workloads.sh`): 50 % point 88-113 us and
  RC tau 85-97 us for cuBLAS fp32/bf16, ws normal, ws compute-only (no DRAM loads), ws loader-only (no compute), 0.5 .. 57 waves,
  clock locks 1545-2750 MHz and amplitudes 6-20 A; 2nd-order fit 2-3 kHz, zeta 0.8-1.1; charge deficit 0.5-2.2 mC -> C_eff
  5-9 mF; fall at the kernel end 190-440 us. Workload-independent => the PSU-cable / input-capacitor / VRM-input electrical
  response, not SMs filling or the first-stage load (first wave's first stage = 432 KB = ~1 us at 448 GB/s).
- Sample interval vs spike depth (`hdo_capture/spike_fast.py`, hardware trigger on CH1 rising 5 A, HFREJ): dV in the first
  300 us = 0.200 V @0.4 ns, 0.192 @20 ns, 0.170 @50 ns, 0.158 @1 us, 0.136 @10 us (5 ms record); sub-50 ns windows add 15-20 %
  through ns-wide spikes (passive 1:1 lead, ground-loop pickup likely), 0.4 vs 20 ns only +4 %; current / power peaks do not
  change below 10 us (P peak -0.4 % at 10 us, -3 % at 100 us, -7 % at 1 ms = PMD2X). t_min of the slow event ~150 us.
- 30-launch burst (1.8 s) at 200 W, all-SM warm: the limiter clamps the GPU's INSTANT internal power at the cap within ~100 ms
  (nvidia-smi power.draw.instant pinned 199-200 W, clock 2850 -> 2670 MHz for the whole burst, PMD 220 -> 210 W, scope P
  225 -> 216 W, V 11.89 -> 11.91 V); `power.draw.average` only lags. Earlier "1 s average" statement was wrong.
- WSL died twice from memory (host 31 GB, WSL 15.5 GB + 4 GB swap): fixes = envelope-decimated summary figure, float32 records,
  csv <= 2 M rows, `onset.load_csv` (np.loadtxt) instead of genfromtxt, spike_fast streams per cycle; never run two heavy
  post-processing jobs concurrently; recommended `.wslconfig` memory=24GB swap=16GB. Never press AUTO on the scope during a run.
- HDO sample rates are 1-2.5-5 steps (20 MSa/s request silently becomes 10 MSa/s): driver snaps to that grid; 500 ms records
  are 40 ns, 200 ms records 20 ns.

## 8h. 2026-11-24: unified wave x size sweep (freqmon vs all-SM), 04b tile-layout test

Folders: `spike_tests_vi/results/20261124_151027_run_all_pl200_warm20_20ns` (freqmon, 35 usable SMs, warm-up = 1 launch of the
same GEMM + 20 ms gap) and `..._161019_run_all_allsm_pl200_warm20x500_20ns` (all-SM, 36 SMs, warm-up >= 500 ms + 20 ms gap);
both 200 W, 20 ns records, 3 single cycles per point, 18 wave targets x K in {2048, 8192} + the cuBLAS 8192^3 reference.
Tables / figures: `wave_sweep_summary.{csv,json,png}`, `overlay_04_waves_K<k>_*.png`, `overlay_04_waves_allK_*.png`, `report.md`.

- di/dt drop (idle - min in the first 300 us): rises with the SM count up to 1 wave (0.137 -> 0.247 V freqmon, 0.144 -> 0.284 V
  all-SM for K=2048), then flat for 1.5 .. 58 waves (freqmon 0.23-0.26 V, all-SM 0.26-0.29 V). All-SM is deeper by 0.01-0.06 V at
  every point (36 vs 35 SMs plus the lower all-SM idle current: 2.8-3.0 A vs 3.2-3.3 A with freqmon awake); K has no effect on
  the drop (the first 300 us only see the SMs filling). The whole-cycle minimum keeps falling slowly with waves (0.25 -> 0.28 V
  freqmon, 0.28 -> 0.31 V all-SM) because long kernels end at a lower rail voltage.
- Kernel time: K=8192 / K=2048 = 3.9-4.4x (median 4.2) for the same M, N: the extra ~5 % is the per-CTA prologue / epilogue
  (0.3 ms at K=2048 is 25 % fixed cost). The cuBLAS 8192^3 reference takes 60-61 ms vs 76-79 ms for wavegemm (58 waves).
- "Does the first wave of a big grid draw more power?" In the freqmon sweep the first-wave current (0.5-1.2 ms after onset,
  K=8192, clocks equal 2800-2820 MHz) is 15.4-15.6 A for 1-8 waves and 16.3-16.6 A for 16-58 waves. Test 04b
  (`spike_tests/04b_wave_shape.sh`: the same 35 CTAs as a 1280x896 square, a 256x4480 row sharing one A panel and an 8960x128
  column sharing one B panel) gives 15.45 / 15.54 / 15.69 A -> tile layout (L2 panel sharing) is NOT the cause. The all-SM sweep,
  whose warm-up is a fixed 500 ms, shows no step at all (17.0-17.7 A from 1 to 57 waves). The freqmon warm-up is ONE launch of
  the same GEMM (`--prewarm-ms 0`), so a 16-wave point was preceded by 22 ms of load, an 8-wave point by 11 ms, a 1-wave point by
  1.3 ms: the step follows the warm-up history, not the operation. Verified: with a 500 ms warm-up in freqmon mode (`..._175128_run_all_warm20x500_20ns`, default 180 W limit because the
  UAC prompt for 200 W was not accepted; the first-wave window is before the limiter reacts, clocks 2787 MHz for all three)
  the first-wave current is 16.26 / 16.31 / 16.51 A for 1 / 8 / 16 waves: the step is gone (residual +0.2 A at 16 waves is
  inside the 16.2-16.7 A cycle scatter). Rule: compare modes or shapes only with the same `PREWARM_MS`.
- Repeat check 2026-11-25 (`spike_tests_vi/results/20261125_093245_run_all_pl200_warm20x500_20ns`, freqmon, 500 ms warm-up, 200 W,
  waves 4 / 8 / 32 / 64 (M10240 N7168, 2240 CTAs) / 58.51 (8192^2), three passes of 3 cycles): first-wave current (0.5-1.2 ms)
  16.33 / 16.18 / 16.59 / 16.50 / 16.87 A, pass medians within 0.2 A of each other, clocks 2760-2820 MHz, dV300 0.25-0.27 V for
  every point. So with equal warm-up the wave count does not change the first-wave power (4 -> 64 waves: +0.2 A). The one
  shape that is consistently higher is the 8192^2 square (+0.4 A over the 64-wave 10240x7168 grid, in every pass and for the
  whole kernel, 17.0-17.3 vs 16.8-16.9 A): a shape effect (power-of-two N = 32 KB row stride?), to be checked with Nsight Compute.
  One cycle (pass 2, 58.51 waves, cycle 2) ran 119.5 ms instead of 79.4 ms with normal current -> host-side stall, not excluded.
- `ncu_profile/` (new): `wave_kernel.py` (the same wavegemm extension, standalone), `run_ncu.sh` (DRAM / L2 / L1 bandwidth,
  hit rates, SM / FMA-pipe throughput, stalls for w1 / 4 / 8 / 32 / 58.51 / 64, `--clock-control none`), `summarize_ncu.py`.
  ncu 2025.4.1 is extracted under `~/nsight-compute` (WSL repo .deb, `dpkg-deb -x`, no root). BLOCKED on 11-25:
  `ERR_NVGPUCTRPERM` even as root -- on WSL2 the counter permission is the Windows driver's: NVIDIA Control Panel > Desktop >
  Enable Developer Settings > Developer > Manage GPU Performance Counters > "Allow access to the GPU performance counters to
  all users" (may need a reboot). Then `cd ncu_profile && ./run_ncu.sh`.
- 04b runs carry the tag suffix `_square/_row/_column`; `summary_wave.py` labels them `kernel=layout` (csv only) and
  `overlay_summary.py` skips them, so they never enter the sweep statistics of the folder they were resumed into.

## 9. Known traps (all handled in code, listed so nobody re-discovers them; the dated list is `BUGS_AND_FIXES.md`)

- DSOX1202A: 12 V offset needs >= 0.5 V/div with a 1:1 probe; `:WAVeform:POINts?` after an AUTO-sweep `:SINGle` is decimated
  (never cache it); `:DIGitize` / `:MEASure` without a trigger hang the parser (USBTMC device clear on open); CH1 of the first
  unit was dead; records are 1.00 us for windows >= ~40 ms, 1.05 us for short ones; 2 s maximum at 1 us.
- usbipd: a Windows network drop wedges both vhci ports (workers in D state): `usb_attach.sh` does the sysfs detach +
  `usbipd detach` + attach; last resort `wsl --shutdown`. The link also drops after ~12 min idle, so every scope test re-attaches.
- CUDA: in-process freqmon blocks any launch on the legacy default stream and the lazy load of a new module -> workload on its
  own stream, stamp kernel warmed before freqmon starts. WSL wall clock drifts; alignment uses GPU `%globaltimer` / monotonic.
- nvidia-smi board power on this GPU reads lower than the PMD cable power (180 vs 189 W), so the PCIe-slot share cannot be
  derived from it. Spec bounds: slot 12 V 66 W, 3.3 V 9.9 W.
- Scope force behind a USB round trip (found 2026-11-17 with the ~9 ms ws single cycles): the old worker polled `done()`
  every 200 ms while waiting for the launch; when the launch event arrived during that query the force waited for the
  ~10 ms USB reply, so on a short kernel the record started after the kernel had ended (the 63 ms cuBLAS cycles of 01-04
  were never affected: the onset still lay inside the 15 ms pre-trigger and all metrics align on the onset). Fix in
  `scope_worker`: no status query between `armed.clear()` (50 ms before the launch) and the force; `force_cmd_ms` is now
  measured from the main process's launch time. Older runs' `force_cmd_ms` (~0.4 ms) understate the true delay.
- Never edit a running bash test script (bash reads it incrementally: run_all.sh died with a syntax error at its last
  line after 04 on 2026-11-17); never `pgrep -f` / `pkill -f` a pattern that is in the caller's own command line (exit 144
  kills the session shell, waiters never return): kill by PID.
- Orphan nvidia-smi loggers after a killed run: `pkill -INT -x nvidia-smi`. Killing with a pattern that matches the shell
  itself returns 144.
- Shell: piping a list into `while read` while calling spike_eval eats the pipe (use `mapfile`); `$(scope_args)` is
  word-split, so channel names travel with underscores (`--v-name PSU_12V_cable`).
- LeCroy HDO4034A (verified 2026-11-17, firmware 10.4.0): remote control must be USBTMC (TCPIP leaves USB commands
  unanswered; changing it re-enumerates the device -> re-attach); it advertises USB488 REN_CONTROL but STALLs it (pyvisa-py
  open fails -> `_visa_open` makes that request best-effort); bulk-in requests >= 64 KB are answered in 65 521-byte transfers
  padded to 65 536 bytes without a short packet (pyvisa-py waits forever) -> `LecroyHDO.CHUNK = 32768`; no edge level ever
  completed a SINGLE acquisition and FRTR = stop, so the driver captures by stopping (section 5). The trigger level readback
  is in "V" even on the amperes channel. `lecroy_check.sh` still verifies offsets, TDIV snap, sample rate and probe units; its
  own record is idle by construction. First real HDO numbers (fp32 8192^3): cable di/dt drop 0.243 V at 153 us, settled
  0.218 V, CP500 2.5 -> 14.9 A peak (PMD2X both cables 19.3 A: check which conductors the clamp encloses).

## 10. Porting checklist (new GPU / scope / machine)

1. `pip install -r requirements.txt` in a venv; CUDA toolkit matching torch; `ninja` in the venv.
2. `config.env`: scope model (`SCOPE_METER`), channels, probe scale, usbipd bus ids; names for the legends.
3. `sudo ./00_preflight.sh` -> all green, shape probe done (`logs/auto_shape.env`). For LeCroy: scope Remote = USBTMC, then `sudo ./lecroy_check.sh`.
4. `sudo ./run_all.sh 01` first (2-3 min), check `report.md` and a `burst_00.png`, then the full sequence.
5. `ws_gemm`: `venv/bin/python ws_gemm/ws_bench.py` first (builds `wsgemm.cu`, prints the variants and their smem; a variant
   that needs more shared memory than the GPU allows is reported by `ensure()`), then `sudo ./run_all.sh C` as the short check.
6. Different SM count / memory / clocks are picked up automatically (wave targets, supported clocks, candidate shapes).
   Only `freqmon_excl.cu`'s 96 KB smem assumption (needs >= 100 KB per SM) and `wavegemm.cu`'s 72 KB pad may need
   `--freqmon-smem-kb` / an edit on GPUs with less shared memory.
