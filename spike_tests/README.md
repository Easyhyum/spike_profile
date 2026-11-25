# spike_tests — GPU 12 V rail di/dt measurement sequence

Measures what one GEMM launch does to the 12 V supply: PMD2X (1 ms, both cables), nvidia-smi (100 ms), the in-kernel
SM-clock monitor `freqmon` (10 us, holds one SM) and an oscilloscope (1 us, two voltage channels or V x I) on the
**same launch**. Works on whatever GPU is attached: the workload shape, the duration points and the wave targets are
derived from the device at preflight.

## Quick start

```bash
cd ~/matmul_test/spike_tests
vi config.env                 # channels / probe / usbipd bus ids (no workload settings here)
sudo ./00_preflight.sh        # GPU + PMD + scope check, then find_max_shape.py picks the strongest-spike GEMM (~1 min)
sudo ./run_all.sh             # 01 02 03 04 -> results/<stamp>_run_all/report.md   (03 needs one UAC "yes" on Windows)
sudo ./run_all.sh 01 04       # a subset; every script alone also gets its own results/<stamp>_<name>/ folder
sudo env FREQMON=0 ./run_all.sh   # all-SM mode: no in-kernel clock monitor, every SM runs the GEMM -> results/<stamp>_run_all_allsm/
sudo env POWER_LIMIT_W=200 ./run_all.sh   # whole sequence at a raised board power limit (max of this GPU) -> results/<stamp>_run_all_pl200/
sudo env PREWARM_GAP_MS=20 ./run_all.sh   # warm clock: a GEMM ends 20 ms before every measured launch -> settled clock from t = 0 (LeCroy only)
sudo env FREQMON=0 PREWARM_GAP_MS=20 PREWARM_MS=500 SCOPE_SAMPLE_NS=20 ./run_all.sh   # all-SM, warm clock, 20 ns samples (50 MSa/s)
sudo env POWER_LIMIT_W=200 PREWARM_GAP_MS=20 SCOPE_SAMPLE_NS=20 WAVE_K="2048 8192" ./run_all.sh 04   # the unified wave x size evaluation only (+ cuBLAS reference); FREQMON=0 PREWARM_MS=500 for all-SM
```

`SCOPE_SAMPLE_NS` (default 1000 = 1 us): scope sample spacing. On the HDO4034A 1-2-5 steps down to 20 ns for the 200 ms
record of the 63 ms GEMM (10 Mpts/channel, 2.5 s transfer per channel pair); longer records lower the rate automatically to
fit 12.5 Mpts (5-launch burst: 40 ns). `burst_NN.csv` is decimated to >= 1 us (summaries, overlays), `burst_NN.npz` keeps the
full rate (float32, ~90 MB per cycle at 20 ns), figures use min/max envelopes. `PREWARM_MS` (default = one launch): all-SM
mode needs ~500 ms of warm-up load (the GPU wakes from P8; with 200-350 ms one cycle in two still shows a 20 ms current dip
at +20..40 ms; with >= 500 ms the dip is gone but the first 5-10 ms start ~10 % low). In freqmon mode one launch is enough.

`PREWARM_GAP_MS` (config.env, default empty): after a load step the 5060 Ti's clock controller holds ~2640 MHz for 30-35 ms
before settling at ~2750 MHz, whatever the lock or power limit (locks at 2775 / 3090 MHz still show the step; only locks
<= 2632 MHz are flat from t = 0). The warm state survives an idle gap of about 20-30 ms, so one extra GEMM of the measured
shape ending 20 ms before the launch gives 2750 MHz from the first sample while the 12 V current is back at idle (< 1 ms).
The record is cropped to the last load segment (`crop_to_last_load`) so the warm-up is not in the figures or metrics; the
PMD window is cropped likewise. Needs the LeCroy stop-capture path (the DSOX edge trigger would fire on the warm-up).

`POWER_LIMIT_W` (config.env, default empty = the GPU's default limit): run_all.sh starts the elevated Windows nvidia-smi
server once (one UAC prompt at the start instead of at 03), applies `-pl <W>` (verified through `power.limit`), keeps the
server for the whole sequence (03 reuses it) and restores the default limit at the end; the server also restores it on its
own exit. Allowed range = `clockctl.py limits` (150-200 W on the 5060 Ti). The limits in force are saved per run in
`marks.json["power_limits"]`, drawn on every power axis (dashed default/current, dotted max) and printed in report.md.

`FREQMON` (config.env, default 1): with 1 the `freqmon` kernel holds one SM and gives the 10 us SM clock in every figure;
with 0 nothing but the GEMM runs (wave targets use all SMs, `--no-freqmon`), the clock columns come from nvidia-smi (100 ms,
`clk_smi` in `runs_summary.csv`; the clock sweep plots use it or the locked target), and the freqmon / combined overlays are
skipped. Use both modes to check that the monitor's one SM does not change the di/dt drop.

## Sequence

| # | script | what | result files |
|---|---|---|---|
| 00 | `00_preflight.sh` | hardware go/no-go; `find_max_shape.py` probes dtype x size (fp32/bf16 x 4096..16384), ranks by mean power, tie-break = 1 launch closest to 60 ms -> `logs/auto_shape.env` (AUTO_DTYPE, AUTO_SIZE, AUTO_DURATION_ROWS, AUTO_SMS) | `logs/auto_shape.json` |
| 01 | `01_baseline.sh` | CYCLES single cycles of the max shape; di/dt drop, settled droop, PMD peak, kernel clock, lane comparison CH(V) vs CH(V2) | `runs_summary.csv`, per-run figures |
| 02 | `02_duration.sh` | cycle length: M = 1/16, 1/4, 1/1 of the max shape (short / mid / long) + bursts of 2 and 5 launches (`BURSTS`, ~125 / ~315 ms) | `runs_summary.csv` |
| 03 | `03_clock.sh` | di/dt vs graphics clock: default, max, max x 0.9 .. 0.5 (supported-clock list of the device); CLOCK_CYCLES each | `clock_sweep_summary.csv/png`, `overlay_03_clock_CH1_CH2.png`, `overlay_03_clock_freqmon.png` |
| 04 | `04_wave.sh` | **unified wave x size sweep**: `wavegemm.cu` (one 256x128 tile per CTA, 1 CTA/SM) with CTAs = wave x usable SMs; 0.5 .. 2.0, 3 .. 32 and the max shape, each at every K in `WAVE_K` (default SIZE; e.g. `WAVE_K="2048 8192"` spans short and long cycles, replacing 02), plus the cuBLAS max shape as reference (`WAVE_REF=1`, the old 01); WAVE_CYCLES each | `wave_sweep_summary.csv/png`, `overlay_04_waves_CH1_CH2.png` (1 wave in red), `overlay_04_waves_freqmon.png` |
| 04b | `04b_wave_shape.sh` | 1-wave tile-layout test: the same `USABLE` CTAs as a square (1280x896), a row (256x4480, one shared A panel) and a column (8960x128, one shared B panel), K = `WAVE_K`; answers whether the grid layout (L2 panel sharing) changes the first-wave power. Tags end in `_square/_row/_column`; `summary_wave.py` keeps them out of the sweep statistics (`kernel=layout`, csv only) | per-run figures, csv rows |
| — | `run_all.sh` | 00 + selected tests + `make_report.py` | `report.md`, `report.json`, `runs_summary.*` |

Removed as redundant (kept under `legacy/`): PMD-only run (every test records PMD), two-cable test (every scope test
records both voltage channels), separate V x I test (a live current probe switches every test to V x I automatically),
bf16-vs-fp32 (the shape probe compares dtypes).

Terminology: one launch per measurement = **single cycle**; several back-to-back launches = **burst (K launches)**.
GEMM shapes are given as M N K (C[MxN] = A[MxK] B[KxN]).

## Measurement order per cycle

PMD2X polling -> scope armed (Arm Event Register confirmed) -> freqmon running -> `%globaltimer` stamp -> kernel launch
-> kernel end -> stamp -> scope window ends (15 ms pre-trigger + load + 15 ms post) -> fetch -> re-arm. PMD stops after the
last cycle. The scope edge trigger (idle - 0.15 V) misses randomly at 0.5 V/div (hysteresis), so `spike_eval.py` forces
the trigger 0.5-0.7 ms after the launch when the edge has not fired; all plots align on the voltage onset (`onset.py`), so
forced and edge-triggered records are equivalent (`edge_triggered` is still logged). Since 2026-11-17 the scope process
sends no status query between the arm handshake and the launch (a `done()` query is a ~10 ms USB round trip that used to
delay the force past the end of short kernels); `force_cmd_ms` is measured from the launch itself.

## Metrics (`summary_runs.py`, per cycle; report shows medians)

- `didt_drop_300us`: idle - minimum within 300 us after the onset (the di/dt event), `t_min_us` its time
- `scope_i_rise_10_90_us` (+ `_rise_start_us`, `_t10_us`, `_t90_us`, `scope_didt_A_per_us`, `scope_didt_max_A_per_us`): current rise
  at the kernel start, idle -> 1-5 ms plateau; also drawn in `burst_NN_rise.png` (one panel with the number written on it)
- `droop_settled`: idle - median during the load; `drop_cycle_min`: idle - minimum over the whole cycle
- `lane_idle_mV / lane_load_mV / lane_max_mV`: CH(V2) - CH(V) at idle, during the load, maximum instantaneous (20 us smoothed)
- `pmd_p_max / pmd_i_max`: PMD2X peak (PCIE1 + PCIE2); `clk_kernel`: median freqmon clock during the cycle
- 03: the same per clock setting (`summary_clock.py`); 04: per wave + GFLOP, TFLOP/s, arithmetic intensity (`summary_wave.py`)

## Figures

Per run: `burst_NN.png` (4 x windows: 1 us / 10 us / 100 us / 1 ms ticks; both channels, P and I columns when a current
probe is present; every panel labels its max (P, I) or min (V, with the drop vs idle); every power axis carries the board
power limits from `nvidia-smi -q -d POWER`, saved per run in `marks.json` as `power_limits`: dashed = default / current cap,
dotted = max power limit), `burst_NN_100ms.png` / `burst_NN_200ms.png` (the same P / I / V columns over one 0 .. 100 ms and 0 .. 200 ms window, 1 ms grid),
`burst_NN_freq.png` (same windows + in-kernel clock), `burst_NN_V_scope.png`, `_V_scope_0-300us.png`,
`_V_pmd.png`, `_V_overlay.png` (scope V + PMD V + nvidia-smi + freqmon), `_V_scope_ch<N>.png` (second channel + difference).
Per sweep: `overlay_*_CH1_CH2.png` (every setting overlaid, colour = clock or waves, same values -> line style, opacity =
cycle index, legend below, colour bar right; the exact 1-wave setting in red) and `overlay_*_freqmon.png`.
`replot_burst.py <run>` and `overlay_summary.py <dir> --param clk|waves --signal voltage|freq` rebuild them.

## Related campaigns

- `../spike_tests_vi/`: current + voltage version of this sequence (symlinked scripts, own config.env).
- `../ws_gemm/`: warp-specialized bf16 GEMM campaign (`wsgemm.cu`, loader pacing / soft starts / kernel variants) with its
  own `pacing.sh`, `variants.sh`, `run_all.sh` and `summary_ws.py`; it symlinks `common.sh`, `check_hw.py`, `usb_attach.sh`
  and `summary_runs.py` from here. The engine `../spike_eval.py` gained `--kernel ws` plus `--ws-*` knobs for it; the default
  behaviour of this folder's tests (`--kernel cublas` / `wave`) is unchanged. Nothing from that campaign lives in this folder.

## Hardware notes

- LeCroy HDO4034A (+ CP500): `SCOPE_METER=2`, `I_SCALE=1` for the ProBus probe, dry run with `sudo ./lecroy_check.sh`; see `../spike_tests_vi/README.md`.

- Channels (config.env): `V_CH` = trigger / primary 12 V (CH1 = PSU cable), `V2_CH` = second 12 V (CH2 = PCIe slot), names
  in `V_NAME` / `V2_NAME`. For the real campaign set `V2_CH=` empty and `I_CH`, `I_SCALE`, `I_OFFSET` for the current probe.
- DSOX1202A: 0.5 V/div needed for a 12 V offset with a 1:1 probe; `:WAVeform:POINts?` after an AUTO-sweep `:SINGle` reports
  a decimated record, so the point count is re-queried per read; records are 1.00 us/sample for windows >= ~40 ms (1.05 us
  for short ones); 2 s maximum at 1 us.
- usbipd: a Windows network drop wedges both vhci ports (workers in D state); `usb_attach.sh` detaches via sysfs and
  `usbipd detach`, then re-attaches; the runner does this before every scope test. Since 11-24 it also probes an "attached"
  device (GET_DESCRIPTOR through pyusb) and re-attaches when it does not answer, the failure mode that ended a sweep with
  "no scope with VID". Last resort `wsl --shutdown`. Interrupted sequences continue with `RESUME_DIR=<folder>`; 04 skips
  targets that already have a complete run (`SKIP_DONE=1`).
- Clock locks: WSL nvidia-smi cannot lock; `clockctl.py` runs an elevated nvidia-smi server on Windows (one UAC prompt).
  Locks above the power-limited clock are accepted but not held under load (freqmon shows the real clock).
- Kernel geometry of any launch (grid, block, smem, CTAs/SM) can be printed with `launch_hook.c` (LD_PRELOAD, see file header).
- PMD2X measures the two PCIe cables; the PCIe slot current is not measured, so PMD power is the cable share only.
