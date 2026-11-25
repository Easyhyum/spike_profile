# spike_profile — GPU 12 V rail di/dt measurements of GEMM workloads

What a GPU kernel does to its 12 V supply in the first microseconds to milliseconds after launch, measured on the same launch
with an oscilloscope (rail voltage + total cable current), the ElmorLabs PMD2X (per-cable V/I at ~1 kHz), nvidia-smi and an
in-kernel SM clock monitor. Developed on an RTX 5060 Ti (Windows 11 + WSL2, instruments through usbipd-win); every sweep
derives its targets from the attached GPU.

Two evaluations share one engine and branch here:

```
spike_profile/
  spike_eval.py          measurement engine: one run = N cycles of one workload, all instruments, onset-aligned records + figures
  onset.py               kernel-start detection on a voltage record (shared by every summary / plot)
  pmd2_fast.py           PMD2X serial poller + nvidia-smi logger;   plot_spike_csv.py  per-run voltage figures
  wavegemm.cu            fp32 GEMM, one 256x128 tile per CTA, 1 CTA/SM -> exact wave count
  freqmon_excl.cu        in-kernel SM clock monitor (owns one SM);   gtstamp.cu  GPU %globaltimer launch stamps
  spike_tests/           evaluation 1 — VOLTAGE wiring (Keysight DSOX1202A, edge trigger): 00 preflight, 01 baseline,
                         02 duration, 03 clock sweep, 04 unified wave x size sweep, 04b tile layout, report / summaries
  spike_tests_vi/        evaluation 2 — VOLTAGE + CURRENT wiring (LeCroy HDO4034A + CP500 clamp, stop-capture): the same
                         scripts as symlinks into ../spike_tests, its own config.env, README and results_summary/
  HANDOFF.md             full handoff: topology, measurement logic, tests, metrics, reference numbers, traps, porting checklist
  BUGS_AND_FIXES.md      every bug found during the campaign, cause and fix, dated
```

`spike_tests_vi/results_summary/` keeps the report and summary figures of the three decisive runs (freqmon sweep, all-SM
sweep, 2026-11-25 repeat check); raw records (`burst_NN.npz`, 10 M points each), logs and virtual environments are not in the
repository (`.gitignore`). HANDOFF.md also describes sibling campaigns (`ws_gemm/`, `model_run/`, `hdo_capture/`,
`ncu_profile/`) that live in the full working tree, not in this repository; `spike_eval.py --kernel ws` needs `ws_gemm/`.

## Quick start

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt      # torch (CUDA), numpy, matplotlib, pyvisa, pyvisa-py, pyusb, pyserial, ninja
cd spike_tests_vi                   # or spike_tests (voltage-only wiring)
vi config.env                       # scope model and channels, probe scale, usbipd bus ids
sudo ./00_preflight.sh              # GPU + PMD + scope check, picks the strongest-spike GEMM for this GPU
sudo ./run_all.sh                   # 01 02 03 04 -> results/<stamp>_run_all/report.md
sudo env FREQMON=1 POWER_LIMIT_W=200 PREWARM_GAP_MS=20 PREWARM_MS=500 SCOPE_SAMPLE_NS=20 WAVE_K="2048 8192" ./run_all.sh 04
sudo env FREQMON=0 ...  ./run_all.sh 04                  # all SMs (no clock monitor), same knobs
sudo env RESUME_DIR=results/<folder> ... ./run_all.sh 04 # continue an interrupted sequence in the same folder
```

Everything runs as root (serial port, USBTMC). Clock locks and power limits go through `spike_tests/clockctl.py`, an elevated
nvidia-smi helper on the Windows side (one UAC prompt per sequence).

## Main results (RTX 5060 Ti, HANDOFF.md section 8)

- A 63 ms fp32 8192^3 GEMM steps the cable current 2.5 -> 19.5 A in ~200 us (an RC ramp of the board input, workload
  independent) and drops the PSU cable 0.24-0.29 V within the first 150 us; the drop grows with active SMs up to 1 wave and is
  flat from 1 to 64 waves; K (kernel length) does not change it.
- All-SM mode (no clock monitor) drops 0.01-0.06 V deeper than freqmon mode at every wave count (36 vs 35 SMs, lower idle).
- The clock controller holds ~2640 MHz for 30-40 ms after a load step before settling at ~2750 MHz; a warm-up GEMM ending 20 ms
  before the measured launch removes the step (`PREWARM_GAP_MS`, `PREWARM_MS`). Compare shapes only with equal warm-up: an
  apparent "+1 A in the first wave of big grids" was the warm-up history (one launch of the same GEMM), not the operation.
- The 180 / 200 W power cap acts on the GPU's internal instantaneous power within ~100 ms of a sustained load.
