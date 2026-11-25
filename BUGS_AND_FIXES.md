# Bugs found during the campaign and how they were fixed

Chronological record of everything that produced wrong or missing data, with the fix now in the code. Dates are 2026.
Read this before trusting any result folder older than the fix date (the folder name carries the date).

| date | area | symptom | cause | fix / status |
|---|---|---|---|---|
| 11-11 | DSOX1202A | 7 633-point records at 10.9 us instead of 1 us | `:WAVeform:POINts?` after an AUTO-sweep `:SINGle` returns a decimated count that was cached | re-query the point count per read |
| 11-11 | DSOX1202A | edge trigger fired in 3 of 10 cycles | trigger hysteresis at 0.5 V/div | software `:TRIGger:FORCe` after the launch; all plots align on the voltage onset |
| 11-11 | collector | burst-to-record mapping shifted after a spurious trigger | record completed before burst 0 | drop records done before burst 0, de-duplicate per burst (larger drop wins) |
| 11-11 | usbipd | both vhci ports wedged after a Windows network drop | workers in D state | `usb_attach.sh`: sysfs detach + `usbipd detach` + attach |
| 11-12 | shell | only the first wave target ran | `while read` pipe loop: spike_eval consumed the pipe's stdin | `mapfile` + `< /dev/null` |
| 11-13 | onset | "voltage stays at max" in an overlay (late force) | onset threshold caught a 10 mV noise dip | threshold = idle - max(6 sigma, 10 mV, 30 % of the deepest excursion) |
| 11-13 | report | tests 01/02 missing from report.md | tag regex | fixed in summary_runs.py |
| 11-17 | scope worker | records of ~9 ms kernels started after the kernel | the force waited behind a ~10 ms USB status query | no status query between arm and force; `force_cmd_ms` measured from the launch |
| 11-17 | HDO4034A | `*IDN?` never answered over USB | scope Remote = TCPIP | set Remote = USBTMC on the scope (re-attach afterwards) |
| 11-17 | HDO4034A | pyvisa-py open failed with a pipe error | scope advertises USB488 REN_CONTROL but STALLs it | best-effort REN in `_visa_open` |
| 11-17 | HDO4034A | waveform reads timed out | >= 64 KB requests answered in 65 521-byte transfers padded to 65 536, no short packet | `LecroyHDO.CHUNK = 32768` |
| 11-17 | HDO4034A | records idle, no load; every edge level failed | FRTR = stop, the record is the window BEFORE the force; forcing 0.5 ms after the launch recorded the 100 ms before it | stop-capture: FRTR at launch + window - pre; edge level parked out of range |
| 11-17 | wiring | CP500 read 12 A while the PMD2X read 19 A | the clamp enclosed only one of the two 12 V cables | re-clamped; verified 19.5 A vs 19.2 A |
| 11-17 | plots | legend said "Keysight DSOX1202A" on HDO runs | hard-coded name | name from the record's `*IDN?` |
| 11-17 | run_all | sequence died after test 04 with a bash syntax error | the script was edited while it was running | never edit a running script; resume with `RESUME_DIR` |
| 11-17 | shell | session shell killed (exit 144), waiters never returned | `pgrep -f` / `pkill -f` matched the caller's own command line | kill by PID only |
| 11-17 | all-SM mode | kernel time and clock columns empty | the `%globaltimer` stamp kernel was only loaded with freqmon | stamp kernel always loaded; host-time fallback; nvidia-smi clock column `clk_smi` |
| 11-18 | model_run | Qwen3-4B (and once 1.7B) died with CUDA "unspecified launch failure" | the PMD poller was forked from the CUDA process (WSL dxg handle duplicated into the child); standalone loops never failed | run_all starts the poller outside the model process (`--pmd-external`); retries per model |
| 11-18 | model_run | second attempt had no PMD data | the first attempt's poller still held the serial port | pollers stopped by PID before each try |
| 11-19 | model_run | op timeline overran the prefill by 6-9 % | CUDA-event clock vs WSL monotonic differ by 1-4 % | two-point (start / end) mapping |
| 11-19 | model_run | per-prefill "latency" 893 ms for a 2 s prefill | HF `generate(max_new_tokens=1)` LogitsProcessor stamps the CPU enqueue time | completion-time stamps for prefill phases |
| 11-19 | model_run | "16384-token" phase ran 12 001 tokens | prompt text repeated too few times | prompt built for the requested length, asserted |
| 11-19 | HDO4034A | 2.8 s window recorded as 2.0 s and missed the phase start | 1-2-5 TDIV snap rounded DOWN below the requested window | driver steps TDIV up to the next 1-2-5 value |
| 11-19 | plots | x tick labels unreadable on long windows | one label per ms | 1-2-5 label step, unit as minor grid |
| 11-19 | WSL | VS Code / Claude session dropped repeatedly | WSL VM at its 15.5 GB cap (host 31 GB, page cache 10 GB) while plotting 10 M-point records; OOM killer took the IDE server | envelope decimation, model freed before plotting, no concurrent analysis, page-cache drop; recommended `.wslconfig` memory / `autoMemoryReclaim=gradual` |
| 11-19 | model_run | PMD samples attributed to the wrong op | one-point scope alignment | GPU-sleep marker (`--op-marker`) stamped by events, scope aligned on its notch |
| 11-20 | HDO4034A | 19.9 A at idle after a scope power cycle | CFMT reset to BYTE | every tool sets `CFMT DEF9,WORD,BIN` |
| 11-20 | HDO4034A | 20 ns request silently became 10 MSa/s | rates are 1-2.5-5 steps | driver snaps to that grid |
| 11-20 | all-SM warm-up | 20 ms current dip at +20..40 ms in half the cycles | 200-350 ms warm-up too short after P8 | `PREWARM_MS=500` |
| 11-20 | nvidia-smi | clock column useless at 100 ms polling | the driver refreshes `clocks.sm` only every ~0.5 s (power 100 ms, utilization 1.8 s) | documented; clocks from freqmon |
| 11-20 | measurement hygiene | two experiments overlapped on the same GPU / scope (one started by another session) | no check before starting | check `ps` for spike_eval / llm_power_profile / pmd2_fast before any run |
| 11-23 | run_all | sequence interrupted in 02 (x5 burst) | shutdown | `RESUME_DIR=<folder>` continues in the same folder; the 1-cycle partial run was set aside as `partial_*` |
| 11-24 | usbipd | scope vanished mid-sweep ("no scope with VID"), `usb_attach.sh` said "already attached" and the all-SM wave sweep aborted after 6 runs | the vhci port looked healthy (state 006, live socket) while every URB failed (-104) after a link drop; the stale check also read the wrong status column | `usb_attach.sh` probes the device (GET_DESCRIPTOR via pyusb) and re-attaches when it does not answer; vhci column fixed; 04 skips complete targets (`SKIP_DONE`) so a sequence can be resumed with `RESUME_DIR` |
| 11-24 | test design | 01 (fixed shape), 02 (rows / bursts) and 04 (waves) measured overlapping things | three sweeps for one question | 04 is now the unified wave x size sweep (`WAVE_K`, cuBLAS reference `WAVE_REF`), tags carry K, M, N; overlays per K and all K together |
| 11-24 | summary | 32-wave K8192 point (freqmon sweep): two cycles with 0.04 / 0.02 V "drop" | the measured kernel sits ~30 ms early in those two records (onset fell to 0, idle read 11.93 V) | cycles whose idle is > 0.08 V below the folder's idle are flagged `bad_record` and left out of the medians; the point will be re-measured |
| 11-24 | collector | 20 ns runs with 10 M-point records (cuBLAS reference, 58.5 waves) saved 1-2 of 3 cycles; no warning visible | the 30 s silence budget was measured from the record's arrival, but processing one 10 M-point record (crop, onset, npz, csv, figures) takes > 30 s, so the loop ended before records 1-2 were read; `run_eval`'s `grep -v warn` also hid the engine's "WARNING: scope delivered" line | budget counted after processing (60 s); run logs keep WARNING lines; the four affected points re-measured |
| 11-24 | warm-clock crop | all-SM sweep: every K=2048 sub-wave point (0.3 ms kernels) had "idle" 11.93-12.06 V and tiny drops | `crop_to_last_load` smoothed the voltage over 1 ms, so a 0.3 ms load vanished and the 500 ms warm-up was kept as "the last load"; the onset then landed inside the warm-up | smoothing scaled to the load length (min 1 ms, on/5); `recrop_warm.py` re-applies the crop to saved npz records (18 records of the all-SM sweep fixed) |
| 11-24 | analysis | first-wave current +1 A for >= 16 waves (freqmon sweep) looked like a property of big grids | the freqmon warm-up is one launch of the same GEMM (`--prewarm-ms 0`), so the load history 20 ms before the measured kernel scales with the kernel length; the all-SM sweep (500 ms warm-up) has no such step and the 04b tile-layout test rules out L2 panel sharing | verified: freqmon + `PREWARM_MS=500` gives 16.3 / 16.3 / 16.5 A for 1 / 8 / 16 waves (`..._175128_run_all_warm20x500_20ns`); compare modes with the same `PREWARM_MS`; HANDOFF 8h |
| 11-24 | summary | 04b layout runs (`RESUME_DIR` into the freqmon sweep folder) were counted as four extra 1-wave points in `wave_sweep_summary.json` | same tag prefix `t04_K8192_..._w1.0_35cta` | `summary_wave.py` tags `_square/_row/_column` runs `kernel=layout` (csv only); `overlay_summary.py` skips them |
| 11-25 | ncu | Nsight Compute in WSL2: `ERR_NVGPUCTRPERM` for user and root | the performance-counter permission belongs to the Windows driver, not to the WSL kernel | NVIDIA Control Panel > Developer > Manage GPU Performance Counters > allow all users; ncu itself runs from a `dpkg-deb -x` extract (`~/nsight-compute`), no root install needed |

## Open issues

- CH4 (PCIe slot 12 V) probe reads pegged since 11-19; the lane comparison needs it re-seated or replaced.
- The CUDA launch-failure root cause (dxg handle in a forked child) is inferred from bisection, not from a driver log; the
  Windows log shows only `nvlddmkm` event 153 / LiveKernelEvent 141 at each failure.
- Op-level attribution inside vLLM needs a torch.profiler timeline (module hooks do not see fused / CUDA-graph kernels).
- In all-SM mode short kernels are bimodal (P8 cold start): compare per-cycle values or use `PREWARM_*`.
- The freqmon 32-wave K8192 record frame shift (kernel ~30 ms early in 2 of 3 records, 11-24) is excluded by `bad_record` but not explained (stop-capture lag?).
- Nsight Compute campaign (`ncu_profile/run_ncu.sh`) waits for the Windows-side performance-counter permission (11-25).
- 04b `rows2` layout (512x2240) is skipped: 2240 is not a multiple of the 128-wide tile; use 512x2304 (36 CTAs) if a two-row variant is wanted.
