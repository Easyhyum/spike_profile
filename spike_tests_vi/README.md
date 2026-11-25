# spike_tests_vi — current + voltage campaign

Same sequence and scripts as `../spike_tests` (every `*.sh` / `*.py` here is a symlink), with its own `config.env`,
`logs/` and `results/`:

- CH1 = current probe, all 12 V current (1 .. 90 A; `I_CH=1`, `I_SCALE` = A per V of the probe output, or 1 for a ProBus probe
  such as the CP500 that the scope already reads in amperes; screen 12 A/div centred at 45 A)
- CH2 = PSU 12 V cable (`V_CH=2`), CH4 = PCIe slot 12 V (`V2_CH=4`) -- the wiring used with the HDO4034A on 2026-11-17
- trigger: DSOX = rising edge on the current channel (`TRIG_I_LEVEL` = 3 A) with the software force fallback; HDO = stop-capture
  (see below). The onset alignment works the same for both.
- results carry V, I and P = V x I per 1 us sample: `burst_NN.png` gets the P and I columns, `runs_summary.csv` / `report.md` the
  `scope_i_*` / `scope_p_*` metrics (idle, di/dt step in A within 300 us, cycle max, load), overlays get I and P columns.

```bash
cd ~/matmul_test/spike_tests_vi
vi config.env                 # set I_SCALE from the probe; check I_CH / V_CH
sudo ./00_preflight.sh        # expects "I ch 1  live" and "V ch 2  OK (~12 V)"; probes the GEMM shape for this GPU
sudo ./run_all.sh             # 01 baseline, 02 duration, 03 clock (one UAC "yes"), 04 wave -> results/<stamp>_run_all/report.md
```

The voltage + voltage campaign (CH1 PSU cable, CH2 PCIe slot) stays in `../spike_tests`: `cd ~/matmul_test/spike_tests && sudo ./run_all.sh`.

## HDO4034A + CP500 (LeCroy) instead of the DSOX1202A

Same scripts; only `config.env` changes: `SCOPE_METER=2`, `SCOPE_BUSID=<usbipd bus id of the HDO>` (or `SCOPE_BUSID=` and
`SCOPE_RESOURCE=TCPIP::<ip>::INSTR` over LAN with the scope's remote control set to VXI11), `I_SCALE=1` (the CP500 is a ProBus
probe: the scope reads amperes, no A/V conversion), `V_DIV=0.5` (12-bit ADC; the driver moves to 1 V/div automatically if the
12 V offset is refused). The LeCroy driver verifies on a real acquisition the accepted V/div, the window (TDIV snaps to
1-2-5 steps: a 93 ms request becomes 100 ms) and the 1 MSa/s sample rate, and logs them as `[scope] note:` lines.

Verified on the HDO4034A (firmware 10.4.0) on 2026-11-17:

- **Remote control must be USBTMC** on the scope (Utilities > Utilities Setup > Remote > Control from). With TCPIP/VXI11
  selected the USB link answers control requests but never a command (`*IDN?` times out). Changing it re-enumerates the USB
  device: run `usb_attach.sh <busid>` again.
- **Capture model: FRTR = stop.** In SINGLE mode this scope never completed an acquisition on its own (no edge level on the
  current or on a voltage channel brought `TRMD?` to STOP), and `FRTR` ends the acquisition at once: the record is always the
  last window BEFORE the force, whatever `TRDL` says. The driver therefore sends `FRTR` at launch + (window - pre), which gives
  exactly [-15 ms, +85 ms] around the launch for the 63 ms GEMM, and parks the edge level out of range (165 A on the current
  channel) so no hardware trigger can end the record early. Log line: `capture = stop: FRTR at launch + 85 ms`. `edge_triggered`
  is meaningless for HDO runs (the onset is at +0.6 .. +0.8 ms from the nominal launch = the FRTR timing jitter).
- **USBTMC quirks** handled in `spike_eval._visa_open` / `LecroyHDO`: the scope advertises USB488 REN_CONTROL but STALLs it
  (pyvisa-py pipe error on open -> made best-effort), and a bulk-in request of >= 64 KB is answered in 65 521-byte transfers
  padded to 65 536 (no short packet, pyvisa-py waits forever) -> requests are limited to 32 KB (`LecroyHDO.CHUNK`).
- Settings the scripts change on the scope (nothing else): C1 12 A/div offset -45 A, C2/C4 0.5 V/div offset -12 V, coupling
  DC 1 MOhm, TDIV 10 ms (100 ms record), `FixedSampleRate` 1 MSa/s, MSIZ 100k, TRDL -35 ms (nominal), edge trigger source C1
  rising with the level parked, TRMD SINGLE / STOP; preflight (`check_hw.py`) additionally takes one AUTO acquisition at
  1 ms/div with C1 at 10 A/div and leaves TRMD AUTO. No setup is restored afterwards: save the scope setup first if needed.
- CP500 vs PMD2X on the same launch: the clamp read 12 A load / 14.9 A peak while the PMD2X (both cables) read 19.3 A peak,
  and P = V x I from the scope 146 W load vs PMD ~190 W. Check which conductors the clamp encloses before comparing.

```bash
cd ~/matmul_test/spike_tests_vi
vi config.env                 # SCOPE_METER=2, SCOPE_BUSID / SCOPE_RESOURCE, I_SCALE=1
sudo ./lecroy_check.sh        # dry run: open, setup, arm, force, fetch one record -> check the note line and the units (I in 'A')
sudo ./run_all.sh             # then the normal sequence
```
