# Spike test report: 20261125_093245_run_all_pl200_warm20x500_20ns

- GPU: NVIDIA GeForce RTX 5060 Ti, 4.78 W, PCIe gen 5 (max 5) x8 (slot x16)  (36 SMs; 35 usable with freqmon (one SM holds the 10 us clock monitor)); PCIe link during the runs: gen 5 x8 (samples: {'5': 4804} ; {'8': 4804})
- Scope: LECROY,HDO4034A,LCRY3521N56907,10.4.0; PMD2X: /dev/ttyACM0
- Workload (auto, find_max_shape.py): fp32 M=N=K=8192, 1 launch = 63.53 ms, 17.3 TFLOP/s, 177.6 W mean (nvidia-smi); every measurement is a single cycle = 1 launch unless noted
- Board power limit during the runs: 200 W (default 180 W, max 200 W; nvidia-smi -q -d POWER)
- Warm clock: a GEMM of the same shape ended 20 ms before every measured launch (clock at its settled level from t = 0)
- Alignment: t = 0 at the rail-voltage onset (scope, 20 ns per sample; csv decimated to >= 1 us, npz full rate); PMD2X 1 ms; freqmon in-kernel clock 10 us

## 04 wave x size sweep (wavegemm single cycles; cuBLAS max shape as reference)

| kernel | K | waves | CTAs | M | N | kernel ms | TFLOP/s | di/dt drop V | CH(V2) drop V | PMD W | PMD A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wave | 8192 | 4.00 | 140 | 2560 | 1792 | 5.43 | 13.8 | 0.244 | n/a | 191 | 16.3 |
| wave | 8192 | 8.00 | 280 | 3584 | 2560 | 11.08 | 13.6 | 0.241 | n/a | 193 | 16.4 |
| wave | 8192 | 32.00 | 1120 | 8192 | 4480 | 43.72 | 13.8 | 0.245 | n/a | 201 | 17.2 |
| wave | 8192 | 58.51 | 2048 | 8192 | 8192 | 79.39 | 13.8 | 0.257 | n/a | 210 | 17.7 |
| wave | 8192 | 64.00 | 2240 | 10240 | 7168 | 88.53 | 13.6 | 0.242 | n/a | 203 | 17.3 |

## Figures (folder level)

- `overlay_04_waves_K8192_CH1_CH2.png`
- `overlay_04_waves_K8192_combined.png`
- `overlay_04_waves_K8192_freqmon.png`
- `overlay_04_waves_allK_CH1_CH2.png`
- `overlay_04_waves_allK_freqmon.png`
- `wave_sweep_summary.png`

Per-run folders hold `burst_NN.png` (4 x windows, both channels), `burst_NN_freq.png` (+ in-kernel clock), `burst_NN_V_*.png`, csv/npz raw data, `marks.json`, `summary.json`.
