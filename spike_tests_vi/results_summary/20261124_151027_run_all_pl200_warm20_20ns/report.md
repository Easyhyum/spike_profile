# Spike test report: 20261124_151027_run_all_pl200_warm20_20ns

- GPU: NVIDIA GeForce RTX 5060 Ti, 4.72 W, PCIe gen 5 (max 5) x8 (slot x16)  (36 SMs; 35 usable with freqmon (one SM holds the 10 us clock monitor)); PCIe link during the runs: gen 5 x8 (samples: {'5': 2975} ; {'8': 2975})
- Scope: LECROY,HDO4034A,LCRY3521N56907,10.4.0; PMD2X: /dev/ttyACM0
- Workload (auto, find_max_shape.py): fp32 M=N=K=8192, 1 launch = 63.53 ms, 17.3 TFLOP/s, 177.6 W mean (nvidia-smi); every measurement is a single cycle = 1 launch unless noted
- Board power limit during the runs: 200 W (default 180 W, max 200 W; nvidia-smi -q -d POWER)
- Warm clock: a GEMM of the same shape ended 20 ms before every measured launch (clock at its settled level from t = 0)
- Alignment: t = 0 at the rail-voltage onset (scope, 20 ns per sample; csv decimated to >= 1 us, npz full rate); PMD2X 1 ms; freqmon in-kernel clock 10 us

## 04 wave x size sweep (wavegemm single cycles; cuBLAS max shape as reference)

| kernel | K | waves | CTAs | M | N | kernel ms | TFLOP/s | di/dt drop V | CH(V2) drop V | PMD W | PMD A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wave | 2048 | 0.51 | 18 | 768 | 768 | 0.31 | 7.8 | 0.137 | n/a | 38 | 3.5 |
| wave | 2048 | 0.60 | 21 | 768 | 896 | 0.31 | 9.1 | 0.157 | n/a | 35 | 2.9 |
| wave | 2048 | 0.69 | 24 | 1024 | 768 | 0.31 | 10.3 | 0.176 | n/a | 113 | 9.5 |
| wave | 2048 | 0.80 | 28 | 1024 | 896 | 0.31 | 12.2 | 0.203 | n/a | 64 | 5.3 |
| wave | 2048 | 0.91 | 32 | 1024 | 1024 | 0.31 | 13.7 | 0.229 | n/a | 36 | 3.2 |
| wave | 2048 | 1.00 | 35 | 1280 | 896 | 0.32 | 14.6 | 0.247 | n/a | 130 | 11.0 |
| wave | 2048 | 1.49 | 52 | 1024 | 1664 | 0.62 | 11.3 | 0.244 | n/a | 65 | 5.4 |
| wave | 2048 | 1.60 | 56 | 1792 | 1024 | 0.62 | 12.2 | 0.247 | n/a | 109 | 9.1 |
| wave | 2048 | 1.71 | 60 | 1536 | 1280 | 0.62 | 13.0 | 0.243 | n/a | 83 | 6.9 |
| wave | 2048 | 1.80 | 63 | 1792 | 1152 | 0.62 | 13.6 | 0.248 | n/a | 35 | 3.2 |
| wave | 2048 | 1.89 | 66 | 1536 | 1408 | 0.67 | 13.3 | 0.247 | n/a | 172 | 14.4 |
| wave | 2048 | 2.00 | 70 | 1792 | 1280 | 0.66 | 14.2 | 0.241 | n/a | 164 | 13.9 |
| wave | 2048 | 3.00 | 105 | 1792 | 1920 | 0.99 | 14.2 | 0.240 | n/a | 176 | 14.6 |
| wave | 2048 | 4.00 | 140 | 2560 | 1792 | 1.37 | 13.7 | 0.245 | n/a | 174 | 14.5 |
| wave | 2048 | 8.00 | 280 | 3584 | 2560 | 2.61 | 14.4 | 0.241 | n/a | 183 | 15.4 |
| wave | 2048 | 16.00 | 560 | 5120 | 3584 | 5.65 | 13.3 | 0.231 | n/a | 189 | 16.0 |
| wave | 2048 | 32.00 | 1120 | 8192 | 4480 | 11.21 | 13.4 | 0.238 | n/a | 193 | 16.4 |
| wave | 2048 | 58.51 | 2048 | 8192 | 8192 | 20.48 | 13.4 | 0.242 | n/a | 201 | 17.1 |
| wave | 8192 | 0.51 | 18 | 768 | 768 | 1.31 | 7.4 | 0.131 | n/a | 109 | 9.2 |
| wave | 8192 | 0.60 | 21 | 768 | 896 | 1.32 | 8.5 | 0.150 | n/a | 120 | 10.1 |
| wave | 8192 | 0.69 | 24 | 1024 | 768 | 1.32 | 9.8 | 0.177 | n/a | 135 | 11.5 |
| wave | 8192 | 0.80 | 28 | 1024 | 896 | 1.35 | 11.1 | 0.200 | n/a | 155 | 13.0 |
| wave | 8192 | 0.91 | 32 | 1024 | 1024 | 1.35 | 12.7 | 0.218 | n/a | 166 | 14.0 |
| wave | 8192 | 1.00 | 35 | 1280 | 896 | 1.35 | 14.0 | 0.233 | n/a | 178 | 14.9 |
| wave | 8192 | 1.49 | 52 | 1024 | 1664 | 2.71 | 10.3 | 0.230 | n/a | 177 | 14.8 |
| wave | 8192 | 1.60 | 56 | 1792 | 1024 | 2.65 | 11.4 | 0.239 | n/a | 183 | 15.6 |
| wave | 8192 | 1.71 | 60 | 1536 | 1280 | 2.66 | 12.1 | 0.237 | n/a | 179 | 15.4 |
| wave | 8192 | 1.80 | 63 | 1792 | 1152 | 2.65 | 12.7 | 0.234 | n/a | 178 | 14.9 |
| wave | 8192 | 1.89 | 66 | 1536 | 1408 | 2.69 | 13.2 | 0.235 | n/a | 179 | 15.0 |
| wave | 8192 | 2.00 | 70 | 1792 | 1280 | 2.68 | 14.0 | 0.234 | n/a | 185 | 15.7 |
| wave | 8192 | 3.00 | 105 | 1792 | 1920 | 4.08 | 13.8 | 0.241 | n/a | 186 | 15.6 |
| wave | 8192 | 4.00 | 140 | 2560 | 1792 | 5.38 | 14.0 | 0.230 | n/a | 186 | 15.6 |
| wave | 8192 | 8.00 | 280 | 3584 | 2560 | 10.99 | 13.7 | 0.234 | n/a | 192 | 16.2 |
| wave | 8192 | 16.00 | 560 | 5120 | 3584 | 21.95 | 13.7 | 0.242 | n/a | 197 | 16.6 |
| wave | 8192 | 32.00 | 1120 | 8192 | 4480 | 43.66 | 13.8 | 0.258 | n/a | 198 | 17.1 |
| wave | 8192 | 58.51 | 2048 | 8192 | 8192 | 79.29 | 13.9 | 0.258 | n/a | 205 | 17.5 |
| cublas | 8192 | 58.51 | 2048 | 8192 | 8192 | 61.00 | 18.0 | 0.281 | n/a | 243 | 20.9 |

## Figures (folder level)

- `overlay_04_waves_K2048_CH1_CH2.png`
- `overlay_04_waves_K2048_combined.png`
- `overlay_04_waves_K2048_freqmon.png`
- `overlay_04_waves_K8192_CH1_CH2.png`
- `overlay_04_waves_K8192_combined.png`
- `overlay_04_waves_K8192_freqmon.png`
- `overlay_04_waves_allK_CH1_CH2.png`
- `overlay_04_waves_allK_freqmon.png`
- `overlay_clk.png`
- `wave_sweep_summary.png`

Per-run folders hold `burst_NN.png` (4 x windows, both channels), `burst_NN_freq.png` (+ in-kernel clock), `burst_NN_V_*.png`, csv/npz raw data, `marks.json`, `summary.json`.
