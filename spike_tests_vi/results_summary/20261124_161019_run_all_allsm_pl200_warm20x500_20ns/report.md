# Spike test report: 20261124_161019_run_all_allsm_pl200_warm20x500_20ns

- GPU: NVIDIA GeForce RTX 5060 Ti, 4.72 W, PCIe gen 5 (max 5) x8 (slot x16)  (36 SMs; ALL SMs run the GEMM (FREQMON=0: no in-kernel clock monitor; clock = nvidia-smi 100 ms)); PCIe link during the runs: gen 5 x8 (samples: {'5': 2538} ; {'8': 2538})
- Scope: LECROY,HDO4034A,LCRY3521N56907,10.4.0; PMD2X: /dev/ttyACM0
- Workload (auto, find_max_shape.py): fp32 M=N=K=8192, 1 launch = 63.53 ms, 17.3 TFLOP/s, 177.6 W mean (nvidia-smi); every measurement is a single cycle = 1 launch unless noted
- Board power limit during the runs: 200 W (default 180 W, max 200 W; nvidia-smi -q -d POWER)
- Warm clock: a GEMM of the same shape ended 20 ms before every measured launch (clock at its settled level from t = 0)
- Alignment: t = 0 at the rail-voltage onset (scope, 20 ns per sample; csv decimated to >= 1 us, npz full rate); PMD2X 1 ms; freqmon in-kernel clock 10 us

## 04 wave x size sweep (wavegemm single cycles; cuBLAS max shape as reference)

| kernel | K | waves | CTAs | M | N | kernel ms | TFLOP/s | di/dt drop V | CH(V2) drop V | PMD W | PMD A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wave | 2048 | 0.50 | 18 | 768 | 768 | 0.31 | 7.8 | 0.144 | n/a | 31 | 2.8 |
| wave | 2048 | 0.61 | 22 | 512 | 1408 | 0.33 | 8.9 | 0.170 | n/a | 43 | 3.5 |
| wave | 2048 | 0.69 | 25 | 1280 | 640 | 0.31 | 10.7 | 0.212 | n/a | 124 | 10.5 |
| wave | 2048 | 0.81 | 29 | 256 | 3712 | 0.36 | 10.9 | 0.219 | n/a | 32 | 2.8 |
| wave | 2048 | 0.89 | 32 | 1024 | 1024 | 0.31 | 13.7 | 0.247 | n/a | 138 | 11.6 |
| wave | 2048 | 1.00 | 36 | 1536 | 768 | 0.31 | 15.4 | 0.284 | n/a | 33 | 2.9 |
| wave | 2048 | 1.50 | 54 | 1536 | 1152 | 0.61 | 11.9 | 0.289 | n/a | 149 | 12.6 |
| wave | 2048 | 1.61 | 58 | 512 | 3712 | 0.69 | 11.2 | 0.271 | n/a | 82 | 6.9 |
| wave | 2048 | 1.69 | 61 | 256 | 7808 | 0.71 | 11.5 | 0.268 | n/a | 157 | 13.3 |
| wave | 2048 | 1.81 | 65 | 1280 | 1664 | 0.63 | 13.8 | 0.275 | n/a | 84 | 7.0 |
| wave | 2048 | 1.89 | 68 | 1024 | 2176 | 0.65 | 14.0 | 0.284 | n/a | 110 | 9.4 |
| wave | 2048 | 2.00 | 72 | 2048 | 1152 | 0.66 | 14.6 | 0.277 | n/a | 183 | 15.3 |
| wave | 2048 | 3.00 | 108 | 2304 | 1536 | 0.99 | 14.6 | 0.277 | n/a | 184 | 15.6 |
| wave | 2048 | 4.00 | 144 | 3072 | 1536 | 1.30 | 14.9 | 0.273 | n/a | 198 | 16.9 |
| wave | 2048 | 8.00 | 288 | 4096 | 2304 | 2.60 | 14.9 | 0.274 | n/a | 197 | 16.7 |
| wave | 2048 | 16.00 | 576 | 6144 | 3072 | 5.28 | 14.6 | 0.291 | n/a | 209 | 17.9 |
| wave | 2048 | 32.00 | 1152 | 8192 | 4608 | 11.38 | 13.6 | 0.262 | n/a | 205 | 17.6 |
| wave | 2048 | 56.89 | 2048 | 8192 | 8192 | 20.18 | 13.6 | 0.272 | n/a | 211 | 17.9 |
| wave | 8192 | 0.50 | 18 | 768 | 768 | 1.31 | 7.4 | 0.138 | n/a | 110 | 9.2 |
| wave | 8192 | 0.61 | 22 | 512 | 1408 | 1.35 | 8.7 | 0.176 | n/a | 129 | 10.9 |
| wave | 8192 | 0.69 | 25 | 1280 | 640 | 1.34 | 10.0 | 0.189 | n/a | 141 | 11.8 |
| wave | 8192 | 0.81 | 29 | 256 | 3712 | 1.38 | 11.3 | 0.219 | n/a | 161 | 13.5 |
| wave | 8192 | 0.89 | 32 | 1024 | 1024 | 1.34 | 12.8 | 0.238 | n/a | 175 | 14.8 |
| wave | 8192 | 1.00 | 36 | 1536 | 768 | 1.34 | 14.4 | 0.273 | n/a | 203 | 17.0 |
| wave | 8192 | 1.50 | 54 | 1536 | 1152 | 2.66 | 10.9 | 0.279 | n/a | 199 | 16.8 |
| wave | 8192 | 1.61 | 58 | 512 | 3712 | 2.76 | 11.3 | 0.276 | n/a | 195 | 16.6 |
| wave | 8192 | 1.69 | 61 | 256 | 7808 | 2.74 | 12.0 | 0.280 | n/a | 199 | 16.7 |
| wave | 8192 | 1.81 | 65 | 1280 | 1664 | 2.72 | 12.8 | 0.263 | n/a | 194 | 16.6 |
| wave | 8192 | 1.89 | 68 | 1024 | 2176 | 2.77 | 13.2 | 0.273 | n/a | 198 | 16.8 |
| wave | 8192 | 2.00 | 72 | 2048 | 1152 | 2.70 | 14.3 | 0.273 | n/a | 201 | 17.0 |
| wave | 8192 | 3.00 | 108 | 2304 | 1536 | 4.04 | 14.4 | 0.271 | n/a | 206 | 17.6 |
| wave | 8192 | 4.00 | 144 | 3072 | 1536 | 5.42 | 14.3 | 0.269 | n/a | 200 | 17.2 |
| wave | 8192 | 8.00 | 288 | 4096 | 2304 | 10.85 | 14.2 | 0.259 | n/a | 206 | 17.6 |
| wave | 8192 | 16.00 | 576 | 6144 | 3072 | 22.12 | 14.0 | 0.268 | n/a | 205 | 17.6 |
| wave | 8192 | 32.00 | 1152 | 8192 | 4608 | 43.62 | 14.2 | 0.271 | n/a | 213 | 18.2 |
| wave | 8192 | 56.89 | 2048 | 8192 | 8192 | 76.47 | 14.4 | 0.282 | n/a | 216 | 18.4 |
| cublas | 8192 | 56.89 | 2048 | 8192 | 8192 | 59.62 | 18.4 | 0.283 | n/a | 253 | 21.3 |

## Figures (folder level)

- `overlay_04_waves_K2048_CH1_CH2.png`
- `overlay_04_waves_K8192_CH1_CH2.png`
- `overlay_04_waves_allK_CH1_CH2.png`
- `overlay_clk.png`
- `wave_sweep_summary.png`

Per-run folders hold `burst_NN.png` (4 x windows, both channels), `burst_NN_freq.png` (+ in-kernel clock), `burst_NN_V_*.png`, csv/npz raw data, `marks.json`, `summary.json`.
