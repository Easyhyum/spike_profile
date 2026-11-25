#!/usr/bin/env python
"""wave_targets.py <wave> [<wave> ...] -> one line per target: label ctas M N
A wave is (M/256)*(N/128) CTAs over the usable SMs (device SM count - 1 held by freqmon) with 1 CTA per SM (wavegemm.cu,
256x128 tile like the cuBLAS kernel). ctas = round(wave * usable); M = 256*a, N = 128*b with a*b = ctas and a <= b
(the most square factorization; primes -> M=256, N=128*ctas)."""
import math, subprocess, sys
try:
    import torch; sms = torch.cuda.get_device_properties(0).multi_processor_count
except Exception:
    sms = 36
import os
usable = int(os.environ.get("AUTO_USABLE_SMS") or (sms - 1))
for w in sys.argv[1:]:
    wf = float(w); ctas = max(1, int(round(wf * usable)))
    a = max(d for d in range(1, int(math.isqrt(ctas)) + 1) if ctas % d == 0); b = ctas // a
    print(f"w{w} {ctas} {256*a} {128*b}")
