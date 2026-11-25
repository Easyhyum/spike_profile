"""onset.py — kernel start / end on a scope voltage record, robust to small drops (tens of mV) and to a record whose
trigger came late (forced trigger a few ms after the launch, so the event sits in the pre-trigger part)."""
import numpy as np


def smooth(x, n):
    xp = np.concatenate([np.full(n, x[:n].mean()), x, np.full(n, x[-n:].mean())])
    return np.convolve(xp, np.ones(n) / n, mode="same")[n:-n]


def n_for(t, seconds, nmin=1):
    """number of samples covering `seconds` at the record's sample spacing (smoothing windows are defined in time, not samples)"""
    dt = float(np.median(np.diff(t[:1000]))) if len(t) > 1 else 1e-6
    return max(nmin, int(round(seconds / dt)))


def onset_from_drop(t, v, n_sm=None, min_drop=0.010):
    """Returns (t_onset, v_idle, t_end). Idle = median of the smoothed pre-trigger part (robust to an event inside it),
    sigma = 1.4826*MAD of the same; onset = first smoothed sample below idle - max(6 sigma, min_drop), walked back to the
    last sample within 2 sigma of idle; end = last sample below the half level between idle and the settled load level."""
    if n_sm is None: n_sm = n_for(t, 20e-6)                    # 20 us smoothing whatever the sample rate
    sm = smooth(v, n_sm)
    pre = t < -0.5e-3
    base = sm[pre] if pre.sum() >= 200 else sm[: max(200, len(sm) // 10)]
    idle = float(np.median(base)); sig = float(1.4826 * np.median(np.abs(base - idle))) + 1e-6
    # the event must be a real drop: at least 6 sigma / min_drop AND 30 % of the deepest excursion of the record, so a few-mV
    # noise dip milliseconds before the kernel (seen when the trigger came late) cannot be taken for the onset
    thr = idle - max(6 * sig, min_drop, 0.3 * (idle - float(sm.min())))
    idx = np.where(sm < thr)[0]
    if len(idx) == 0:
        return 0.0, idle, None
    i0 = int(idx[0])
    while i0 > 0 and sm[i0 - 1] < idle - 2 * sig:
        i0 -= 1
    s0 = float(t[i0])
    body = sm[t > s0 + 0.3e-3]
    settled = float(np.percentile(body, 15)) if len(body) else idle
    half = idle + 0.5 * (settled - idle)
    below = np.where(sm[i0:] < half)[0]
    end = float(t[i0 + below[-1]]) if len(below) else None
    return s0, idle, end


def rise_time(t, y, s0=0.0, n_sm=None, plateau=(1e-3, 5e-3), lo_frac=0.1, hi_frac=0.9):
    """Rise of a current (or power) step at the kernel start. Idle = median of y before s0 - 0.5 ms, level = median of y over
    [s0 + plateau[0], s0 + plateau[1]] (the plateau right after the ramp, before any DVFS step). Returns a dict with the idle and
    level, t_start (first point above idle + 4 sigma after s0 - 0.5 ms), t10 / t90 (first crossings of idle + 10 % / 90 % of the
    step) and rise_s = t90 - t10, all times in the same frame as t (seconds). None when there is no step (< 0.5 A / W)."""
    if n_sm is None: n_sm = n_for(t, 10e-6)                    # 10 us smoothing
    ys = smooth(y, n_sm); pre = t < s0 - 0.5e-3
    if pre.sum() < 50: return None
    idle = float(np.median(ys[pre])); sig = float(1.4826 * np.median(np.abs(ys[pre] - idle))) + 1e-9
    pl = (t >= s0 + plateau[0]) & (t <= s0 + plateau[1])
    if pl.sum() < 10: return None
    level = float(np.median(ys[pl])); step = level - idle
    if step < 0.5: return None
    after = t >= s0 - 0.5e-3
    def first(thr):
        k = np.flatnonzero(after & (ys > thr)); return float(t[k[0]]) if len(k) else None
    t_start, t10, t90 = first(idle + 4 * sig), first(idle + lo_frac * step), first(idle + hi_frac * step)
    return dict(idle=idle, level=level, step=step, t_start=t_start, t10=t10, t90=t90,
                rise_s=(t90 - t10) if (t10 is not None and t90 is not None) else None)


def max_slope(t, y, s0=0.0, span_s=10e-6, n_sm=None, search_s=2e-3):
    """Largest rise of the smoothed signal over span_s within [s0, s0 + search_s]: (slope per second, time of the window start)."""
    if n_sm is None: n_sm = n_for(t, 20e-6)
    ys = smooth(y, n_sm); sel = (t >= s0) & (t <= s0 + search_s)
    if sel.sum() < 50: return None, None
    tt, yy = t[sel], ys[sel]; dt = float(np.median(np.diff(tt))); k = max(1, int(round(span_s / dt)))
    d = yy[k:] - yy[:-k]
    if not len(d): return None, None
    j = int(np.argmax(d)); return float(d[j] / (k * dt)), float(tt[j])


def load_csv(path, dtype=np.float64):
    """burst_NN.csv / freqmon.csv / pmd-like csv -> numpy record array (field access by header name), with np.loadtxt instead of
    np.genfromtxt (genfromtxt needs ~10x the file size in RAM; a 100 MB record cost > 1 GB and helped crash WSL)."""
    with open(path) as f:
        header = f.readline().strip().lstrip("#").split(",")
    names = [h.strip().split("(")[0].strip() or f"c{k}" for k, h in enumerate(header)]
    a = np.loadtxt(path, delimiter=",", skiprows=1, dtype=dtype, ndmin=2)
    if a.shape[1] != len(names): names = names[: a.shape[1]] + [f"c{k}" for k in range(len(names), a.shape[1])]
    return np.rec.fromarrays([a[:, k] for k in range(a.shape[1])], names=names)
