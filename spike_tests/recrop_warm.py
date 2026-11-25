#!/usr/bin/env python
"""recrop_warm.py <run_dir> [...] — re-apply crop_to_last_load (warm-clock runs) to the saved burst_NN.npz records with the current
spike_eval code, rewrite burst_NN.npz / .csv (csv decimated to >= 1 us like the engine) and re-run replot_burst + plot_spike_csv.
Use after a crop fix for runs recorded before it (the saved record still holds the warm-up, so the crop can be redone)."""
import glob, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.argv, argv = ["x"], sys.argv
import spike_eval as se
for d in argv[1:]:
    M = json.load(open(os.path.join(d, "marks.json"))); a = M["args"]; gap = a.get("prewarm_gap_ms", 0) or 0
    if gap <= 0: print("not a warm-clock run:", d); continue
    on_s = M["on_ms"] * 1e-3; n = 0
    for f in sorted(glob.glob(os.path.join(d, "burst_[0-9][0-9].npz"))):
        z = np.load(f); t, v, i, p = z["t"], z["v"], z["i"], z["p"]; v2 = z["v2"] if "v2" in z.files else None; meta = json.loads(str(z["meta"]))
        keep = se.crop_to_last_load(t, v, on_s, a["pre_ms"] * 1e-3, gap * 1e-3)
        if keep is None or keep.all(): continue
        t, v, i, p = t[keep], v[keep], i[keep], p[keep]; v2 = v2[keep] if v2 is not None else None
        s0 = float(se.scope_onset(t, v)); meta["onset_s"] = s0; meta["recropped"] = True
        np.savez(f, t=t, v=v, i=i, p=p, **({"v2": v2} if v2 is not None else {}), meta=json.dumps(meta))
        dt = float(np.median(np.diff(t[:1000]))); step = max(1, int(round(1e-6 / dt)))
        cols = [t[::step], v[::step], i[::step], p[::step]] + ([v2[::step]] if v2 is not None else []); hdr = "t_s,v_V,i_A,p_W" + (",v2_V" if v2 is not None else "")
        np.savetxt(f.replace(".npz", ".csv"), np.c_[tuple(cols)], delimiter=",", header=hdr, comments=""); n += 1
    print(f"{os.path.basename(d)}: re-cropped {n} records")
    if n:
        subprocess.run([sys.executable, os.path.join(HERE, "replot_burst.py"), d], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([sys.executable, os.path.join(ROOT, "plot_spike_csv.py"), d], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
