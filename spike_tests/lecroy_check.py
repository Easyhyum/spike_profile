#!/usr/bin/env python
"""lecroy_check.py — one-shot dry run of the HDO4034A driver with the campaign's config.env values, no GPU load:
open -> setup (offset / timebase / trigger-position / sample-rate verification) -> arm -> force -> fetch -> report.
Run it once after attaching the HDO (USB: usbipd attach; LAN: SCOPE_RESOURCE=TCPIP::<ip>::INSTR, remote control = VXI11):

    cd ~/matmul_test/spike_tests_vi && sudo ./lecroy_check.sh        # wrapper that sources config.env
    sudo venv/bin/python spike_tests/lecroy_check.py --v-ch 2 --i-ch 1 --i-scale 1 [--resource TCPIP::...::INSTR]

Expected: note lines with the accepted V/div, 'first sample at -15.0 ms', 'SR 1000000', the probe name on the current
channel, 'capture = stop: FRTR at launch + 85 ms', then a record of ~1 us/sample whose current channel unit is 'A' (ProBus
CP500) and whose idle values make sense. The HDO records the window BEFORE the force (FRTR = stop), so the dry run's record
is idle by construction; spike_eval sends the force (window - pre) after the launch.
"""
import argparse, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resource", default=None); ap.add_argument("--v-ch", type=int, default=2); ap.add_argument("--v2-ch", type=int, default=0)
    ap.add_argument("--i-ch", type=int, default=0); ap.add_argument("--v-probe", type=float, default=1); ap.add_argument("--v-div", type=float, default=0.5)
    ap.add_argument("--v-center", type=float, default=12.0); ap.add_argument("--i-scale", type=float, default=1.0); ap.add_argument("--i-div", type=float, default=12)
    ap.add_argument("--i-center", type=float, default=45); ap.add_argument("--trig-src", default="v"); ap.add_argument("--trig-level", default="auto:0.15")
    ap.add_argument("--trig-slope", default="neg"); ap.add_argument("--window-ms", type=float, default=93); ap.add_argument("--pre-ms", type=float, default=15)
    a = ap.parse_args()
    import numpy as np, spike_eval as se
    no_current = not a.i_ch
    cfg = dict(resource=a.resource, no_setup=False, v_probe=a.v_probe, v_div=a.v_div, v_center=a.v_center, i_scale=a.i_scale, i_offset=0.0,
               i_div=a.i_div, i_center=a.i_center, v_ch=a.v_ch, i_ch=a.i_ch, v2_ch=a.v2_ch or None, no_current=no_current,
               trig_src=(a.i_ch if (a.trig_src == "i" and not no_current) else a.v_ch), trig_slope=a.trig_slope,
               trig_level_volts=(float(a.trig_level) / a.i_scale if (a.trig_src == "i" and not no_current) else
                                 (a.trig_level if str(a.trig_level).startswith("auto") else float(a.trig_level))),
               window_s=a.window_ms * 1e-3, pre_s=a.pre_ms * 1e-3, wait_s=10)
    t0 = time.time(); sc = se.LecroyHDO(cfg); print("open :", sc.resource, "|", sc.idn, f"({time.time()-t0:.1f} s)")
    t0 = time.time(); sc.setup(); print(f"setup ({time.time()-t0:.1f} s):", sc.note)
    print("trigger level used:", getattr(sc, "trig_level_used", None))
    sc.arm(); print("armed; TRMD? =", sc.q("TRMD?")); time.sleep(0.5)
    sc.force(); t0 = time.time()
    while not sc.done() and time.time() - t0 < 10: time.sleep(0.05)
    print(f"forced -> done in {time.time()-t0:.2f} s; TRMD? = {sc.q('TRMD?')}")
    t, v, i, meta = sc.fetch()
    print(f"record: {len(v)} pts, dt = {meta['dt']*1e6:.3f} us, t = {t[0]*1e3:+.2f} .. {t[-1]*1e3:+.2f} ms (0 = trigger)")
    print(f"CH{a.v_ch} V: mean {v.mean():.4f} std {v.std()*1e3:.2f} mV  ({sc._desc(a.v_ch)['unit']!r})")
    if a.v2_ch: v2 = meta["v2"]; print(f"CH{a.v2_ch} V2: mean {v2.mean():.4f} std {v2.std()*1e3:.2f} mV")
    if not no_current: print(f"CH{a.i_ch} I: mean {i.mean():.3f} std {i.std():.3f} (scope unit {meta.get('i_unit')!r}; x I_SCALE {a.i_scale} -> A)")
    sc.stop(); sc.close(); print("OK")


if __name__ == "__main__":
    main()
