#!/usr/bin/env python
"""
spike_eval.py — GPU power-spike evaluation: a train of count-bound fp32 SGEMM bursts (the strongest-spike
workload on the RTX 5060 Ti: 8192x8192 @ 8192x8192, PMD2X peak ~230-245 W) while measuring with

    -m 0   PMD2X (1 ms raw V/A per rail) + nvidia-smi (100 ms)                      [always on]
    -m 1   + Keysight DSOX1202A  (USBTMC)      CH1 = 12 V rail voltage, CH2 = current probe
    -m 2   + Teledyne LeCroy HDO4034A (USBTMC or TCPIP)  same channel assignment

All meters observe the SAME kernel launches. Measurement order per burst: (1) the PMD2X poller
(multiprocessing.Process, ~1 kHz raw V/A per rail) is running and confirmed, (2) the oscilloscope process
(multiprocessing.Process) has armed a single trigger and confirmed it, (3) a %globaltimer stamp kernel and the
K matmul launches go out. nvidia-smi logs at --smi-ms in a subprocess and an in-kernel SM-clock monitor (freqmon,
10 us, own stream, occupies one SM) runs across the whole burst train. The scope record is pulled back at
1 us/sample; voltage (and current / P = V x I when a current probe is present) is stored raw per burst with
peak / overshoot / undershoot statistics; plot_spike_csv.py draws the per-burst figures incl. the overlay of
scope V, PMD V, nvidia-smi clock and freqmon clock on one time axis (0 = kernel start).

Examples:
    sudo ./venv/bin/python spike_eval.py -m 0 -b 10 -o 60 -d 500
    sudo ./venv/bin/python spike_eval.py -m 1 -b 10 -o 60 -d 1000 --i-scale 10 --trig-level 3
    sudo ./venv/bin/python spike_eval.py -m 2 -b 10 -o 60 -d 1000 --resource TCPIP::192.168.0.20::INSTR
(sudo: pmd2_fast.py needs the serial port and pyusb needs /dev/bus/usb; without -m the help is printed)

Outputs, one folder per run (spike_eval_<YYYYMMDD_HHMMSS>/):
    marks.json, summary.json, pmd.csv (1 ms), gpu_*.csv (nvidia-smi)
    freqmon.csv                      in-kernel SM clock (t_gpu_ns, MHz) at 10 us
    burst_NN.npz / burst_NN.csv      raw scope samples (t, v, i, p) per burst             (-m 1/2)
    burst_NN.png                     4 x-windows (1 us / 10 us / 100 us / 1 ms units) x (P, I, V); V shows both channels (-m 1/2)
    burst_NN_freq.png                same windows, V (both channels) + in-kernel freqmon SM clock on a right axis  (-m 1/2)
    burst_NN_pmd.csv / _pmd.png      PMD2X 1 ms raw V/I/P around the burst (all modes)
    burst_NN_V_scope.png / _V_scope_0-300us.png / _V_pmd.png / _V_overlay.png   (plot_spike_csv.py)
    summary.png                      all bursts overlaid (-m 1/2)
"""
import numpy as np
import shutil, argparse, csv, json, multiprocessing as mp, os, struct, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_S = 1e-6          # default sample spacing (1 us); --scope-sample-ns overrides it (passed to the scope process in cfg["sample_s"])
HDO_MAX_PTS = 12.5e6     # HDO4034A memory per channel


def usbtmc_device_clear(vid_pid_list=((0x2A8D, 0x0387), (0x2A8D, None), (0x0957, None), (0x05FF, None))):
    """USBTMC INITIATE_CLEAR + bulk clear_halt via pyusb. Recovers a scope whose parser is stuck
    (e.g. after an aborted :DIGitize); pyvisa-py's clear() is not supported for USB sessions."""
    try:
        import usb.core, usb.util
    except Exception:
        return False
    for vid, pid in vid_pid_list:
        d = usb.core.find(idVendor=vid, idProduct=pid) if pid else usb.core.find(idVendor=vid)
        if d is None:
            continue
        try:
            intf = d.get_active_configuration()[(0, 0)]
            usb.util.claim_interface(d, intf.bInterfaceNumber)
            d.ctrl_transfer(0xA1, 5, 0, intf.bInterfaceNumber, 1, timeout=2000)          # INITIATE_CLEAR
            for _ in range(50):
                if d.ctrl_transfer(0xA1, 6, 0, intf.bInterfaceNumber, 2, timeout=2000)[0] != 2:
                    break
                time.sleep(0.1)
            for e in intf:
                if usb.util.endpoint_type(e.bmAttributes) == usb.util.ENDPOINT_TYPE_BULK:
                    try: d.clear_halt(e.bEndpointAddress)
                    except Exception: pass
            usb.util.release_interface(d, intf.bInterfaceNumber); usb.util.dispose_resources(d)
            return True
        except Exception:
            try: usb.util.dispose_resources(d)
            except Exception: pass
    return False


# ============================================================ scope drivers (run inside the worker process)
def _visa_open(resource, vids, timeout_ms):
    import pyvisa
    rm = pyvisa.ResourceManager("@py")
    if resource is None:
        res = rm.list_resources("USB?*::INSTR")
        cand = [r for r in res if any(f"::{v}::" in r.upper() for v in vids)]
        if not cand:
            raise RuntimeError(f"no scope with VID {vids} found; USB resources: {list(res)} "
                               "(usbipd attach --wsl --busid <id>; run with sudo; or pass --resource)")
        resource = cand[0]
    if "USB" in resource.upper():
        usbtmc_device_clear()                       # unstick a parser left blocked by an earlier session
        try:                                        # LeCroy HDO advertises USB488 REN_CONTROL but STALLs the request
            from pyvisa_py.protocols import usbtmc as _tmc  # (pipe error in pyvisa-py's open): make that step best-effort
            if not getattr(_tmc.USBTMC, "_ren_patched", False):
                _orig = _tmc.USBTMC._enable_remote_control
                def _soft(self):
                    try: _orig(self)
                    except Exception: pass
                _tmc.USBTMC._enable_remote_control = _soft; _tmc.USBTMC._ren_patched = True
        except Exception:
            pass
    inst = rm.open_resource(resource)
    inst.timeout = timeout_ms
    inst.chunk_size = 1 << 20
    inst.query("*IDN?"); inst.write("*CLS")
    return inst, resource


class KeysightDSOX:
    """DSOX1202A (InfiniiVision 1000 X). Waveform read in RAW mode decimated to the requested points."""
    VIDS = ("0X2A8D", "10893", "0X0957", "2391")      # hex or decimal form of the resource string

    def __init__(self, cfg):
        self.cfg = cfg
        self.inst, self.resource = _visa_open(cfg["resource"], self.VIDS, 30000)
        self.idn = self.inst.query("*IDN?").strip()
        self.npts = int(round(cfg["window_s"] / cfg.get("sample_s", SAMPLE_S)))

    def setup(self):
        c, w = self.cfg, self.inst.write
        self.inst.write("*CLS")
        vc, ic = c["v_ch"], c["i_ch"]
        w(f":CHANnel{vc}:DISPlay ON"); w(f":CHANnel{vc}:COUPling DC"); w(f":CHANnel{vc}:PROBe {c['v_probe']}")
        self._scale_with_offset(vc, c['v_div'], c['v_center'])
        if c.get("v2_ch"):
            v2 = c["v2_ch"]
            w(f":CHANnel{v2}:DISPlay ON"); w(f":CHANnel{v2}:COUPling DC"); w(f":CHANnel{v2}:PROBe {c['v_probe']}")
            self._scale_with_offset(v2, c['v_div'], c['v_center'])
        if not c.get("no_current"):
            w(f":CHANnel{ic}:DISPlay ON"); w(f":CHANnel{ic}:COUPling DC"); w(f":CHANnel{ic}:PROBe 1")
            self._scale_with_offset(ic, c['i_div'] / c['i_scale'], c['i_center'] / c['i_scale'])
        w(f":TRIGger:HFReject ON")
        w(":TIMebase:MODE MAIN"); w(f":TIMebase:SCALe {c['window_s'] / 10}")
        w(f":TIMebase:POSition {c['window_s'] / 2 - c['pre_s']}")        # trigger at pre_s from the left edge
        w(":ACQuire:TYPE HRESolution")                                     # average the decimated samples
        lvl = c['trig_level_volts']
        if isinstance(lvl, str) and lvl.startswith("auto"):
            delta = float(lvl.split(":")[1]) if ":" in lvl else 0.15
            w(":TRIGger:SWEep AUTO"); w(":SINGle"); self.inst.query("*OPC?"); time.sleep(0.3)
            for _ in range(100):
                if self.done(): break
                time.sleep(0.02)
            _, vv, _ = self._read(c["v_ch"]); import numpy as np
            lvl = float(np.median(vv)) - delta
            self.note = getattr(self, "note", "") + f" trigger auto: idle {float(np.median(vv)):.3f} V -> level {lvl:.3f} V;"
        w(":TRIGger:MODE EDGE"); w(f":TRIGger:EDGE:SOURce CHANnel{c['trig_src']}")
        w(f":TRIGger:EDGE:SLOPe {'POSitive' if c['trig_slope'] == 'pos' else 'NEGative'}")
        w(f":TRIGger:EDGE:LEVel {lvl},CHANnel{c['trig_src']}"); w(":TRIGger:SWEep NORMal"); w(":TRIGger:NREJect OFF")
        self.trig_level_used = lvl
        errs = []
        for _ in range(10):
            e = self.inst.query(":SYSTem:ERRor?").strip()
            if e.startswith("+0"): break
            errs.append(e)
        if errs: self.note = getattr(self, "note", "") + " setup errors: " + " | ".join(errs) + ";"
        self.inst.query("*OPC?")

    def _scale_with_offset(self, ch, div, center):
        """1000 X offset range depends on V/div (1:1 probe: 12 V needs >= 1 V/div); step up until accepted."""
        for sc in sorted({div, div * 2, div * 5, div * 10, div * 25}):
            self.inst.write(f":CHANnel{ch}:SCALe {sc}"); self.inst.write(f":CHANnel{ch}:OFFSet {center}")
            if self.inst.query(":SYSTem:ERRor?").startswith("+0"):
                self.note = getattr(self, "note", "") + (f" CH{ch} {div}->{sc} V/div (offset range);" if sc != div else "")
                return sc
        return div

    def arm(self):
        self.inst.query(":AER?")                       # clear a stale arm event
        self.inst.write(":SINGle"); self.inst.query("*OPC?")
        t0 = time.monotonic()                          # wait until the trigger is really armed (Arm Event Register)
        while time.monotonic() - t0 < 5.0:
            try:
                if int(self.inst.query(":AER?")) == 1:
                    return
            except Exception:
                pass
            time.sleep(0.02)

    def done(self):
        try:
            return (int(self.inst.query(":OPERegister:CONDition?")) & 8) == 0   # RUN bit cleared
        except Exception:
            return False

    def stop(self):
        self.inst.write(":STOP")

    def force(self):
        self.inst.write(":TRIGger:FORCe")

    def _read(self, ch):
        import numpy as np
        w = self.inst.write
        w(f":WAVeform:SOURce CHANnel{ch}"); w(":WAVeform:FORMat WORD"); w(":WAVeform:BYTeorder LSBFirst")
        w(":WAVeform:UNSigned 0"); w(":WAVeform:POINts:MODE RAW")
        w(":WAVeform:POINts MAX"); self.avail = int(float(self.inst.query(":WAVeform:POINts?")))   # per read: AUTO-sweep records are decimated
        w(f":WAVeform:POINts {min(self.npts, self.avail)}")
        pre = self.inst.query(":WAVeform:PREamble?").strip().split(",")
        xinc, xorig, xref, yinc, yorig, yref = (float(x) for x in pre[4:10])
        raw = self.inst.query_binary_values(":WAVeform:DATA?", datatype="h", is_big_endian=False,
                                            container=np.array, chunk_size=1 << 20)
        y = (raw.astype(np.float64) - yref) * yinc + yorig
        t = (np.arange(len(raw)) - xref) * xinc + xorig
        return t, y, xinc

    def fetch(self):
        import numpy as np
        t1, v, dt1 = self._read(self.cfg["v_ch"])
        meta = dict(dt=dt1, idn=self.idn)
        if self.cfg.get("v2_ch"):
            _, v2, _ = self._read(self.cfg["v2_ch"]); n = min(len(v), len(v2)); meta["v2"] = v2[:n]; t1, v = t1[:n], v[:n]
        if self.cfg.get("no_current"):
            return t1, v, np.full_like(v, np.nan), meta
        t2, i, dt2 = self._read(self.cfg["i_ch"])
        n = min(len(v), len(i))
        if "v2" in meta: meta["v2"] = meta["v2"][:n]
        return t1[:n], v[:n], i[:n], meta

    def close(self):
        self.inst.close()


class LecroyHDO:
    """HDO4034A (MAUI remote control: USBTMC VID 05FF, or SCOPE_RESOURCE=TCPIP::<ip>::INSTR with the scope's remote
    control set to VXI11/LXI). 12-bit ADC, so 1 V/div still resolves 0.25 mV. The sample rate is pinned to 1 MSa/s
    (FixedSampleRate), TDIV snaps to 1-2-5 steps, so the record is 10 x TDIV x 1e6 points; the real window, the trigger
    position (MAUI's TRDL sign is verified on a real acquisition) and the sample rate are read back and logged.
    A ProBus current probe (CP500) makes the channel read amperes directly -> I_SCALE = 1 for that channel.
    CAPTURE MODEL (measured 2026-11-17): in SINGLE mode this scope never completes an acquisition on its own (no edge
    level, on a current or a voltage channel, ever brought TRMD? to STOP), and FRTR ends the acquisition at once: the record
    is always the last `window` seconds BEFORE the force, whatever TRDL says (TRDL only changes the nominal HORIZ_OFFSET).
    So the driver captures by STOPPING: arm before the launch, send FRTR at launch + (window - pre) -> record = [-pre, window
    - pre] around the launch. The hardware trigger level is parked out of range so it cannot end the record early."""
    VIDS = ("0X05FF", "1535")
    CHUNK = 32768       # USBTMC request size: asked for >= 64 KB the HDO answers in 65521-byte transfers padded to 65536 (a multiple
                        # of the 512-byte packet, no short packet), which pyvisa-py waits on forever; <= 32 KB requests complete

    def __init__(self, cfg):
        self.cfg = cfg; self.note = ""
        self.inst, self.resource = _visa_open(cfg["resource"], self.VIDS, 30000)
        self.inst.chunk_size = self.CHUNK     # see CHUNK
        self.inst.write("CHDR OFF")
        self.idn = self.inst.query("*IDN?").strip()
        self.npts = int(round(cfg["window_s"] / cfg.get("sample_s", SAMPLE_S)))

    def q(self, s):
        return self.inst.query(s).strip()

    def _num(self, s):
        try:
            return float(self.q(s).split(",")[-1].split()[-1].rstrip("VASs"))
        except Exception:
            return None

    def _vbs(self, expr, timeout_ms=4000):
        old = self.inst.timeout; self.inst.timeout = timeout_ms
        try: return self.q(f"VBS? 'return={expr}'")
        except Exception: return ""
        finally: self.inst.timeout = old

    def _vchan(self, ch, div, center, probe):
        w = self.inst.write
        w(f"C{ch}:TRA ON"); w(f"C{ch}:CPL D1M"); w(f"C{ch}:ATTN {probe}")
        for sc in sorted({div, div * 2, div * 5, div * 10}):
            w(f"C{ch}:VDIV {sc}"); w(f"C{ch}:OFST {-center}")
            got = self._num(f"C{ch}:OFST?")
            if got is not None and abs(got + center) <= 0.05 * max(abs(center), 1.0):
                if sc != div: self.note += f" C{ch} {div}->{sc} V/div (offset range);"
                return sc
        self.note += f" C{ch}: offset {-center} V not accepted up to {div * 10} V/div;"
        return div

    def _ichan(self, ch, div_units, center_units):
        w = self.inst.write; w(f"C{ch}:TRA ON")
        pn = self._vbs(f"app.Acquisition.C{ch}.ProbeName")
        if pn and pn.lower() not in ("", "none", "no probe", "0"):
            self.note += f" C{ch} probe: {pn} (scope reads probe units, I_SCALE should be 1);"
        else:
            w(f"C{ch}:CPL D1M"); w(f"C{ch}:ATTN 1")
        w(f"C{ch}:VDIV {div_units}"); w(f"C{ch}:OFST {-center_units}")
        got = self._num(f"C{ch}:OFST?")
        if got is None or abs(got + center_units) > 0.05 * max(abs(center_units), 1.0):
            self.note += f" C{ch}: offset {-center_units} not accepted (read back {got});"

    def _desc(self, ch):
        return self.parse_wavedesc(self.inst.query_binary_values(f"C{ch}:WF? DESC", datatype="B", container=bytes))

    def _auto_acq(self):
        self.inst.write("TRMD AUTO"); time.sleep(0.4 + 2 * getattr(self, "window_actual", self.cfg["window_s"])); self.inst.write("TRMD STOP")

    def setup(self):
        c, w = self.cfg, self.inst.write
        w("*CLS"); w("TRMD STOP")
        vc, ic = c["v_ch"], c["i_ch"]
        self._vchan(vc, c["v_div"], c["v_center"], c["v_probe"])
        if c.get("v2_ch"):
            self._vchan(c["v2_ch"], c["v_div"], c["v_center"], c["v_probe"])
        if not c.get("no_current"):
            self._ichan(ic, c["i_div"] / c["i_scale"], c["i_center"] / c["i_scale"])
        w("CFMT DEF9,WORD,BIN"); w("CORD LO")
        # ---- timebase: TDIV snaps to 1-2-5; fixed 1 MSa/s. Stop-capture: the record is the window BEFORE the force and the
        # scope ends it 0-45 ms AFTER the force (measured 6 ms at 100 ms, 13 ms at 200 ms, 25-43 ms at 0.5-1 s windows), so
        # the window gets a 60 ms allowance on top of pre + load + post and the force goes out at launch + load + post.
        self.STOP_LAG_ALLOW = 0.060
        want = (c["window_s"] + self.STOP_LAG_ALLOW) / 10
        w(f"TDIV {want}"); tdiv = self._num("TDIV?") or want
        if 10 * tdiv < c["window_s"] + self.STOP_LAG_ALLOW - 1e-6:          # the HDO snapped DOWN (e.g. 281 -> 200 ms/div): step up to the next 1-2-5 value
            import math
            dec = 10 ** math.floor(math.log10(want)); up = next(x * dec for x in (1, 2, 5, 10) if x * dec >= want - 1e-12)
            w(f"TDIV {up}"); tdiv = self._num("TDIV?") or up; self.note += f" TDIV stepped up to {tdiv * 1e3:.0f} ms/div so the record covers the window;"
        self.window_actual = 10 * tdiv
        w("VBS 'app.Acquisition.Horizontal.Maximize = \"FixedSampleRate\"'")
        # the HDO accepts sample rates in 1-2.5-5 steps (20 MSa/s is silently turned into 10 MSa/s): snap the request to that grid,
        # then step down until window x rate fits the 12.5 Mpts memory (otherwise the scope caps the points and truncates the record)
        def snap_down(x):
            m = 10 ** np.floor(np.log10(x)); lead = x / m
            return m * (5 if lead >= 5 else 2.5 if lead >= 2.5 else 1)
        sr = snap_down(1.0 / c.get("sample_s", SAMPLE_S))
        while self.window_actual * sr > HDO_MAX_PTS and sr > 1e3:
            m = 10 ** np.floor(np.log10(sr)); lead = sr / m
            sr = m * (2.5 if lead >= 5 else 1 if lead >= 2.5 else 0.5)
        self.sample_s = 1.0 / sr
        if abs(self.sample_s - c.get("sample_s", SAMPLE_S)) > 1e-12:
            self.note += f" sample spacing {c.get('sample_s', SAMPLE_S)*1e9:.0f} ns does not fit {self.window_actual*1e3:.0f} ms in {HDO_MAX_PTS/1e6:.1f} Mpts -> {self.sample_s*1e9:.0f} ns;"
        w(f"VBS 'app.Acquisition.Horizontal.SampleRate = {sr:.0f}'")
        w(f"MSIZ {int(round(self.window_actual * sr))}")
        w("TRSE EDGE,SR,C%d,HT,OFF" % c["trig_src"]); w("TRMD STOP")
        t0 = None
        for sign in (-1, +1):
            w(f"TRDL {sign * (self.window_actual / 2 - c['pre_s'])}")
            try:
                self._auto_acq(); t0 = self._desc(vc)["t0"]
            except Exception as e:
                self.note += f" timebase check failed: {e!r};"; break
            if abs(t0 + c["pre_s"]) < 0.25 * self.window_actual:
                break
        sr = self._vbs("app.Acquisition.Horizontal.SampleRate")
        self.note += (f" window {self.window_actual * 1e3:.0f} ms (TDIV {tdiv * 1e3:.1f} ms), first sample at {t0 * 1e3:+.1f} ms from the trigger"
                      if t0 is not None else " window/trigger position unverified") + (f", SR {sr};" if sr else ";")
        # ---- "trigger": stop-capture (see the class docstring). The edge level is parked one screen beyond the range of
        # the source channel so a hardware trigger can never end the record before the force does.
        src_is_i = (not c.get("no_current")) and c["trig_src"] == ic
        if src_is_i:
            park = (c["i_center"] + 10 * c["i_div"]) / c["i_scale"]
        else:
            park = c["v_center"] - 10 * c["v_div"] if c["trig_slope"] != "pos" else c["v_center"] + 10 * c["v_div"]
        w(f"C{c['trig_src']}:TRLV {park}"); w(f"C{c['trig_src']}:TRSL {'POS' if c['trig_slope'] == 'pos' else 'NEG'}")
        self.trig_level_used = park
        self.force_delay_s = c["window_s"] - c["pre_s"]               # = load + post: scope_worker sends FRTR this long after the launch
        self.t0_nominal = self.force_delay_s - self.window_actual      # first sample if the record ended exactly at the force
        self.note += (f" capture = stop: FRTR at launch + {self.force_delay_s * 1e3:.0f} ms -> record [{self.t0_nominal * 1e3:.0f}, "
                      f"{self.force_delay_s * 1e3:+.0f}] ms around the launch (+ the scope's stop lag, 0-45 ms, absorbed by the "
                      f"{self.STOP_LAG_ALLOW * 1e3:.0f} ms allowance); edge level parked at {park:g} (out of range);")
        self.inst.query("*OPC?")
        # warm-up acquisition: the first SINGLE after a timebase change was seen to end ~150 ms before the force -> take and discard one
        try:
            self.arm(); time.sleep(0.3); self.force(); t_w = time.monotonic()
            while not self.done() and time.monotonic() - t_w < 5: time.sleep(0.02)
            self.inst.write("TRMD STOP")
        except Exception as e:
            self.note += f" warm-up acquisition failed: {e!r};"

    def arm(self):
        self.inst.write("TRMD SINGLE"); self.inst.query("*OPC?")

    def done(self):
        try:
            return self.inst.query("TRMD?").strip().upper().startswith("STOP")
        except Exception:
            return False

    def stop(self):
        self.inst.write("TRMD STOP")

    def force(self):
        self.inst.write("FRTR")

    @staticmethod
    def parse_wavedesc(b):
        """LeCroy WAVEDESC template (346 bytes): byte order, data type, point count, gain/offset, dt, first-sample time, unit."""
        order = "<" if struct.unpack_from("<h", b, 34)[0] == 1 else ">"
        f = lambda fmt, off: struct.unpack_from(order + fmt, b, off)[0]
        unit = b[196:244].split(b"\0")[0].decode(errors="replace").strip()
        return dict(order=order, comm_type=f("h", 32), wave_array_1=f("i", 60), count=f("i", 116),
                    gain=f("f", 156), offset=f("f", 160), dt=f("f", 176), t0=f("d", 180), unit=unit)

    def _read(self, ch):
        import numpy as np
        d = self._desc(ch)
        dtype = "h" if d["comm_type"] == 1 else "b"
        raw = self.inst.query_binary_values(f"C{ch}:WF? DAT1", datatype=dtype, is_big_endian=(d["order"] == ">"),
                                            container=np.array, chunk_size=self.CHUNK)
        y = raw.astype(np.float64) * d["gain"] - d["offset"]
        t0 = getattr(self, "t0_nominal", None); t0 = d["t0"] if t0 is None else t0
        t = np.arange(len(raw)) * d["dt"] + t0
        self.last_unit = d["unit"]
        return t, y, d["dt"]

    def fetch(self):
        import numpy as np
        t1, v, dt1 = self._read(self.cfg["v_ch"])
        meta = dict(dt=dt1, idn=self.idn, capture="stop", force_delay_s=getattr(self, "force_delay_s", None))
        if self.cfg.get("v2_ch"):
            _, v2, _ = self._read(self.cfg["v2_ch"]); n = min(len(v), len(v2)); meta["v2"] = v2[:n]; t1, v = t1[:n], v[:n]
        if self.cfg.get("no_current"):
            return t1, v, np.full_like(v, np.nan), meta
        t2, i, dt2 = self._read(self.cfg["i_ch"]); meta["i_unit"] = self.last_unit
        n = min(len(v), len(i))
        if "v2" in meta: meta["v2"] = meta["v2"][:n]
        return t1[:n], v[:n], i[:n], meta

    def close(self):
        self.inst.close()


def scope_worker(meter, cfg, armed, stop, q, log, launch=None, launch_t=None):
    """Separate process: arm -> signal armed -> wait for the single acquisition -> fetch -> queue -> re-arm.
    `launch` is set by the main process right before the kernel launch: if the edge trigger has not fired ~3 ms later
    (edge-trigger hysteresis at 0.5 V/div makes it miss randomly) the trigger is forced, so every burst yields a record;
    the plots align on the voltage onset, so a forced record is as usable as an edge-triggered one."""
    import numpy as np
    try:
        scope = (KeysightDSOX if meter == 1 else LecroyHDO)(cfg)
        log.put(f"[scope] {scope.resource} | {scope.idn}")
        if not cfg["no_setup"]:
            scope.setup()
            if getattr(scope, "note", ""):
                log.put("[scope] note:" + scope.note)
        k = 0
        while not stop.is_set():
            scope.arm(); armed.set()
            t_arm = time.monotonic(); forced = False; force_ms = None; t_chk = 0.0
            while True:
                if stop.is_set() or time.monotonic() - t_arm > cfg["wait_s"]:
                    break
                # the main process clears `armed` ~50 ms before it launches: from then on no status query may be in flight,
                # or the force would queue behind its USB round trip (~10 ms; seen as records that start after the kernel)
                imminent = launch is not None and not forced and not armed.is_set()
                if launch is not None and not forced and launch.wait(1.0 if imminent else 0.002):   # launching right now
                    launch.clear()
                    t_l = launch_t.value if (launch_t is not None and launch_t.value > 0) else time.monotonic()
                    delay = getattr(scope, "force_delay_s", 0.0) or 0.0        # LeCroy: FRTR = stop -> send it at launch + window - pre
                    while time.monotonic() < t_l + delay and not stop.is_set():
                        time.sleep(min(0.001, max(0.0, t_l + delay - time.monotonic())))
                    scope.force(); forced = True                                # DSOX: a no-op when the edge trigger already fired
                    force_ms = (time.monotonic() - t_l) * 1e3
                elif launch is None or forced:
                    time.sleep(0.002)
                if imminent and not forced: continue
                if time.monotonic() - t_chk > (0.03 if forced else 0.2):        # done() is a USB round trip: poll it sparsely
                    t_chk = time.monotonic()
                    if scope.done(): break
            if stop.is_set() and not scope.done():
                scope.stop(); break
            if not scope.done():
                log.put(f"[scope] no trigger within {cfg['wait_s']} s, re-arming"); scope.stop(); continue
            t_trig = time.monotonic()
            t, v, i, meta = scope.fetch()
            meta["t_done_mono"] = t_trig; meta["force_sent"] = forced; meta["force_cmd_ms"] = force_ms
            if forced: log.put(f"[scope] force command issued {force_ms:.1f} ms after the launch" + (" (stop-capture: the record ends here)" if meta.get("capture") == "stop" else " (ignored by the scope if the edge had fired)"))
            q.put((k, t, v * 1.0, i * cfg["i_scale"] + cfg["i_offset"], meta))
            log.put(f"[scope] record {k}: {len(v)} pts, dt={meta['dt']*1e6:.2f} us, fetch {time.monotonic()-t_trig:.2f} s"
                    + (" (force sent)" if forced else ""))
            k += 1
        scope.close()
    except Exception as e:
        log.put(f"[scope] ERROR: {e!r}")
    q.put(None)


def pmd_worker(csv_path, ready, stop, log, interval_ms=1.0):
    """Separate process: poll the PMD2X sensor packet as fast as possible and append CSV rows with t_mono."""
    try:
        sys.path.insert(0, HERE)
        import pmd2_fast as pf
        port = pf.find_port()
        if port is None:
            log.put("[pmd] no PMD2X serial port found"); return
        ser = pf.open_device(port)
        f = open(csv_path, "w"); f.write(pf.csv_header() + "\n")
        n = 0; t0 = time.perf_counter(); nxt = t0; target = interval_ms / 1000.0
        while not stop.is_set():
            buf = pf.read_packet(ser)
            if buf is not None:
                f.write(pf.csv_row(buf, time.time(), time.monotonic()) + "\n"); n += 1
                if n == 20:
                    ready.set(); log.put(f"[pmd] {port} polling ({n} rows)")
            nxt += target; sl = nxt - time.perf_counter()
            if sl > 0: time.sleep(sl)
        f.close(); ser.close()
        log.put(f"[pmd] stopped: {n} rows in {time.perf_counter()-t0:.1f} s ({n/(time.perf_counter()-t0):.0f}/s)")
    except Exception as e:
        log.put(f"[pmd] ERROR: {e!r}")


# ============================================================ analysis helpers (used by plot_spike_csv.py too)
def burst_stats(t, v, i, p, on_s):
    import numpy as np
    pre = t < -0.3e-3
    on = (t >= 0.3e-3) & (t < on_s - 0.3e-3)
    if pre.sum() < 10 or on.sum() < 10:
        return None
    has_i = not np.all(np.isnan(i))
    ref = i if has_i else -v
    k = 50; sm = np.convolve(ref, np.ones(k) / k, mode="same")
    r_idle = float(np.mean(sm[pre])); r_set = float(np.median(sm[on])); thr = r_idle + 0.5 * (r_set - r_idle)
    above = np.where(sm > thr)[0]
    t_end = float(t[above[-1]]) if len(above) else on_s
    on = (t >= 0.3e-3) & (t < t_end - 0.3e-3); post = (t > t_end + 0.3e-3); mid = (t >= 0.3 * t_end) & (t < 0.8 * t_end)
    if on.sum() < 10:
        return None
    s = {}
    for name, x in (("v", v), ("i", i), ("p", p)):
        if np.all(np.isnan(x)):
            s[name] = None; continue
        x_idle = float(np.mean(x[pre])); x_set = float(np.median(x[mid])) if mid.sum() else float(np.median(x[on]))
        s[name] = dict(idle=x_idle, settled=x_set, max_on=float(x[on].max()), min_on=float(x[on].min()),
                       t_max_on_us=float(t[on][x[on].argmax()] * 1e6), t_min_on_us=float(t[on][x[on].argmin()] * 1e6),
                       overshoot=float(x[on].max() - x_set), undershoot=float(x_set - x[on].min()),
                       post_max=float(x[post].max()) if post.sum() else None, post_min=float(x[post].min()) if post.sum() else None)
    if s["v"] is None:
        return None
    s["v"]["droop_vs_idle"] = s["v"]["idle"] - s["v"]["min_on"]
    s["v"]["release_overshoot"] = (s["v"]["post_max"] - s["v"]["idle"]) if s["v"]["post_max"] is not None else None
    s["t_end_ms"] = t_end * 1e3
    return s


def parse_freqmon(mbuf, mcnt, maxs, iv_ns):
    import numpy as np
    nm = min(int(mcnt[0].item()), maxs); mon_sm = int(mcnt[1].item())
    dm = mbuf[:2 * nm].view(nm, 2).cpu().numpy().astype(np.float64)
    g = dm[:, 0]; c = dm[:, 1]
    dg = g[1:] - g[:-1]; dc = c[1:] - c[:-1]; ok = (dg > iv_ns * 0.5) & (dg < iv_ns * 2.0)
    f = dc[ok] / dg[ok] * 1e3; x = (g[1:] + g[:-1])[ok] / 2; v = (f > 100) & (f < 3200)
    return x[v], f[v], mon_sm


def drain_log(log):
    if log is None: return
    while True:
        try:
            print(f"[{time.monotonic() % 10000:9.3f}] " + log.get_nowait(), flush=True)
        except Exception:
            return


def pmd_summary(pmd_csv, marks):
    import numpy as np
    if not os.path.exists(pmd_csv):
        return None
    r = csv.reader(open(pmd_csv)); h = next(r); rows = [x for x in r if len(x) == len(h)]
    if not rows:
        return None
    a = np.array(rows, dtype=float); c = {k: i for i, k in enumerate(h)}
    tm = a[:, c["t_mono"]]; praw = a[:, c["PCIE1_W"]] + a[:, c["PCIE2_W"]]
    out = []
    for m in marks:
        s = (tm >= m["t_start"] - 0.002) & (tm <= m["t_end"] + 0.002)
        if s.sum() < 2:
            out.append(None); continue
        x = a[s]
        out.append(dict(p_max=float(praw[s].max()), p_mean=float(praw[s].mean()),
                        v1_min=float(x[:, c["PCIE1_V"]].min()), v1_mean=float(x[:, c["PCIE1_V"]].mean()),
                        v2_min=float(x[:, c["PCIE2_V"]].min()), v2_mean=float(x[:, c["PCIE2_V"]].mean()),
                        i1_max=float(x[:, c["PCIE1_A"]].max()), i2_max=float(x[:, c["PCIE2_A"]].max())))
    idle = praw < 40
    return dict(rows=len(a), rate_hz=float(len(a) / (tm[-1] - tm[0])), idle_p=float(praw[idle].mean()) if idle.any() else None,
                p_max=float(praw.max()), per_burst=out)


def smi_summary(gpu_csv):
    if not gpu_csv or not os.path.exists(gpu_csv):
        return None
    rows = list(csv.reader(open(gpu_csv)))
    if len(rows) < 2:
        return None
    h = [x.strip() for x in rows[0]]
    def col(name):
        j = [k for k, x in enumerate(h) if x.startswith(name)][0]
        vals = []
        for r in rows[1:]:
            try: vals.append(float(r[j].strip().split()[0]))
            except Exception: pass
        return vals
    pi = col("power.draw.instant"); pa = col("power.draw.average"); ck = col("clocks.current.sm")
    return dict(samples=len(pi), p_instant_max=max(pi) if pi else None, p_avg_max=max(pa) if pa else None,
                sm_clock_min=min(ck) if ck else None, sm_clock_max=max(ck) if ck else None)


# x-axis windows for the per-burst figure: (unit, span) -> window [-2*unit, +span) with ticks every unit
WINDOWS = [(1e-6, 20e-6), (10e-6, 200e-6), (100e-6, 2e-3), (1e-3, None)]   # None = whole record


def _unit_label(u):
    return {1e-6: "1 us", 10e-6: "10 us", 100e-6: "100 us", 1e-3: "1 ms"}[u]


V_COL, V2_COL, FQ_COL = "#1f77b4", "#ff7f0e", "#9467bd"   # CH(v) blue, CH(v2) orange, freqmon purple -- same in every figure


def _env(tt, y, max_pts=60000):
    """min/max envelope decimation for plotting long high-rate records (keeps spikes that plain striding would drop)."""
    import numpy as np
    n = len(y)
    if n <= max_pts: return tt, y
    k = int(np.ceil(n / (max_pts // 2))); m = (n // k) * k
    yy = y[:m].reshape(-1, k); tb = tt[:m].reshape(-1, k)
    out_t = np.empty(2 * yy.shape[0]); out_y = np.empty_like(out_t)
    lo = yy.argmin(axis=1); hi = yy.argmax(axis=1); first_lo = lo <= hi
    out_t[0::2] = np.where(first_lo, tb[np.arange(len(lo)), lo], tb[np.arange(len(hi)), hi]); out_y[0::2] = np.where(first_lo, yy.min(axis=1), yy.max(axis=1))
    out_t[1::2] = np.where(first_lo, tb[np.arange(len(hi)), hi], tb[np.arange(len(lo)), lo]); out_y[1::2] = np.where(first_lo, yy.max(axis=1), yy.min(axis=1))
    return out_t, out_y


def _window_sel(t, unit, span):
    lo = -2 * unit; hi = t[-1] if span is None else span
    sel = (t >= lo) & (t <= hi)
    if sel.sum() < 2:
        sel = slice(None)
    scale = 1e6 if unit < 100e-6 else 1e3; xl = "us" if unit < 100e-6 else "ms"
    return sel, scale, xl, lo, hi


def _window_axis(a, MultipleLocator, tt, unit, span, scale, lo, hi, lab, xl, dt_us, device, r, extra_x=""):
    a.axvline(0, color="k", ls="--", lw=0.7)
    n_ticks = (hi - lo) / unit if span is not None else 20
    if span is not None and n_ticks <= 40:
        a.xaxis.set_major_locator(MultipleLocator(unit * scale))
    else:                                               # long window: labels every 1-2-5 x unit (<= ~24 labels), the unit stays as the minor grid
        import math
        raw = (hi - lo) / 24 / unit; step = 10 ** math.floor(math.log10(max(raw, 1)))
        step = step * (1 if raw <= step else 2 if raw <= 2 * step else 5 if raw <= 5 * step else 10)
        a.xaxis.set_major_locator(MultipleLocator(step * unit * scale)); a.xaxis.set_minor_locator(MultipleLocator(unit * scale))
        a.grid(which="minor", alpha=0.15)
    a.set_xlim(tt[0], tt[-1]); a.grid(alpha=0.3); a.tick_params(labelsize=7)
    a.set_ylabel(f"{lab}\n[{_unit_label(unit)} ticks]", fontsize=8)
    if r == 0: a.set_title(f"{device}  {lab}", fontsize=10)
    a.set_xlabel(f"time from {'kernel start (voltage onset)' if device != 'PMD2X' else 'burst start'} ({xl}); 1 sample = {dt_us:.2f} us{extra_x}", fontsize=8)


def _mark_extreme(a, tt, y, kind, fmt, unit, col, ref=None, row=0, where="upper left"):
    """Annotate the max (kind='max') or min of y inside the drawn window: marker + 'max 232 W @ +51.2 ms' box.
    ref = idle level: the label then also shows the difference (e.g. the drop below idle for a voltage channel)."""
    import numpy as np
    if len(y) == 0 or np.all(np.isnan(y)): return
    k = int(np.nanargmax(y) if kind == "max" else np.nanargmin(y)); yk = float(y[k])
    txt = f"{kind} {fmt % yk} {unit} @ {tt[k]:+.1f}" + ("" if ref is None else f"  ({yk - ref:+.3f} {unit} vs idle)")
    a.plot([tt[k]], [yk], marker="v" if kind == "max" else "^", color=col, ms=7, mec="k", mew=0.6, zorder=5)
    xy = (0.02, 0.96 - 0.09 * row) if where == "upper left" else (0.98, 0.04 + 0.09 * row)
    a.annotate(txt, xy=xy, xycoords="axes fraction", fontsize=7, ha="left" if where == "upper left" else "right",
               va="top" if where == "upper left" else "bottom", color=col, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.85))


_CAP_CACHE = {}


def power_limits(run_dir=None):
    """Board power limits for the figures, as a dict {current, default, max, min} in W.
    Source: `power_limits` saved in the run's marks.json at measurement time (nvidia-smi -q -d POWER), else
    nvidia-smi -q -d POWER now (same GPU assumed), else the run's nvidia-smi log `power.limit` column (current only)."""
    import glob as _g, csv as _csv, subprocess, re as _re
    key = run_dir or "__now__"
    if key in _CAP_CACHE: return _CAP_CACHE[key]
    lim = None
    try:
        if run_dir and os.path.exists(os.path.join(run_dir, "marks.json")):
            lim = json.load(open(os.path.join(run_dir, "marks.json"))).get("power_limits") or None
        if lim is None and run_dir:                                   # sequence-level record (run_all.sh writes it at the start)
            for cand in (os.path.join(run_dir, "power_limits.json"), os.path.join(os.path.dirname(os.path.abspath(run_dir)), "power_limits.json")):
                if os.path.exists(cand):
                    try: lim = json.load(open(cand)) or None; break
                    except Exception: pass
        if lim is None:
            smi = next((c for c in (shutil.which("nvidia-smi"), "/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi") if c and os.path.exists(c)), "nvidia-smi")
            out = subprocess.run([smi, "-q", "-d", "POWER"], capture_output=True, text=True, timeout=10).stdout      # absolute path: sudo's PATH lacks /usr/lib/wsl/lib
            got = {}
            for name, k in (("Current Power Limit", "current"), ("Requested Power Limit", "requested"), ("Default Power Limit", "default"),
                            ("Min Power Limit", "min"), ("Max Power Limit", "max")):
                m = _re.search(name + r"\s*:\s*([\d.]+) W", out)
                if m and k not in got: got[k] = float(m.group(1))
            if got: lim = got
        if lim is None and run_dir:
            fs = sorted(_g.glob(os.path.join(run_dir, "gpu_*.csv"))); vals = []
            for row in _csv.reader(open(fs[-1])) if fs else []:
                if row and row[0].startswith("timestamp"): ci = [k for k, x in enumerate(row) if x.strip().startswith("power.limit")]; continue
                try: vals.append(float(row[ci[0]].strip().split()[0]))
                except Exception: pass
            if vals: lim = dict(current=sorted(vals)[len(vals) // 2])
    except Exception:
        lim = None
    _CAP_CACHE[key] = lim; return lim


power_cap_w = power_limits          # old name


def _cap_line(a, lim):
    """Power-limit markers on a power axis: dashed line at the default limit (and the current limit when it differs),
    dotted line at the max limit, from nvidia-smi -q -d POWER; a text note instead when the axis range is far below."""
    if not lim: return
    if isinstance(lim, (int, float)): lim = dict(current=float(lim))
    dflt = lim.get("default") or lim.get("current"); cur = lim.get("current"); mx = lim.get("max")
    marks = []
    if dflt: marks.append((dflt, "--", "gray", f"power cap {dflt:.0f} W (default)" if cur in (None, dflt) else f"default limit {dflt:.0f} W"))
    if cur and dflt and abs(cur - dflt) > 0.5: marks.append((cur, "--", "#444444", f"current limit {cur:.0f} W"))
    if mx and all(abs(mx - m[0]) > 0.5 for m in marks): marks.append((mx, ":", "#8c564b", f"max power limit {mx:.0f} W"))
    if not marks: return
    lo, hi = a.get_ylim()
    if hi < 0.5 * min(m[0] for m in marks):
        a.annotate(" / ".join(f"{m[3]}" for m in marks) + "  (above this range)", xy=(0.98, 0.96), xycoords="axes fraction", ha="right", va="top", fontsize=6.5, color="gray"); return
    top = max(m[0] for m in marks)
    for yv, ls, col, lab in marks:
        a.axhline(yv, color=col, ls=ls, lw=0.9, zorder=1)
        a.annotate(lab, xy=(0.98, yv), xycoords=("axes fraction", "data"), xytext=(0, -2), textcoords="offset points", ha="right", va="top", fontsize=6.5, color=col)
    if hi < top * 1.04: a.set_ylim(lo, top * 1.04)


def plot_burst(path, t, v, i, p, title, windows=WINDOWS, device="scope", dt_us=1.0, v2=None, v_lab="rail voltage (V)", v2_lab=None, cap_w=None):
    """One PNG per burst: rows = x windows (1 us / 10 us / 100 us / 1 ms units), cols = P, I, V.
    v2 (second voltage channel) is drawn on the V axes in a second colour."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
        from matplotlib.ticker import MultipleLocator
    except Exception as e:
        print("plot skipped:", e); return
    cols = [(p, "P = V x I (W)", "#d62728"), (i, "current (A)", "#2ca02c"), (v, "rail voltage (V)", V_COL)]
    cols = [c for c in cols if not np.all(np.isnan(c[0]))]            # no current probe -> voltage column only
    fig, ax = plt.subplots(len(windows), len(cols), figsize=(5.7 * len(cols) + 2, 3.0 * len(windows)), squeeze=False)
    for r, (unit, span) in enumerate(windows):
        sel, scale, xl, lo, hi = _window_sel(t, unit, span)
        tt = t[sel] * scale
        lw = 0.7 if unit >= 100e-6 else 1.0; mk = "." if unit == 1e-6 else None
        for cidx, (y, lab, col) in enumerate(cols):
            a = ax[r, cidx]; is_v = y is v
            mk_ = mk if len(tt) <= 400 else None
            a.plot(*_env(tt, y[sel]), lw=lw, color=col, marker=mk_, ms=3, label=v_lab if is_v else None)
            if is_v and v2 is not None:
                a.plot(*_env(tt, v2[sel]), lw=lw, color=V2_COL, marker=mk_, ms=3, alpha=0.85, label=v2_lab or "2nd channel")
                a.legend(loc="lower right", fontsize=7)
            _window_axis(a, MultipleLocator, tt, unit, span, scale, lo, hi, lab, xl, dt_us, device, r)
            # extreme-value labels: max for P and I, min (= deepest drop) for the voltage channels, idle = median before t=0
            pre = t < -0.5e-3
            if is_v:
                _mark_extreme(a, tt, v[sel], "min", "%.3f", "V", col, ref=float(np.median(v[pre])) if pre.sum() > 50 else None, row=0)
                if v2 is not None: _mark_extreme(a, tt, v2[sel], "min", "%.3f", "V", V2_COL, ref=float(np.median(v2[pre])) if pre.sum() > 50 else None, row=1)
            else:
                _mark_extreme(a, tt, y[sel], "max", "%.1f", "W" if y is p else "A", col)
                if y is p: _cap_line(a, cap_w)
                if y is i and unit in (10e-6, 100e-6):                       # current rise time 10-90 % (idle -> 1-5 ms plateau)
                    try:
                        from onset import rise_time, max_slope
                        rt = rise_time(t, i, 0.0); sl, at = max_slope(t, i, 0.0)
                        if rt and rt["rise_s"] is not None:
                            for tk, frac in ((rt["t10"], 0.1), (rt["t90"], 0.9)):
                                a.plot([tk * scale], [rt["idle"] + frac * rt["step"]], marker="o", ms=6, mfc="none", mec="k", mew=1.0, zorder=6)
                            a.annotate(f"rise 10-90 % = {rt['rise_s']*1e6:.0f} us ({rt['idle']:.1f} -> {rt['level']:.1f} A plateau; leaves idle at {rt['t_start']*1e6:+.0f} us)"
                                       + (f"; max di/dt {sl/1e6:.3f} A/us @ {at*1e6:+.0f} us" if sl else ""),
                                       xy=(0.02, 0.86), xycoords="axes fraction", fontsize=7, ha="left", va="top", color="k",
                                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="k", alpha=0.85))
                    except Exception:
                        pass
    fig.suptitle(title, fontsize=11); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def plot_rise(path, t, y, title, unit="A", s0=0.0, color="#2ca02c"):
    """ONE panel: the current step at the kernel start with the 10 % / 90 % points and the rise time written out.
    Returns the rise dict (onset.rise_time) or None."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
        from onset import rise_time, max_slope
    except Exception as e:
        print("plot skipped:", e); return None
    rt = rise_time(t, y, s0)
    if not rt or rt["rise_s"] is None: return None
    sl, at = max_slope(t, y, s0)
    t0 = rt["t_start"] if rt["t_start"] is not None else rt["t10"]
    span = max(5 * (rt["t90"] - t0), 50e-6); sel = (t >= t0 - 0.3 * span) & (t <= t0 + span)
    fig, a = plt.subplots(figsize=(9, 4.2))
    tt = (t[sel] - t0) * 1e6; a.plot(*_env(tt, y[sel], 40000), color=color, lw=0.9)
    for tk, fr, lab in ((rt["t10"], 0.1, "10 %"), (rt["t90"], 0.9, "90 %")):
        a.plot([(tk - t0) * 1e6], [rt["idle"] + fr * rt["step"]], "o", ms=8, mfc="none", mec="k", mew=1.3, zorder=6)
        a.axvline((tk - t0) * 1e6, color="k", ls=":", lw=0.8)
    a.axhline(rt["idle"], color="gray", ls="--", lw=0.8); a.axhline(rt["level"], color="gray", ls="--", lw=0.8)
    a.text(0.98, 0.06, f"rise time 10-90 % = {rt['rise_s']*1e6:.1f} us\n{rt['idle']:.2f} -> {rt['level']:.2f} {unit} (plateau 1-5 ms)\n"
                       f"leaves idle at t = 0, 90 % at {(rt['t90'] - t0)*1e6:+.0f} us" + (f"\nmax di/dt {sl/1e6:.3f} {unit}/us" if sl else ""),
           transform=a.transAxes, ha="right", va="bottom", fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="k", alpha=0.9))
    dt = float(np.median(np.diff(t[:1000]))) if len(t) > 1 else 1e-6
    a.set_xlabel(f"time from the start of the current rise (us); 1 sample = {dt*1e9:.0f} ns"); a.set_ylabel(f"current ({unit})"); a.grid(alpha=0.3)
    a.set_title(title if len(title) <= 90 else title[:87] + "...", fontsize=9); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)
    return rt


def _arch_flags():
    """nvcc flags for the attached GPU (sm_120 on the RTX 5060 Ti, sm_90 on H100, ...)."""
    import torch
    mj, mn = torch.cuda.get_device_capability(0)
    return ["-O3", "-gencode", f"arch=compute_{mj}{mn},code=sm_{mj}{mn}"]


def _cuda_env():
    """torch.utils.cpp_extension needs CUDA_HOME/nvcc and ninja: taken from CUDA_HOME, else ../cuda*, ~/cuda*, /usr/local/cuda (sudo's PATH has none of them)."""
    import glob as _g
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get("CUDA_HOME")] + sorted(_g.glob(os.path.join(here, "..", "cuda*")), reverse=True) + sorted(_g.glob(os.path.expanduser("~/cuda*")), reverse=True) + ["/usr/local/cuda"]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "bin", "nvcc")):
            os.environ["CUDA_HOME"] = os.path.abspath(c); os.environ["PATH"] = os.path.join(os.path.abspath(c), "bin") + os.pathsep + os.environ.get("PATH", ""); break
    vb = os.path.join(sys.prefix, "bin")
    if os.path.exists(os.path.join(vb, "ninja")) and vb not in os.environ.get("PATH", ""):
        os.environ["PATH"] = vb + os.pathsep + os.environ["PATH"]


def crop_to_last_load(t, v, on_s, pre_s, gap_s):
    """Mask keeping the record from min(pre, gap - 1 ms) before the start of the LAST load segment (the measured kernel) -
    the warm-up GEMM before it is dropped. Load = smoothed rail voltage below idle - 50 % of the record's excursion; the last
    segment at least 0.4 x on_s long wins. None when no such segment exists (the caller then keeps the whole record)."""
    import numpy as np
    from onset import smooth
    from onset import n_for
    dt = float(np.median(np.diff(t[:1000]))) if len(t) > 1 else 1e-6
    sm_s = min(1e-3, max(50e-6, on_s / 5))                        # smoothing must not erase the measured load itself: a 0.3 ms wave kernel
    vs = smooth(v, n_for(t, sm_s)); hi = np.percentile(vs, 99.5); lo = np.percentile(vs, 2)    # vanished under 1 ms smoothing and the crop then kept the
    # 500 ms all-SM warm-up as "the last load" (onset landed inside it, idle read 11.9-12.06 V; 2026-11-24). <= 1 ms: ripple cannot split a load;
    # hi = idle level even when idle is only a few % of a long record (warm-up + 1.8 s burst), hence the 99.5th percentile
    if hi - lo < 0.02: return None
    load = vs < hi - 0.5 * (hi - lo)
    edges = np.flatnonzero(np.diff(load.astype(np.int8))); starts = [e + 1 for e in edges if load[e + 1]]; ends = [e + 1 for e in edges if not load[e + 1]]
    if load[0]: starts = [0] + starts
    if load[-1]: ends = ends + [len(load)]
    segs = list(zip(starts, ends)); merged = []
    for s_, e_ in segs:                                       # bridge gaps shorter than 5 ms (DVFS steps, limiter dips) but not the >= 10 ms warm-up gap
        if merged and (s_ - merged[-1][1]) * dt < 5e-3: merged[-1] = (merged[-1][0], e_)
        else: merged.append((s_, e_))
    segs = [(s_, e_) for s_, e_ in merged if (e_ - s_) * dt >= min(0.4 * on_s, 0.4 * on_s if on_s > 2e-3 else 0.25 * on_s)]
    if not segs: return None
    s0 = segs[-1][0]; back = min(pre_s, max(gap_s - 1e-3, 2e-3))
    return t >= t[s0] - back


def scope_onset(t, v):
    """kernel start on the scope axis (onset.py: smoothed, robust idle, works for drops of a few tens of mV)."""
    from onset import onset_from_drop
    return onset_from_drop(t, v)[0]


def plot_burst_freq(path, t, v, p, fq_t, fq_f, title, windows=WINDOWS, device="scope", dt_us=1.0, v2=None, v_lab="rail voltage (V)", v2_lab=None, cap_w=None):
    """burst_NN_freq.png: the same x windows as burst_NN.png, rail voltage (both channels, two colours) with the in-kernel
    freqmon SM clock overlaid on a right axis; a P column is added when a current probe is present.
    fq_t is in the scope-trigger frame (freqmon first-launch stamp aligned to the voltage onset)."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
        from matplotlib.ticker import MultipleLocator
    except Exception as e:
        print("plot skipped:", e); return
    have_p = p is not None and not np.all(np.isnan(p))
    ncol = 2 if have_p else 1
    fig, ax = plt.subplots(len(windows), ncol, figsize=(9.5 * ncol, 3.2 * len(windows)), squeeze=False)
    note = ";  freqmon: 1st-launch GPU stamp = 0"
    for r, (unit, span) in enumerate(windows):
        sel, scale, xl, lo, hi = _window_sel(t, unit, span)
        tt = t[sel] * scale
        lw = 0.7 if unit >= 100e-6 else 1.0; mk = "." if unit == 1e-6 else None
        fs = (fq_t >= lo) & (fq_t <= hi) if fq_t is not None else None
        cols = [(v, "rail voltage (V)", V_COL, True)] + ([(p, "P = V x I (W)", "#d62728", False)] if have_p else [])
        for cidx, (y, lab, col, is_v) in enumerate(cols):
            a = ax[r, cidx]; mk_ = mk if len(tt) <= 400 else None
            a.plot(*_env(tt, y[sel]), lw=lw, color=col, marker=mk_, ms=3, label=v_lab if is_v else lab)
            if is_v and v2 is not None:
                a.plot(*_env(tt, v2[sel]), lw=lw, color=V2_COL, marker=mk_, ms=3, alpha=0.85, label=v2_lab or "2nd channel")
            _window_axis(a, MultipleLocator, tt, unit, span, scale, lo, hi, lab, xl, dt_us, device, r, extra_x=note)
            pre = t < -0.5e-3
            if is_v:
                _mark_extreme(a, tt, v[sel], "min", "%.3f", "V", col, ref=float(np.median(v[pre])) if pre.sum() > 50 else None, row=0)
                if v2 is not None: _mark_extreme(a, tt, v2[sel], "min", "%.3f", "V", V2_COL, ref=float(np.median(v2[pre])) if pre.sum() > 50 else None, row=1)
            else:
                _mark_extreme(a, tt, y[sel], "max", "%.1f", "W", col); _cap_line(a, cap_w)
            a2 = a.twinx()
            if fs is not None and fs.sum():
                a2.step(fq_t[fs] * scale, fq_f[fs], where="post", color=FQ_COL, lw=0.9, alpha=0.9, marker="s" if unit <= 10e-6 else None, ms=3,
                        label="freqmon SM clock (in-kernel, 10 us)")
                _mark_extreme(a2, fq_t[fs] * scale, fq_f[fs], "max", "%.0f", "MHz", FQ_COL, row=0, where="lower right")
                _mark_extreme(a2, fq_t[fs] * scale, fq_f[fs], "min", "%.0f", "MHz", FQ_COL, row=1, where="lower right")
                a2.set_ylabel("SM clock (MHz)", color=FQ_COL, fontsize=8); a2.tick_params(labelsize=7, colors=FQ_COL)
                a2.yaxis.get_major_formatter().set_useOffset(False)
                if fq_f[fs].max() - fq_f[fs].min() < 50: a2.set_ylim(fq_f[fs].min() - 30, fq_f[fs].max() + 30)
                a2.legend(loc="center right", fontsize=7)
            else:
                a2.set_yticks([]); a2.text(0.99, 0.05, "no freqmon samples in window", ha="right", transform=a.transAxes, fontsize=7, color="gray")
            a.legend(loc="upper right", fontsize=7)
    fig.suptitle(title.replace("  measured with", "\nmeasured with") + "\noverlay: rail voltage (both channels) + in-kernel freqmon SM clock", fontsize=10); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def pmd_burst_records(pmd_csv, marks, pad_s=0.05):
    """Per-burst (t, v, i, p) from the PMD2X 1 ms log (PCIE1+PCIE2 cables), t relative to burst start."""
    import numpy as np
    if not os.path.exists(pmd_csv):
        return []
    r = csv.reader(open(pmd_csv)); h = next(r); rows = [x for x in r if len(x) == len(h)]
    if not rows:
        return []
    a = np.array(rows, dtype=float); c = {k: j for j, k in enumerate(h)}
    tm = a[:, c["t_mono"]]
    out = []
    for m in marks:
        s = (tm >= m["t_start"] - pad_s) & (tm <= m["t_end"] + pad_s)
        if s.sum() < 2:
            out.append(None); continue
        x = a[s]
        i_ = x[:, c["PCIE1_A"]] + x[:, c["PCIE2_A"]]
        v_ = (x[:, c["PCIE1_V"]] * x[:, c["PCIE1_A"]] + x[:, c["PCIE2_V"]] * x[:, c["PCIE2_A"]]) / np.maximum(i_, 1e-3)
        p_ = x[:, c["PCIE1_W"]] + x[:, c["PCIE2_W"]]
        out.append((tm[s] - m["t_start"], v_, i_, p_))
    return out


def make_plot(outdir, records, on_ms, meter):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
    except Exception as e:
        print("plot skipped:", e); return
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    cm = plt.get_cmap("viridis")
    for k, (t, v, i, p) in enumerate(records):              # envelope-decimated: 10 records x 5 M points took > 10 GB in matplotlib
        col = cm(k / max(1, len(records) - 1)); ms = t * 1e3
        ax[0].plot(*_env(ms, p, 40000), lw=0.5, color=col, label=f"burst {k}" if k < 10 else None)
        ax[1].plot(*_env(ms, v, 40000), lw=0.5, color=col); ax[2].plot(*_env(ms, i, 40000), lw=0.5, color=col)
    for x in ax:
        x.axvline(0, color="k", ls="--", lw=0.8); x.axvline(on_ms, color="k", ls=":", lw=0.8); x.grid(alpha=0.3)
    _cap_line(ax[0], power_cap_w(outdir))
    ax[0].set_ylabel("P = V x I (W)"); ax[1].set_ylabel("CH1 V"); ax[2].set_ylabel("CH2 A"); ax[2].set_xlabel("ms from trigger")
    ax[0].legend(fontsize=7, ncol=5); ax[0].set_title(f"{'DSOX1202A' if meter == 1 else 'HDO4034A'}  1 us/sample  {len(records)} bursts of {on_ms:.0f} ms")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "summary.png"), dpi=120); plt.close(fig)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--meter", type=int, choices=[0, 1, 2],
                    help="0: PMD2X + nvidia-smi + freqmon | 1: + Keysight DSOX1202A | 2: + LeCroy HDO4034A (required)")
    ap.add_argument("-b", "--nburst", type=int, default=10, help="number of bursts (measurement periods)")
    ap.add_argument("-o", "--on-ms", type=float, default=60, help="burst (GPU load) length per iteration, ms")
    ap.add_argument("-d", "--off-ms", type=float, default=1000, help="idle gap between bursts, ms")
    g = ap.add_argument_group("workload")
    g.add_argument("--size", type=int, default=8192, help="N: B is NxN, K=N (8192 = strongest spike)")
    g.add_argument("--rows", type=int, default=None, help="rows of A (default = size; e.g. 2048 -> ~17 ms matmul granularity)")
    g.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32", help="fp32 reaches ~230-245 W peaks; bf16 ~170 W")
    g.add_argument("--v-name", default=None, help="label of the voltage channel in figures/tables, e.g. PSU_12V_cable (underscores -> spaces)")
    g.add_argument("--v2-name", default=None, help="label of the second voltage channel, e.g. PCIe_slot_12V")
    g.add_argument("--cols", type=int, default=None, help="N: columns of B/C (default = size); B is [size x N]")
    g.add_argument("--kernel", choices=["cublas", "wave", "ws"], default="cublas",
                   help="cublas = torch.matmul (cuBLAS picks tiles/split-K); wave = wavegemm.cu, one 256x128 tile per CTA, 1 CTA/SM "
                        "-> exact wave count (fp32 only); ws = wsgemm.cu, warp-specialized bf16 GEMM (producer warps load, "
                        "consumer warps mma) with the loader pacing knobs below")
    g.add_argument("--ws-variant", type=int, default=0, help="wsgemm variant id (tile / stages / warp groups; ws_bench.py lists them)")
    g.add_argument("--ws-depth", type=int, default=0, help="loader run-ahead in stages (0 = all stages)")
    g.add_argument("--ws-spread", type=float, default=0.0, help="trickle: > 0 moves the loader's idle time into pauses between chunk groups (0 = burst)")
    g.add_argument("--ws-groups", type=int, default=0, help="trickle: chunk groups per tile (0 = one per cp.async instruction)")
    g.add_argument("--ws-slack-ns", type=int, default=96, help="trickle: waiting time per tile the loader keeps as slack")
    g.add_argument("--ws-group-m", type=int, default=8, help="rasterization group along M (L2 reuse)")
    g.add_argument("--ws-stagger-us", type=float, default=0.0, help="soft start: i-th CTA of the first wave starts i*us late (first launch of a cycle only)")
    g.add_argument("--ws-ramp-us", type=float, default=0.0, help="soft start: duty ramp duration (first launch of a cycle only)")
    g.add_argument("--ws-duty0", type=float, default=0.1, help="soft start: initial duty")
    g.add_argument("--ws-slot-us", type=float, default=100.0, help="soft start: PWM slot length")
    g.add_argument("--ws-noload", action="store_true", help="debug: compute only (no global-memory loads after the first tiles)")
    g.add_argument("--ws-nocompute", type=int, default=0, help="debug: loader only; 1 = free-running loader, N > 1 = consumers sleep N ns per tile instead of computing")
    g.add_argument("--launches", type=int, default=None, help="fixed launches per cycle (overrides the -o/--on-ms rounding); 1 = single cycle")
    g.add_argument("--prewarm-ms", type=float, default=0.0,
                   help="warm clock: minimum duration of the warm-up load before the gap (count-bound launches of the measured GEMM until this much "
                        "GPU time has run; 0 = one launch). From P8 idle (no freqmon) the controller needs ~200 ms awake before it stops re-tuning")
    g.add_argument("--prewarm-gap-ms", type=float, default=0.0,
                   help="warm clock: one extra GEMM of the same shape ends this many ms before the measured launch, so the clock controller is already "
                        "at its settled level (<= 20 ms keeps ~2750 MHz from t = 0 on the 5060 Ti; >= 50 ms the 30 ms step at ~2640 MHz returns). "
                        "The 12 V current is back at idle within 1 ms, so the di/dt step is unchanged. HDO stop-capture only (the DSOX edge trigger would fire on the warm-up)")
    g.add_argument("--idle-before", type=float, default=3.0, help="s idle before the first burst")
    g.add_argument("--idle-after", type=float, default=2.0, help="s idle after the last burst")
    g = ap.add_argument_group("PMD2X / nvidia-smi / freqmon (always on)")
    g.add_argument("--no-pmd", action="store_true", help="skip the PMD2X poller")
    g.add_argument("--pmd-ms", type=float, default=1.0, help="PMD2X poll interval, ms")
    g.add_argument("--smi-ms", type=int, default=100, help="nvidia-smi interval, ms")
    g.add_argument("--no-freqmon", action="store_true", help="skip the in-kernel SM clock monitor")
    g.add_argument("--freqmon-us", type=float, default=10.0, help="freqmon sample interval, us")
    g.add_argument("--freqmon-smem-kb", type=int, default=96, help="shared memory the monitor holds so no GEMM CTA shares its SM (<=99)")
    g = ap.add_argument_group("oscilloscope (-m 1/2)")
    g.add_argument("--resource", help="VISA resource (default: first USB device of the scope's vendor), e.g. TCPIP::10.0.0.5::INSTR")
    g.add_argument("--no-setup", action="store_true", help="keep the front-panel settings (only arm/fetch)")
    g.add_argument("--no-current", action="store_true", help="voltage channel only (no current probe): skip I and P, trigger on V")
    g.add_argument("--v-ch", type=int, choices=[1, 2, 3, 4], default=2, help="scope channel carrying the rail voltage")
    g.add_argument("--i-ch", type=int, choices=[1, 2, 3, 4], default=1, help="scope channel carrying the current probe")
    g.add_argument("--v2-ch", type=int, choices=[1, 2, 3, 4], default=None, help="second voltage channel (same probe/scale), e.g. to compare two 12 V cables")
    g.add_argument("--v-probe", type=float, default=1, help="voltage probe attenuation (1 or 10)")
    g.add_argument("--v-div", type=float, default=0.5, help="V/div (0.5 keeps 12 V on screen with a 1:1 probe, halves trigger hysteresis vs 1 V/div)")
    g.add_argument("--v-center", type=float, default=12.0, help="voltage channel screen centre, V")
    g.add_argument("--i-scale", type=float, default=1.0, help="current probe scale, A per V (100 mV/A probe -> 10)")
    g.add_argument("--i-offset", type=float, default=0.0, help="current offset to subtract, A")
    g.add_argument("--i-div", type=float, default=5.0, help="current A/div")
    g.add_argument("--i-center", type=float, default=7.5, help="current screen centre, A")
    g.add_argument("--trig-src", choices=["v", "i"], default="v", help="trigger on the voltage or the current channel")
    g.add_argument("--trig-level", default="11.92", help="trigger level in channel units (V for v, A for i), or 'auto[:delta]' = idle rail voltage minus delta (default 0.15 V = above the ~0.1 V hysteresis at 0.5 V/div)")
    g.add_argument("--trig-slope", choices=["pos", "neg"], default="neg")
    g.add_argument("--scope-sample-ns", type=float, default=1000.0, help="scope sample spacing in ns (1000 = 1 us; HDO: 1-2-5 steps, 20 ns = 50 MSa/s; "
                   "the rate is lowered automatically when window / spacing exceeds the 12.5 Mpts memory). csv files stay at >= 1 us, npz keeps the full rate")
    g.add_argument("--pre-ms", type=float, default=15.0, help="ms captured before the trigger (>= 10 so a forced trigger a few ms after the launch still holds the onset)")
    g.add_argument("--post-ms", type=float, default=15.0, help="ms captured after the burst end")
    g.add_argument("--wait", type=float, default=30.0, help="s the scope waits for a trigger before re-arming")
    g.add_argument("--free-run", action="store_true", help="do not wait for the scope to be armed before each burst")
    ap.add_argument("--no-plots", action="store_true", help="skip plot_spike_csv.py at the end")
    ap.add_argument("--outdir", default=None, help="parent directory for the run folder (default: next to this script)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.meter is None:
        ap.print_help(); sys.exit(2)

    _cuda_env()                                       # CUDA_HOME / nvcc / ninja on PATH (portable: env, ../cuda*, /usr/local/cuda)
    import numpy as np, torch
    stamp = time.strftime("%Y%m%d_%H%M%S") + (f"_{a.tag}" if a.tag else "")
    outdir = os.path.join(a.outdir or HERE, f"spike_eval_{stamp}"); os.makedirs(outdir, exist_ok=True)
    print("output dir:", outdir, flush=True)

    # ---------- workload + pilot (thorough warm-up, then single-launch latency)
    dt = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    if a.prewarm_gap_ms > 0 and a.meter == 1: sys.exit("--prewarm-gap-ms needs the stop-capture scope path (LeCroy, -m 2): the DSOX edge trigger would fire on the warm-up GEMM")
    M = a.rows or a.size; N_ = a.cols or a.size; dev = torch.device("cuda")
    A = torch.randn(M, a.size, device=dev, dtype=dt); B = torch.randn(a.size, N_, device=dev, dtype=dt)
    C = torch.empty(M, N_, device=dev, dtype=dt); torch.cuda.synchronize()
    wave_info = None; ws_info = None; mm_first = None
    if a.kernel == "wave":
        if dt != torch.float32: sys.exit("--kernel wave is fp32 only")
        _cuda_env()
        from torch.utils.cpp_extension import load
        wg = load(name="wavegemm", sources=[os.path.join(HERE, "wavegemm.cu")],
                  extra_cuda_cflags=_arch_flags(), verbose=False)
        tm_, tn_ = wg.tile_m(), wg.tile_n()
        if M % tm_ or N_ % tn_: sys.exit(f"--kernel wave needs M multiple of {tm_} and N multiple of {tn_}")
        ctas = (M // tm_) * (N_ // tn_); occ = wg.max_blocks_per_sm(); sms = wg.sm_count(); usable = sms - (0 if a.no_freqmon else 1)
        wave_info = dict(kernel=f"wavegemm {tm_}x{tn_} tile/CTA", tile=[tm_, tn_], ctas=ctas, ctas_per_sm=occ, sms=sms, usable_sms=usable, waves=ctas / (usable * occ))
        print(f"[wave ] {ctas} CTAs (M/{tm_} x N/{tn_}), {occ} CTA/SM, {sms} SMs - {sms-usable} (freqmon) = {usable} usable -> {wave_info['waves']:.3f} waves", flush=True)
        mm = lambda: wg.gemm(A, B, C)
    elif a.kernel == "ws":
        if dt != torch.bfloat16: sys.exit("--kernel ws is bf16 only (use --dtype bf16)")
        _cuda_env()
        from torch.utils.cpp_extension import load
        wsm = load(name="wsgemm", sources=[os.path.join(HERE, "ws_gemm", "wsgemm.cu")], extra_cuda_cflags=_arch_flags(), verbose=False)
        wv = wsm.variants()[a.ws_variant]
        if M % wv["BM"] or N_ % wv["BN"] or a.size % wv["BK"]: sys.exit(f"--kernel ws variant {wv['name']}: M, N, K must be multiples of {wv['BM']}, {wv['BN']}, {wv['BK']}")
        ws_kw = dict(variant=a.ws_variant, depth=a.ws_depth, spread=a.ws_spread, groups=a.ws_groups, slack_ns=a.ws_slack_ns, group_m=a.ws_group_m,
                     noload=a.ws_noload, nocompute=a.ws_nocompute)
        ws_soft = dict(stagger_us=a.ws_stagger_us, ramp_us=a.ws_ramp_us, duty0=a.ws_duty0, slot_us=a.ws_slot_us)
        ws_info = dict(name=wv["name"], stages=wv["stages"], consumer_warps=wv["cw"], producer_warps=wv["pw"], smem=wv["smem"],
                       ctas=(M // wv["BM"]) * (N_ // wv["BN"]), **ws_kw, **ws_soft)
        print(f"[ws   ] {wv['name']}: tile {wv['BM']}x{wv['BN']}x{wv['BK']}, {wv['stages']} stages, {wv['cw']} consumer + {wv['pw']} producer warps, "
              f"depth {a.ws_depth or wv['stages']}, spread {a.ws_spread} / {a.ws_groups or wv['ni']} groups, soft start: ramp {a.ws_ramp_us} us, stagger {a.ws_stagger_us} us/SM", flush=True)
        mm = lambda: wsm.gemm(A, B, C, **ws_kw)
        if a.ws_stagger_us > 0 or a.ws_ramp_us > 0:
            mm_first = lambda: wsm.gemm(A, B, C, **ws_kw, **ws_soft)       # soft start only on the first launch after idle
    else:
        mm = lambda: torch.matmul(A, B, out=C)
    if mm_first is None: mm_first = mm
    # the workload lives on its own stream: the legacy default stream would implicitly synchronize with the
    # long-running freqmon kernel and block every launch until the monitor ends (observed: 41 s stall)
    s2 = torch.cuda.Stream(); torch.cuda.set_stream(s2)
    t_w = time.perf_counter(); nw = 0
    while nw < 20 or time.perf_counter() - t_w < 2.0:
        mm(); nw += 1
        if nw % 4 == 0: s2.synchronize()
    s2.synchronize()
    lat = []
    for _ in range(10):
        s_ev = torch.cuda.Event(enable_timing=True); e_ev = torch.cuda.Event(enable_timing=True)
        s_ev.record(s2); mm(); e_ev.record(s2); s2.synchronize()
        lat.append(s_ev.elapsed_time(e_ev))
    lat = np.array(lat); t_one = float(np.median(lat))
    K = a.launches if a.launches else max(1, int(round(a.on_ms / t_one))); on_ms = K * t_one
    print(f"[pilot] {a.dtype} {a.kernel} [{M}x{a.size}] @ [{a.size}x{N_}] after {nw} warm-up launches ({time.perf_counter()-t_w:.1f} s):")
    print(f"[pilot] 1-launch latency = {t_one:.3f} ms = {t_one*1e3:.0f} us  (10 runs: min {lat.min():.3f} / max {lat.max():.3f} ms, "
          f"{2*M*N_*a.size/t_one/1e9:.1f} TFLOP/s)")
    cyc = "single cycle (1 launch)" if K == 1 else f"burst of K={K} launches"
    print(f"[plan ] GEMM M={M} N={N_} K={a.size} ({a.dtype}, {a.kernel}): {cyc} = {on_ms:.1f} ms (target {a.on_ms:.0f} ms), off {a.off_ms:.0f} ms, {a.nburst} cycles", flush=True)

    # ---------- freqmon + globaltimer stamp extensions (compiled once, cached)
    fm = gs = None; IV = int(a.freqmon_us * 1000)
    _cuda_env()
    from torch.utils.cpp_extension import load
    gs = load(name="gtstamp", sources=[os.path.join(HERE, "gtstamp.cu")],      # %globaltimer stamps around every cycle (gpu_ms),
              extra_cuda_cflags=_arch_flags(), verbose=False)                 # also without freqmon (--no-freqmon)
    if not a.no_freqmon:
        fm = load(name="freqmon_excl", sources=[os.path.join(HERE, "freqmon_excl.cu")],
                  extra_cuda_cflags=_arch_flags(), verbose=False)
        total_s = 3 * (a.idle_before + a.nburst * (on_ms / 1e3 + a.off_ms / 1e3 + (4.0 if a.meter else 0.5)) + a.idle_after) + 60
        MAXS = int(total_s * 1e9 / IV) + 1000
        mbuf = torch.zeros(2 * MAXS, dtype=torch.int64, device=dev); mcnt = torch.zeros(2, dtype=torch.int32, device=dev)
        mstop = torch.zeros(1, dtype=torch.int32, device=dev); ms = torch.cuda.Stream()
    stamps = torch.zeros(max(1, a.nburst) * 2, dtype=torch.int64, device=dev)   # [2k] = before burst k, [2k+1] = after
    if gs is not None:
        # first launch of a freshly loaded module triggers lazy module loading, which blocks until every running
        # kernel (i.e. freqmon) has finished -> warm it up before the monitor starts
        gs.stamp(stamps, 0); gs.stamp(stamps, 1); s2.synchronize(); stamps.zero_()
    torch.cuda.synchronize()

    ctx = mp.get_context("spawn")
    log = ctx.Queue()
    # ---------- (1) PMD2X poller process
    pmd_p = pmd_ready = pmd_stop = None; pmd_csv = os.path.join(outdir, "pmd.csv")
    if not a.no_pmd:
        pmd_ready, pmd_stop = ctx.Event(), ctx.Event()
        pmd_p = ctx.Process(target=pmd_worker, args=(pmd_csv, pmd_ready, pmd_stop, log, a.pmd_ms), daemon=True); pmd_p.start()
    # ---------- nvidia-smi logger (subprocess)
    sys.path.insert(0, HERE); import pmd2_fast as pf
    gpu_csv = os.path.join(outdir, f"gpu_{stamp}.csv"); smi_p = pf.start_gpu(gpu_csv, 0, a.smi_ms)
    # ---------- (2) oscilloscope process
    scope_p = armed = sstop = q = launch_evt = launch_t = None
    if a.meter in (1, 2):
        window_s = (a.pre_ms + on_ms + a.post_ms) * 1e-3
        cfg = dict(resource=a.resource, no_setup=a.no_setup, v_probe=a.v_probe, v_div=a.v_div, v_center=a.v_center,
                   i_scale=a.i_scale, i_offset=a.i_offset, i_div=a.i_div, i_center=a.i_center,
                   v_ch=a.v_ch, i_ch=a.i_ch, v2_ch=a.v2_ch, no_current=a.no_current,
                   trig_src=(a.i_ch if (a.trig_src == "i" and not a.no_current) else a.v_ch), trig_slope=a.trig_slope,
                   trig_level_volts=(float(a.trig_level) / a.i_scale if (a.trig_src == "i" and not a.no_current) else
                                     (a.trig_level if str(a.trig_level).startswith("auto") else float(a.trig_level))),
                   window_s=window_s, pre_s=a.pre_ms * 1e-3, wait_s=a.wait, sample_s=a.scope_sample_ns * 1e-9)
        print(f"scope window {window_s*1e3:.0f} ms -> {int(window_s/(a.scope_sample_ns*1e-9))} points/channel at {a.scope_sample_ns:g} ns", flush=True)
        armed, sstop, q, launch_evt = ctx.Event(), ctx.Event(), ctx.Queue(), ctx.Event()
        launch_t = ctx.Value("d", 0.0)
        scope_p = ctx.Process(target=scope_worker, args=(a.meter, cfg, armed, sstop, q, log, launch_evt, launch_t), daemon=True); scope_p.start()
    # wait for both meters to be ready before anything is launched
    t_w = time.time()
    while True:
        drain_log(log)
        ok_pmd = (pmd_ready is None) or pmd_ready.is_set() or (pmd_p is not None and not pmd_p.is_alive())
        ok_scope = (armed is None) or armed.is_set()
        if ok_pmd and ok_scope: break
        if scope_p is not None and not scope_p.is_alive():
            drain_log(log); sys.exit("scope process died before arming (see [scope] messages above)")
        if time.time() - t_w > 90: sys.exit("meters not ready within 90 s")
        time.sleep(0.05)
    drain_log(log)
    print("[order] PMD polling -> scope armed -> (freqmon) -> kernel launches", flush=True)
    first_arm_pending = armed is not None and armed.is_set()      # the scope is already armed for burst 0

    # ---------- freqmon start (own stream, in-process; no allocations from here until it stops)
    if fm is not None:
        with torch.cuda.stream(ms):
            fm.mon_start(mbuf, mcnt, mstop, MAXS, IV, a.freqmon_smem_kb * 1024)
        torch.cuda.set_stream(s2)
        time.sleep(0.2)

    time.sleep(a.idle_before)
    marks = []
    for k in range(a.nburst):
        if pmd_ready is not None and pmd_p.is_alive():
            pmd_ready.wait(5)                                   # (1) PMD is sampling
        if armed is not None and not a.free_run:                # (2) scope armed for THIS burst
            t_w = time.time()
            while not armed.is_set():
                drain_log(log)
                if time.time() - t_w > a.wait + 10:
                    print("scope never re-armed; continuing free-run", flush=True); break
                time.sleep(0.005)
            armed.clear()
            time.sleep(0.05)
        if a.prewarm_gap_ms > 0:                                 # warm clock: the same GEMM for >= prewarm_ms (count-bound), the gap, then the measured launch
            t_pw = time.monotonic(); n_pw = 0
            while True:
                mm(); s2.synchronize(); n_pw += 1
                if (time.monotonic() - t_pw) * 1e3 >= a.prewarm_ms: break
            if k == 0: print(f"[warm ] warm-up: {n_pw} launch(es) = {(time.monotonic() - t_pw) * 1e3:.0f} ms, then {a.prewarm_gap_ms:g} ms gap", flush=True)
            time.sleep(a.prewarm_gap_ms / 1e3)
        # (3) stamp + launches  (launch_evt tells the scope process to force the trigger if the edge trigger misses)
        if launch_evt is not None: launch_t.value = time.monotonic(); launch_evt.set()
        if gs is not None: gs.stamp(stamps, 2 * k)
        t_start = time.monotonic(); w_start = time.time()
        for j_ in range(K):
            (mm_first if j_ == 0 else mm)()
        if gs is not None: gs.stamp(stamps, 2 * k + 1)
        s2.synchronize()
        t_end = time.monotonic()
        marks.append(dict(burst=k, t_start=t_start, t_end=t_end, wall_start=w_start, wall_end=time.time(), on_ms=(t_end - t_start) * 1e3))
        print(f"[{t_start % 10000:9.3f}] burst {k:3d}: {(t_end - t_start)*1e3:6.1f} ms (host)", flush=True)
        drain_log(log)
        time.sleep(a.off_ms / 1e3)
    time.sleep(a.idle_after)

    # ---------- stop freqmon, collect stamps
    gt_stamps = None; fx_ns = ff = None
    if fm is not None:
        mstop.fill_(1); ms.synchronize(); s2.synchronize()
        fx_ns, ff, mon_sm = parse_freqmon(mbuf, mcnt, MAXS, IV)
        np.savetxt(os.path.join(outdir, "freqmon.csv"), np.c_[fx_ns, ff], delimiter=",", header="t_gpu_ns,sm_clock_MHz", comments="", fmt=["%.0f", "%.1f"])
        print(f"freqmon: {len(ff)} samples at {a.freqmon_us:.0f} us on SM {mon_sm}", flush=True)
    if gs is not None:
        gt_stamps = stamps.cpu().numpy().tolist()
        for k, m in enumerate(marks):
            m["gt_before_ns"] = gt_stamps[2 * k]; m["gt_after_ns"] = gt_stamps[2 * k + 1]
            m["gpu_ms"] = (gt_stamps[2 * k + 1] - gt_stamps[2 * k]) / 1e6
    json.dump(dict(args=vars(a), K=K, t_one_ms=t_one, launch_latency_ms=lat.tolist(), warmup_launches=nw, on_ms=on_ms, marks=marks,
                   gemm=dict(M=M, N=N_, K=a.size, dtype=a.dtype, kernel=a.kernel), wave=wave_info, ws=ws_info, power_limits=power_limits(),
                   freqmon_us=a.freqmon_us if fm is not None else None), open(os.path.join(outdir, "marks.json"), "w"), indent=1)

    # ---------- collect scope records
    records, stats = [], []
    if scope_p is not None:
        got = 0; t_w = time.time(); seen = {}            # burst -> (drop V, index into records); spurious/duplicate records are dropped
        # budget: the per-record processing (onset on up to 10 M points, csv/npz, 3 figures) can take several seconds at 20 ns,
        # so wait up to 30 s of silence after the last record rather than 30 s in total (that silently dropped records 6-9 once)
        while got < a.nburst and time.time() - t_w < 60:    # 60 s of silence AFTER the previous record was processed (see below)
            drain_log(log)
            try:
                item = q.get(timeout=0.5)
            except Exception:
                continue
            t_w = time.time()
            if item is None: break
            k, t, v, i, meta = item
            v2 = meta.pop("v2", None)
            if a.prewarm_gap_ms > 0:                             # drop the warm-up GEMM from the record: keep from just before the LAST load segment
                keep = crop_to_last_load(t, v, on_ms * 1e-3, a.pre_ms * 1e-3, a.prewarm_gap_ms * 1e-3)
                if keep is not None and keep.sum() > 1000: t, v, i = t[keep], v[keep], i[keep]; v2 = v2[keep] if v2 is not None else None
                meta["prewarm_gap_ms"] = a.prewarm_gap_ms
            p = v * i
            # which burst does this record belong to? the last burst launched before the acquisition completed
            kb = max([m["burst"] for m in marks if m["t_start"] < meta["t_done_mono"]] or [k])
            meta["burst"] = kb; meta["v_ch"] = a.v_ch; meta["v2_ch"] = a.v2_ch; meta["i_ch"] = None if a.no_current else a.i_ch
            meta["v_name"] = (a.v_name or f"CH{a.v_ch}").replace("_", " "); meta["v2_name"] = (a.v2_name or (f"CH{a.v2_ch}" if a.v2_ch else "")).replace("_", " ")
            s0 = scope_onset(t, v); meta["onset_s"] = s0; meta["edge_triggered"] = bool(abs(s0) < 0.3e-3)
            tp = t - s0                                        # plots: 0 = kernel start (voltage onset); csv keeps the trigger frame
            pre_ = t < -0.5e-3; drop = float(np.mean(v[pre_]) - v.min()) if pre_.sum() > 50 else float(v.max() - v.min())
            if marks and meta["t_done_mono"] < marks[0]["t_start"]:
                print(f"[scope] record {k} completed before burst 0 (spurious trigger, e.g. the freqmon start) -> discarded", flush=True); continue
            if kb in seen:
                if drop <= seen[kb][0]:
                    print(f"[scope] record {k}: second record for burst {kb} with a smaller drop ({drop:.3f} V) -> discarded", flush=True); continue
                print(f"[scope] record {k}: replaces the earlier burst {kb} record (drop {drop:.3f} V > {seen[kb][0]:.3f} V)", flush=True)
            trunc = "" if tp[-1] >= on_ms * 1e-3 else f"  WARNING: record ends {tp[-1]*1e3:.1f} ms after the onset, before the {on_ms:.0f} ms load ended"
            how = (f"stop-captured (FRTR at launch + {meta.get('force_cmd_ms') or 0:.1f} ms)" if meta.get("capture") == "stop"
                   else ('edge-triggered' if meta['edge_triggered'] else 'forced trigger'))
            print(f"[scope] burst {kb}: {how}; voltage onset at {s0*1e3:+.2f} ms from the {'nominal launch' if meta.get('capture') == 'stop' else 'trigger'}"
                  f"; record {tp[0]*1e3:+.1f} .. {tp[-1]*1e3:+.1f} ms around the kernel{trunc}", flush=True)
            dec = max(1, int(round(1e-6 / meta["dt"])), int(np.ceil(len(t) / 2e6)))   # csv at >= 1 us and <= 2 M rows (np.savetxt of a 5 M-row record took minutes); npz keeps the full rate
            meta["csv_decimation"] = dec
            f32 = lambda x: None if x is None else x.astype(np.float32)
            if v2 is not None:
                np.savez(os.path.join(outdir, f"burst_{kb:02d}.npz"), t=t, v=f32(v), i=f32(i), p=f32(p), v2=f32(v2), meta=json.dumps(meta))
                np.savetxt(os.path.join(outdir, f"burst_{kb:02d}.csv"), np.c_[t[::dec], v[::dec], i[::dec], p[::dec], v2[::dec]], delimiter=",",
                           header=f"t_s,v_V,i_A,p_W,v2_V", comments="")
            else:
                np.savez(os.path.join(outdir, f"burst_{kb:02d}.npz"), t=t, v=f32(v), i=f32(i), p=f32(p), meta=json.dumps(meta))
                np.savetxt(os.path.join(outdir, f"burst_{kb:02d}.csv"), np.c_[t[::dec], v[::dec], i[::dec], p[::dec]], delimiter=",", header="t_s,v_V,i_A,p_W", comments="")
            dev_name = ('Keysight DSOX1202A' if a.meter == 1 else 'LeCroy HDO4034A') + f" CH{a.v_ch}" + (f"+CH{a.v2_ch}" if a.v2_ch else "") + ("" if a.no_current else f"/CH{a.i_ch}")
            v_lab = f"CH{a.v_ch} {meta['v_name']} (V)"; v2_lab = f"CH{a.v2_ch} {meta['v2_name']} (V)" if v2 is not None else None
            ttl = (f"{stamp}  {'single cycle' if K == 1 else f'burst (K={K} launches)'} {kb}  GEMM M={M} N={N_} K={a.size} {a.dtype}" + (f" {wave_info['waves']:.2f} waves" if wave_info else "") + (f" ws[{a.tag}]" if ws_info else "") + (f"  warm clock ({a.prewarm_ms:g}+ ms GEMM ended {a.prewarm_gap_ms:g} ms before)" if a.prewarm_gap_ms > 0 else "")
                   + f"  measured with {dev_name}  {meta['dt']*1e6:.2f} us/sample  load {on_ms:.0f} ms")
            plot_burst(os.path.join(outdir, f"burst_{kb:02d}.png"), tp, v, i, p, ttl, device=dev_name, dt_us=meta['dt'] * 1e6,
                       v2=v2, v_lab=v_lab, v2_lab=v2_lab, cap_w=power_cap_w(outdir))
            for span_ms in (100, 200):                                          # whole-cycle views: one row, 0 .. 100 ms and 0 .. 200 ms
                plot_burst(os.path.join(outdir, f"burst_{kb:02d}_{span_ms}ms.png"), tp, v, i, p, ttl + f"  [{span_ms} ms window]", device=dev_name, dt_us=meta['dt'] * 1e6,
                           v2=v2, v_lab=v_lab, v2_lab=v2_lab, cap_w=power_cap_w(outdir), windows=[(1e-3, span_ms * 1e-3)])
            if not a.no_current:
                plot_rise(os.path.join(outdir, f"burst_{kb:02d}_rise.png"), tp, i, f"{stamp}  cycle {kb}  M={M} N={N_} K={a.size} {a.dtype}  current rise, {dev_name.split(' CH')[0]}", unit="A")
            # same windows + in-kernel clock: freqmon GPU time -> scope frame via the first-launch stamp placed at the voltage onset
            fq_t = fq_f = None
            if fx_ns is not None and gt_stamps is not None and kb < len(marks) and marks[kb].get("gt_before_ns") is not None:
                rel = (np.asarray(fx_ns) - marks[kb]["gt_before_ns"]) / 1e9
                fsel = (rel > tp[0] - 0.01) & (rel < tp[-1] + 0.01)
                fq_t, fq_f = rel[fsel], np.asarray(ff)[fsel]
            plot_burst_freq(os.path.join(outdir, f"burst_{kb:02d}_freq.png"), tp, v, None if a.no_current else p, fq_t, fq_f, ttl,
                            device=dev_name, dt_us=meta['dt'] * 1e6, v2=v2, v_lab=v_lab, v2_lab=v2_lab, cap_w=power_cap_w(outdir))
            if kb in seen:
                records[seen[kb][1]] = (t, v, i, p); stats[seen[kb][1]] = burst_stats(t, v, i, p, on_ms * 1e-3); seen[kb] = (drop, seen[kb][1])
            else:
                seen[kb] = (drop, len(records)); records.append((t, v.astype(np.float32), i.astype(np.float32), p.astype(np.float32))); stats.append(burst_stats(t, v, i, p, on_ms * 1e-3)); got += 1
            t_w = time.time()      # processing a 10 M-point record (crop, onset, npz, csv, 5 figures) takes > 30 s: without this reset the budget
                                   # expired during record 0 and records 1-2 were silently dropped (2026-11-24, 20 ns cuBLAS reference)
        sstop.set(); drain_log(log); scope_p.join(timeout=10)
        if got < a.nburst:
            print(f"WARNING: scope delivered {got}/{a.nburst} records", flush=True)

    # ---------- stop PMD / nvidia-smi
    if pmd_p is not None:
        pmd_stop.set(); pmd_p.join(timeout=10); drain_log(log)
    if smi_p is not None:
        pf.stop_gpu(smi_p)
    pmd = pmd_summary(pmd_csv, marks) if os.path.exists(pmd_csv) else None
    smi = smi_summary(gpu_csv)
    if pmd:
        for k, rec in enumerate(pmd_burst_records(pmd_csv, marks, pad_s=(min(0.05, max(0.005, a.prewarm_gap_ms / 2e3)) if a.prewarm_gap_ms > 0 else 0.05))):
            if rec is None: continue
            t, v, i, p = rec
            np.savetxt(os.path.join(outdir, f"burst_{k:02d}_pmd.csv"), np.c_[t, v, i, p], delimiter=",",
                       header="t_s,v_V(cable avg),i_A(PCIE1+2),p_W(PCIE1+2)", comments="")
            plot_burst(os.path.join(outdir, f"burst_{k:02d}_pmd.png"), t, v, i, p,
                       f"{stamp}  {'single cycle' if K == 1 else f'burst (K={K} launches)'} {k}  GEMM M={M} N={N_} K={a.size} {a.dtype}  measured with ElmorLabs PMD2X (PCIE1+PCIE2 cables, 1 ms raw)  load {on_ms:.0f} ms",
                       windows=[(1e-3, None)], device="PMD2X", dt_us=1000.0, cap_w=power_cap_w(outdir))

    # ---------- report
    print("\n== PMD2X (1 ms raw, PCIE1+PCIE2) / nvidia-smi ==")
    if pmd:
        print(f"PMD {pmd['rows']} rows ({pmd['rate_hz']:.0f}/s)  idle {pmd['idle_p']:.1f} W  run max {pmd['p_max']:.1f} W")
        print(f"{'burst':>5}{'Pmax W':>8}{'Pmean W':>9}{'V1min':>8}{'V1mean':>8}{'V2min':>8}{'I1max A':>9}")
        for k, s in enumerate(pmd["per_burst"]):
            print(f"{k:5d}" + ("   (no samples)" if s is None else
                  f"{s['p_max']:8.1f}{s['p_mean']:9.1f}{s['v1_min']:8.3f}{s['v1_mean']:8.3f}{s['v2_min']:8.3f}{s['i1_max']:9.2f}"))
    else:
        print("PMD: no data")
    if smi:
        print(f"nvidia-smi: {smi['samples']} samples, power.draw.instant max {smi['p_instant_max']} W, "
              f"power.draw.average max {smi['p_avg_max']} W, SM clock {smi['sm_clock_min']}-{smi['sm_clock_max']} MHz")
    agg = None
    if records:
        print(f"\n== oscilloscope ({'DSOX1202A' if a.meter == 1 else 'HDO4034A'}, 1 us/sample) ==")
        print(f"{'burst':>5}{'Vidle':>7}{'Vmin':>7}{'Vdroop':>7}{'Vsettled':>9}{'Vmin@us':>9}" + ("" if a.no_current else f"{'Ppeak':>7}{'P_over':>7}{'P_under':>8}{'Ipeak':>6}"))
        rows = []
        for s in stats:
            if s is None: continue
            rows.append(s)
            line = f"{len(rows)-1:5d}{s['v']['idle']:7.3f}{s['v']['min_on']:7.3f}{s['v']['droop_vs_idle']:7.3f}{s['v']['settled']:9.3f}{s['v']['t_min_on_us']:9.0f}"
            if s["p"] is not None:
                line += f"{s['p']['max_on']:7.1f}{s['p']['overshoot']:7.1f}{s['p']['undershoot']:8.1f}{s['i']['max_on']:6.1f}"
            print(line)
        if rows:
            vd = np.array([r["v"]["droop_vs_idle"] for r in rows]); vm = np.array([r["v"]["min_on"] for r in rows])
            agg = dict(v_droop=dict(mean=float(vd.mean()), max=float(vd.max()), min=float(vd.min())), v_min=float(vm.min()))
            print(f"\nover {len(rows)} bursts: V droop mean {vd.mean():.3f} V, max {vd.max():.3f} V, V min {vm.min():.3f} V")
        make_plot(outdir, records, on_ms, a.meter)
    try:
        clk = subprocess.run([pf.find_nvidia_smi() or "nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,power.limit", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        clk = None
    json.dump(dict(stamp=stamp, args=vars(a), K=K, on_ms=on_ms, marks=marks, pmd=pmd, nvidia_smi=smi, clocks_at_end=clk,
                   scope_per_burst=stats, scope_aggregate=agg), open(os.path.join(outdir, "summary.json"), "w"), indent=1)
    print("\nsaved ->", outdir)
    if not a.no_plots:
        subprocess.run([sys.executable, os.path.join(HERE, "plot_spike_csv.py"), outdir])


if __name__ == "__main__":
    main()
