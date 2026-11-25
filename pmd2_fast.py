"""
pmd2_fast.py — PMD2X fast polling (down to 1 ms) + simultaneous GPU logging.

Polls the device as fast as possible (target interval configurable, default 1 ms)
and logs ALL channels in the exact layout the official PMD2 software uses
(16 power channels + device-computed aggregates + 2 temperatures), to a CSV.
Optionally starts nvidia-smi at the same time so the two logs share a timeline.

Field layout verified against decompiled PMD2X_Device (192-byte SensorStruct):
    Vdd u16 | Tchip i16 | Ts[2] i16 | PowerReadings[16] (u16 V, u32 I, u32 P) |
    EpsPower u16 | PciePower u16 | MbPower u16 | TotalPower u16 | Ocp[16] u8

NOTE on rate: the device updates the 0x04 sensor packet at ~10 Hz regardless of
poll speed (confirmed against the official software's own 5 ms logs — same
duplicate-value behaviour). Polling at 1 ms therefore over-samples: you capture
every device update with no latency, at the cost of duplicate rows. Use
--dedup to keep only rows where a value actually changed.

Usage:
    sudo ./venv/bin/python pmd2_fast.py                      # 1 ms poll, PMD2 only
    sudo ./venv/bin/python pmd2_fast.py --gpu                # + nvidia-smi
    sudo ./venv/bin/python pmd2_fast.py --interval 1 --gpu   # explicit 1 ms
    sudo ./venv/bin/python pmd2_fast.py --gpu --dedup        # drop duplicate rows
    sudo ./venv/bin/python pmd2_fast.py --duration 20 --gpu  # stop after 20 s
"""
import argparse
import datetime
import os
import shutil
import subprocess
import sys
import time
from ctypes import (Structure, c_uint8, c_uint16, c_int16, c_uint32, sizeof)

import serial
import serial.tools.list_ports

PRODUCT_STRING = "ElmorLabs PMD2"
USB_VID_PID = "USB VID:PID=0483:5740"
BAUD = 115200
PKT = 192
CMD_WELCOME = 0x00
CMD_READ_SENSOR_VALUES = 0x04

# 16 power channels, official labels (from PMD2X_Device constructor).
RAIL_NAMES = [
    "ATX12V", "ATX5V", "ATX5VSB", "ATX3V3", "HPWR", "EPS1", "EPS2",
    "PCIE1", "PCIE2", "PCIE3",
    "HPWR_W1", "HPWR_W2", "HPWR_W3", "HPWR_W4", "HPWR_W5", "HPWR_W6",
]

GPU_QUERY = ("timestamp,clocks.current.sm,clocks.current.memory,"
             "power.draw.instant,power.draw.average,power.limit,"
             "temperature.gpu,utilization.gpu,pcie.link.gen.current,pcie.link.width.current")


class PowerSensor(Structure):
    _pack_ = 1
    _fields_ = [("Voltage", c_uint16), ("Current", c_uint32), ("Power", c_uint32)]


class SensorStruct(Structure):
    _pack_ = 1
    _fields_ = [
        ("Vdd", c_uint16),
        ("Tchip", c_int16),
        ("Ts", c_int16 * 2),
        ("PowerReadings", PowerSensor * 16),
        ("EpsPower", c_uint16),
        ("PciePower", c_uint16),
        ("MbPower", c_uint16),
        ("TotalPower", c_uint16),
        ("Ocp", c_uint8 * 16),
    ]


assert sizeof(SensorStruct) == PKT, sizeof(SensorStruct)


# ----------------------------------------------------------------------------
# Device
# ----------------------------------------------------------------------------
def find_port():
    for port, desc, hwid in sorted(serial.tools.list_ports.comports()):
        if USB_VID_PID in hwid:
            return port
    return None


def open_device(port):
    ser = serial.Serial(port, BAUD, timeout=0.2, rtscts=False, dsrdtr=False)
    ser.dtr = True
    ser.reset_input_buffer()
    ser.write(bytes([CMD_WELCOME])); ser.flush(); time.sleep(0.1)
    wel = ser.read(15)
    if not wel.startswith(PRODUCT_STRING.encode("ascii")):
        raise RuntimeError(f"bad welcome: {wel!r}")
    ser.reset_input_buffer()
    return ser


def read_packet(ser):
    ser.write(bytes([CMD_READ_SENSOR_VALUES]))
    buf = ser.read(PKT)
    if len(buf) != PKT:
        ser.reset_input_buffer()
        return None
    return buf


# ----------------------------------------------------------------------------
# GPU logger (absolute path so it works under sudo)
# ----------------------------------------------------------------------------
def find_nvidia_smi():
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for c in ("/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi"):
        if os.path.exists(c):
            return c
    return None


def start_gpu(gpu_csv, gpu_index, interval_ms):
    smi = find_nvidia_smi()
    if not smi:
        print("nvidia-smi not found — GPU logging skipped.", file=sys.stderr)
        return None
    base = [smi, "-i", str(gpu_index), f"--query-gpu={GPU_QUERY}",
            "--format=csv", "-lms", str(interval_ms)]
    f = open(gpu_csv, "w")
    for cmd in (["stdbuf", "-oL"] + base, base):
        try:
            # own session/process group: `stdbuf` forks nvidia-smi as a grandchild, so we must be
            # able to signal the whole group, not just the wrapper pid (else nvidia-smi is orphaned
            # and keeps polling the GPU every 100 ms, polluting later measurements).
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.DEVNULL, text=True, start_new_session=True)
            p._f = f
            return p
        except FileNotFoundError:
            continue
    f.close()
    return None


def stop_gpu(p):
    if not p:
        return
    import signal
    try:
        os.killpg(p.pid, signal.SIGINT); p.wait(timeout=3)
    except Exception:
        try: os.killpg(p.pid, signal.SIGKILL); p.wait(timeout=3)
        except Exception: pass
    if getattr(p, "_f", None):
        p._f.close()


def _install_term_handler():
    """Make SIGTERM behave like Ctrl-C so the `finally` block (csv close + stop_gpu) always runs.
    Without this, a parent calling terminate() orphans the nvidia-smi child, which keeps polling
    the GPU every 100 ms and pollutes later measurements."""
    import signal
    def _h(signum, frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _h)
    signal.signal(signal.SIGHUP, _h)


# ----------------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------------
def csv_header():
    cols = ["timestamp", "vdd_V", "tchip_C", "ts0_C", "ts1_C",
            "TotalPower_W", "EpsPower_W", "PciePower_W", "MbPower_W"]
    for name in RAIL_NAMES:
        cols += [f"{name}_V", f"{name}_A", f"{name}_W"]
    cols.append("t_mono")      # time.monotonic(): immune to the WSL wall-clock steps (align by this)
    return ",".join(cols)


def csv_row(buf, t_epoch, t_mono=None):
    s = SensorStruct.from_buffer_copy(buf)
    vals = [f"{t_epoch:.6f}",
            f"{s.Vdd/1000:.3f}", f"{s.Tchip/10:.1f}",
            f"{s.Ts[0]/10:.1f}", f"{s.Ts[1]/10:.1f}",
            str(s.TotalPower), str(s.EpsPower), str(s.PciePower), str(s.MbPower)]
    for i in range(16):
        p = s.PowerReadings[i]
        vals += [f"{p.Voltage/1000:.3f}", f"{p.Current/1000:.3f}", f"{p.Power/1000:.3f}"]
    vals.append(f"{time.monotonic() if t_mono is None else t_mono:.6f}")
    return ",".join(vals)


def value_key(buf):
    # bytes 8..168 = the 16 PowerReadings; ignore Vdd/Tchip/Ts noise for dedup
    return buf[8:168]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    _install_term_handler()
    ap = argparse.ArgumentParser(description="PMD2X fast poll + GPU logging")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="target poll interval in ms (default 1.0)")
    ap.add_argument("--duration", type=float, default=0,
                    help="seconds to run (0 = until Ctrl-C)")
    ap.add_argument("--gpu", action="store_true", help="also log nvidia-smi")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--gpu-interval", type=int, default=100,
                    help="nvidia-smi interval in ms (default 100)")
    ap.add_argument("--dedup", action="store_true",
                    help="only write rows where a power reading changed")
    ap.add_argument("--csv", help="PMD2 CSV path (default pmd2_<stamp>.csv)")
    ap.add_argument("--port")
    args = ap.parse_args()

    port = args.port or find_port()
    if not port:
        print(f"No PMD2 found ({USB_VID_PID}). Attach via usbipd?", file=sys.stderr)
        sys.exit(1)
    try:
        ser = open_device(port)
    except Exception as e:
        print(f"open failed: {e}", file=sys.stderr); sys.exit(1)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pmd2_csv = args.csv or f"pmd2_{stamp}.csv"
    gpu_csv = f"gpu_{stamp}.csv" if args.gpu else None

    f = open(pmd2_csv, "w")
    f.write(csv_header() + "\n")
    print(f"Logging PMD2 -> {pmd2_csv}  (target {args.interval} ms poll"
          f"{', dedup' if args.dedup else ''})")

    gpu_proc = start_gpu(gpu_csv, args.gpu_index, args.gpu_interval) if args.gpu else None
    if gpu_proc:
        print(f"Logging GPU  -> {gpu_csv}  (nvidia-smi every {args.gpu_interval} ms)")
    print("Ctrl-C to stop.\n")

    target = args.interval / 1000.0
    t0 = time.perf_counter()
    nxt = t0
    polls = written = 0
    last_key = None
    report = t0
    try:
        while True:
            now_perf = time.perf_counter()
            buf = read_packet(ser)
            t_epoch = time.time(); t_mono = time.monotonic()
            if buf is not None:
                polls += 1
                if args.dedup:
                    k = value_key(buf)
                    if k != last_key:
                        f.write(csv_row(buf, t_epoch, t_mono) + "\n")
                        written += 1
                        last_key = k
                else:
                    f.write(csv_row(buf, t_epoch, t_mono) + "\n")
                    written += 1
            if now_perf - report >= 2.0:
                el = now_perf - t0
                print(f"  {el:6.1f}s  polls={polls} ({polls/el:,.0f}/s)  "
                      f"rows={written}", end="\r")
                report = now_perf
            if args.duration and (now_perf - t0) >= args.duration:
                break
            nxt += target
            sl = nxt - time.perf_counter()
            if sl > 0:
                time.sleep(sl)
            else:
                nxt = time.perf_counter()
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        dt = time.perf_counter() - t0
        stop_gpu(gpu_proc)
        f.close()
        ser.close()
        print(f"\nDone. {polls} polls in {dt:.1f}s ({polls/dt:,.0f}/s), "
              f"{written} rows -> {pmd2_csv}")
        if gpu_csv:
            print(f"GPU -> {gpu_csv}")
            print(f"\nCompare:  ./venv/bin/python compare.py {pmd2_csv} {gpu_csv} "
                  f"--rails PCIE1,PCIE2,PCIE3,HPWR --out merged_{stamp}.csv")


if __name__ == "__main__":
    main()
