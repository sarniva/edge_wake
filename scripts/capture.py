#!/usr/bin/env python3
"""Capture the AUD_DUMP_* hex dump from the ESP32 serial log and save sample.wav.

Two modes:
  1) Live capture:  python3 scripts/capture.py --port /dev/ttyACM0 --baud 921600
     (resets the board via DTR, records one dump, writes sample.wav)
  2) Parse a saved monitor log:  python3 scripts/capture.py --log monitor.txt

Requires: pyserial, numpy (numpy optional but recommended for stats).
"""
import argparse, sys, time, struct, wave

def parse_lines(lines):
    samples = []
    in_dump = False
    sr = 16000
    for ln in lines:
        s = ln.strip()
        if s.startswith("AUD_DUMP_START"):
            in_dump = True
            samples = []
            for tok in s.split():
                if tok.startswith("sr="):
                    try: sr = int(tok[3:])
                    except ValueError: pass
            continue
        if s.startswith("AUD_DUMP_END"):
            in_dump = False
            break
        if in_dump and "AUD_DATA:" in s:
            hexpart = s.split("AUD_DATA:", 1)[1].strip()
            # hexpart is little-endian uint16 words as %04x of the int16 bits
            for i in range(0, len(hexpart) - 3, 4):
                try:
                    w = int(hexpart[i:i+4], 16)
                except ValueError:
                    break
                if w >= 0x8000: w -= 0x10000
                samples.append(w)
    return samples, sr

def stats(samples):
    import math
    n = len(samples)
    if n == 0: return "no samples"
    peak = max(abs(v) for v in samples)
    rms = math.sqrt(sum(v*v for v in samples) / n)
    dc = sum(samples) / n
    db = -96.0 if rms < 0.5 else 20*math.log10(rms/32768.0)
    return f"n={n} dc={dc:+.1f} rms={rms:.1f} peak={peak} ({100*peak/32768:.1f}% FS) {db:.1f} dBFS"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--log", default=None, help="parse saved monitor log instead of live capture")
    ap.add_argument("--out", default="sample.wav")
    ap.add_argument("--timeout", type=float, default=150.0)
    a = ap.parse_args()

    if a.log:
        with open(a.log, errors="ignore") as f:
            lines = f.read().splitlines()
        # monitor logs prefix lines with timestamps; AUD markers survive
        samples, sr = parse_lines(lines)
    else:
        try:
            import serial
        except ImportError:
            sys.exit("need pyserial: pip install pyserial")
        ser = serial.Serial(a.port, a.baud, timeout=1)
        ser.dtr = False; time.sleep(0.1); ser.dtr = True  # reset board
        t_reset = time.time()
        el = lambda: time.time() - t_reset
        print(f"board reset at +0.0s, listening on {a.port} @ {a.baud} ...")
        print("watch the phases below: only audio between REC_START and REC_END")
        print("ends up in the .wav (VU lines before/after are live, not recorded).")
        buf, linebuf = "", ""
        t0 = time.time()
        samples, sr = [], 16000
        phase, rec_start_t, last_tick = "boot", None, t0
        dump_lines, expected_dump_lines = 0, None
        while time.time() - t0 < a.timeout:
            chunk = ser.read(4096).decode("ascii", "ignore")
            if not chunk:
                if phase == "recording" and time.time() - last_tick > 5:
                    last_tick = time.time()
                    remain = max(0.0, 30 - (time.time() - rec_start_t))
                    print(f"  [+{el():5.1f}s] ... recording, ~{remain:.0f} s left -- SPEAK NOW")
                continue
            buf += chunk
            linebuf += chunk
            while "\n" in linebuf:
                line, linebuf = linebuf.split("\n", 1)
                s = line.strip()
                if "I2S RX ready" in s and phase == "boot":
                    phase = "arming"
                    print(f"  [+{el():5.1f}s] mic clocking started (I2S live), 2 s pause...")
                elif s.startswith("REC_START") and phase in ("boot", "arming"):
                    phase = "recording"
                    rec_start_t, last_tick = time.time(), time.time()
                    print(f"  [+{el():5.1f}s] REC_START -> RECORDING 30 s -- SPEAK NOW")
                elif s.startswith("REC_END") and phase == "recording":
                    phase = "dumping"
                    print(f"  [+{el():5.1f}s] REC_END -> mic window closed, dumping buffer...")
                elif s.startswith("AUD_DUMP_START") and phase == "dumping":
                    for tok in s.split():
                        if tok.startswith("samples="):
                            try:
                                expected_dump_lines = (int(tok[8:]) + 255) // 256
                            except ValueError:
                                pass
                    print(f"  [+{el():5.1f}s] dump started "
                          f"({expected_dump_lines or '?'} lines, replaying buffer, mic ignored)...")
                elif "AUD_DATA:" in s and phase == "dumping":
                    dump_lines += 1
                    step = max(1, (expected_dump_lines or 200) // 10)
                    if dump_lines % step == 0:
                        if expected_dump_lines:
                            print(f"  [+{el():5.1f}s] ... dump "
                                  f"{100 * dump_lines / expected_dump_lines:.0f}%")
                        else:
                            print(f"  [+{el():5.1f}s] ... dump line {dump_lines}")
                elif s.startswith("AUD_DUMP_END"):
                    print(f"  [+{el():5.1f}s] dump complete.")
                    phase = "done"
                    break
            if phase == "done":
                break
        print()
        samples, sr = parse_lines(buf.splitlines())
        ser.close()

    print("parsed:", stats(samples))
    if not samples:
        sys.exit("No AUD dump found. Check wiring/power, reboot, and retry. "
                 "Tip: 'tap the mic' should move the VU lines in the log.")
    with wave.open(a.out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(struct.pack("<%dh" % len(samples), *samples))
    print(f"wrote {a.out} ({len(samples)/sr:.2f} s @ {sr} Hz mono 16-bit)")
    print("Listen:  ffplay/audacity/open", a.out)
    print("Good = clear speech, no flat line, no constant full-scale buzz.")
    if max(abs(v) for v in samples) < 200:
        print("WARNING: very quiet -> mic may be muted/unpowered or L/R floating. "
              "Tie L/R to GND, confirm 3V3, speak within 30 cm.")

if __name__ == "__main__":
    main()
