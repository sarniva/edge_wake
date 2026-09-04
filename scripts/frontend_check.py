#!/usr/bin/env python3
"""Phase 2 verifier: S3 log-mel frontend vs bit-identical numpy reference.

Frozen params (MUST match main/frontend.h):
  SR=16000 FRAME=512 HOP=256 NFFT=512 NMELS=40 FMIN=20 FMAX=4000
  PREEMPH=0.97, symmetric Hamming, HTK mel, floor() bins, peak=1 triangles,
  power=|X|^2/N bins 0..256, natural log(max(e,1e-10)).

Live:   python3 scripts/frontend_check.py --port /dev/ttyACM0 --baud 921600
  (resets board, sends 'f' -> 1 s PCM + 61x40 mel dump, compares)
Log:    python3 scripts/frontend_check.py --log some.log

Pass bar: max abs diff < 0.05 on the log-mel scale (float32-vs-float64
 arthimetic alone accounts for ~1e-6; anything bigger is a real mismatch).
"""
import argparse, struct, sys, time
import numpy as np

SR, FRAME, HOP, NFFT, NMELS = 16000, 512, 256, 512, 40
FMIN, FMAX, PREEMPH, FLOOR = 20.0, 4000.0, 0.97, 1e-10

def ref_logmel(pcm):
    x = np.asarray(pcm, dtype=np.float64) / 32768.0
    y = np.empty_like(x)
    prev = 0.0
    for i, s in enumerate(x):       # stateful pre-emphasis, prev=0 at start
        y[i] = s - PREEMPH * prev
        prev = s
    n = np.arange(FRAME)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (FRAME - 1))
    mlo = 2595 * np.log10(1 + FMIN / 700.0)
    mhi = 2595 * np.log10(1 + FMAX / 700.0)
    pts = np.floor((NFFT + 1) *
                   (700 * (10 ** (np.linspace(mlo, mhi, NMELS + 2) / 2595.0) - 1))
                   / SR).astype(int)
    bank = np.zeros((NMELS, NFFT // 2 + 1))
    for m in range(NMELS):
        lo, ce, hi = pts[m], pts[m + 1], pts[m + 2]
        if ce <= lo or hi <= ce:
            continue
        for k in range(lo, ce):
            if 0 <= k < bank.shape[1]:
                bank[m, k] = (k - lo) / (ce - lo)
        for k in range(ce, hi):
            if 0 <= k < bank.shape[1]:
                bank[m, k] = (hi - k) / (hi - ce)
    out = []
    for start in range(0, len(y) - FRAME + 1, HOP):
        spec = np.fft.rfft(y[start:start + FRAME] * window, NFFT)
        power = (spec.real ** 2 + spec.imag ** 2) / NFFT
        e = bank @ power
        out.append(np.log(np.maximum(e, FLOOR)))
    return np.array(out)

def parse_dump(lines):
    pcm, mel, mode = [], [], None
    for ln in lines:
        s = ln.strip()
        if s.startswith("FE_PCM_START"):
            mode = "pcm"; continue
        if s.startswith("FE_PCM_END"):
            mode = None; continue
        if s.startswith("FE_MEL_START"):
            mode = "mel"; continue
        if s.startswith("FE_MEL_END"):
            mode = None; continue
        if mode == "pcm" and "FE_PCM:" in s:
            hx = s.split("FE_PCM:", 1)[1].strip()
            for i in range(0, len(hx) - 3, 4):
                try:
                    w = int(hx[i:i + 4], 16)
                except ValueError:
                    break
                pcm.append(w - 0x10000 if w >= 0x8000 else w)
        elif mode == "mel" and s.startswith("FE_MEL:"):
            try:
                mel.append([float.fromhex(t) for t in s[7:].split()])
            except ValueError:
                pass
    return pcm, (np.array(mel) if mel else np.zeros((0, NMELS)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--log", default=None)
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()

    if a.log:
        with open(a.log, errors="ignore") as f:
            lines = f.read().splitlines()
    else:
        import serial
        ser = serial.Serial(a.port, a.baud, timeout=1)
        ser.dtr = False; time.sleep(0.1); ser.dtr = True
        t0, buf = time.time(), ""
        time.sleep(2.5)                      # let boot finish (no auto-dump now)
        ser.write(b"f")                      # trigger 1 s frontend dump
        print("sent 'f', waiting for FE dump ... (speak any speech)")
        while time.time() - t0 < a.timeout:
            buf += ser.read(8192).decode("ascii", "ignore")
            if "FE_MEL_END" in buf and "FE_MEL_START" in buf:
                break
        ser.close()
        lines = buf.splitlines()

    pcm, s3 = parse_dump(lines)
    print(f"parsed: pcm={len(pcm)} samples, s3mel={s3.shape}")
    if len(pcm) < FRAME or s3.shape[0] == 0:
        sys.exit("No complete FE dump found (need FE_PCM + FE_MEL blocks).")
    ref = ref_logmel(pcm)
    n = min(len(ref), len(s3))
    diff = np.abs(ref[:n] - s3[:n])
    print(f"frames compared: {n} x {NMELS}")
    print(f"mean abs diff: {diff.mean():.2e}   max abs diff: {diff.max():.2e}")
    fi, mi = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"worst at frame {fi} mel {mi}: ref={ref[fi,mi]:.6f} s3={s3[fi,mi]:.6f}")
    print(f"ref dynamic range: [{ref[:n].min():.2f}, {ref[:n].max():.2f}]")
    if diff.max() < 0.05:
        print("PASS: S3 frontend matches numpy reference.")
    else:
        sys.exit("FAIL: mismatch beyond float32 tolerance - check window/mel/power math.")

if __name__ == "__main__":
    main()
