# AGENTS.md — edge_wake (main SIH firmware project)

> Read this first if you are an agent (or human) picking up work here.
> Companion repo: `edge_wake_dataset` (field recorder firmware + take
> scripts). This repo is the actual SIH prototype firmware.

## Goal

SIH problem: **low-latency, efficient voice activator for edge devices**.
Custom wake word **"Jago Guru"** detected on-device (ESP32-S3 + INMP441),
then subsequent audio streamed to an open-source ASR server. Hard targets:
**RAM < 256 KB internal, idle CPU < 10%, keyword-end → server latency minimal,
near-zero false activations, open-source only** (no proprietary wake SDKs).

## Hardware (verified on bench)

- ESP32-S3, 16 MB flash + 8 MB octal PSRAM (module code N16R8; NOT "N18R8"),
  rev v0.2, MAC `28:84:85:50:72:70`, @240 MHz (`cpu freq` confirmed on boot).
- INMP441: VDD→3V3, GND→GND, L/R→GND, SD→GPIO10, WS→GPIO11, SCK→GPIO12.
- Two USB personalities matter: **UART bridge** (`QinHeng`, bidirectional,
  reliable) vs **native USB Serial/JTAG** (`303a:1001`, reads fine, **host
  writes stall forever** on this bench — triggering must be read-only there).
- Mic gain: 24-bit-in-32-bit LEFT slot >> 12 with saturation (~+24 dB).

## Layout

```
main/mic_capture.c   orchestrator: I2S, ring, VAD, VU, buttons/serial cmds, dumps
main/frontend.c/.h   log-mel frontend (frozen params below), esp-dsp FFT
main/CMakeLists.txt  + PRIV_REQUIRES espressif__esp-dsp
main/idf_component.yml  pins esp-dsp (managed_components/ is gitignored)
scripts/capture.py       30 s dump -> sample.wav, live phase narration
scripts/frontend_check.py  S3 mel vs numpy bit-match checker (PASS bar < 0.05)
sdkconfig.defaults    source of truth (target/PSRAM/console/CPU/stack/DSP size)
sample.wav            reference capture (committed on purpose, used by checks)
```

Sibling `edge_wake_dataset/` repo: separate field-recorder firmware (native
USB, boot auto-take, BOOT takes, CRC) + `capture_takes.py`. Dataset takes
live there / with the dataset, NOT in this repo.

## Frozen DSP params (change in BOTH frontend.h AND frontend_check.py, re-verify)

SR 16000 / FRAME 512 (32 ms) / HOP 256 (16 ms, 62.5 fps) / NFFT 512 /
40 mels / 20–4000 Hz / pre-emph 0.97 / symmetric Hamming / HTK mel with
floor() bins, peak-1 triangles / power |X|²/N bins 0–256 / natural
log(max(e,1e-10)). Frontend is **stateless**: prev sample passed explicitly
per call (see lesson 3).

## Status (git log is truth; commits `6db1b7c..b2dfbed` + `f121fe1`)

- [x] Phase 0 — mic baseline: 16 kHz I2S, 30 s PSRAM capture, serial WAV dump.
- [x] Phase 1 — continuous loop: 2 s ring (self-check OK), energy VAD
      (enter 6000/exit 3000, calibrated to traffic-background room, 300 ms
      hangover), VU telemetry, BOOT 30 s capture preserved.
- [x] Phase 2 — frontend bit-matched to numpy (max 6.6e-05, PASS), aes3 and
      ANSI kernels proven equivalent; ships aes3 at **1.04 ms/frame**.
- [~] Phase 3 — dataset collection IN PROGRESS ("Jago Guru" positives via
      dataset repo; negatives = Speech Commands + MUSAN; clip tool 3a pending).
- [ ] Phase 4 — train DS-CNN (transfer-learn ML-zoo, DET/FA-hr eval).
- [ ] Phase 5 — int8 quant + `.tflite` embed.  [ ] Phase 6 — on-device KWS.
- [ ] Phase 7 — pre-roll + WebSocket stream.  [ ] Phase 8 — faster-whisper
      server (laptop has GTX 2050 4 GB + i5/8 GB; use CUDA).
- [ ] Phase 9/10 — UI/state machine, SIH scorecard.

## Commands (IDF v6.1, `source ~/.espressif/tools/activate_idf_v6.1.sh` first)

```bash
idf.py set-target esp32s3 && idf.py build && idf.py -p /dev/ttyACM0 flash
idf.py -p /dev/ttyACM0 monitor        # buzzy at 921600 console baud
python scripts/capture.py --port /dev/ttyACM0 --baud 921600 --timeout 150
python scripts/frontend_check.py --port /dev/ttyACM0 --baud 921600
```
Serial cmds (UART port only; native-USB writes stall): `f` = 1 s mel dump,
`r` = 30 s capture. BOOT: tap = mel dump, hold 1.5 s = 30 s capture.

## Hard lessons (do not relearn)

1. **`sdkconfig.defaults` edits do NOT apply over an existing `sdkconfig`.**
   After any defaults change: `rm -f sdkconfig && idf.py set-target esp32s3`
   then rebuild. Verify with `grep` on `sdkconfig`.
2. **Main-task stack is 3584 B by default** → set to 8192 (done). Big arrays
   (`hop`, FFT scratch `s_x`) must be `static`, never stack. Stack overflow
   detector fires clearly — trust it.
3. **Pre-emphasis must be stateless across overlapped frames.** A running tail
   advances 2× too fast over 50%-overlap replays AND would poison production
   inference the same way (found via quiet-bin ±2 log-unit mismatch; loud bins
   matched and hid it). Training (torchaudio, whole-utterance) uses immediate
   predecessors — inference must too.
4. **Long printf bursts starve IDLE → task_wdt text lands mid-dump.** Feed/yield
   in long loops; keep CRC markers in every dump; parser must tolerate +
   report bad lines (never silently drop tails). Dataset repo disables IDLE
   WDT checks; THIS repo keeps them.
5. **Every dump needs integrity proof**: `%a` exact floats for features,
   CRC32 for PCM, and the host must verify before trusting a WAV.
6. **Commit policy**: working milestones as `Phase N: ...` commits; push only
   when the user asks. `build/`, `sdkconfig`, `managed_components/`,
   `takes/`, `*.log` stay untracked (see `.gitignore`).
