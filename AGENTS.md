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
scripts/frontend_check.py  S3 mel vs numpy bit-match checker (PASS bar < 0.05)
sdkconfig.defaults    source of truth (target/console/CPU/stack/DSP size; NO PSRAM use)
sample.wav            legacy reference capture (kept as test audio; unused by checks)
```

Sibling `edge_wake_dataset/` repo: separate field-recorder firmware (native
USB, boot auto-take, BOOT takes, CRC) + `capture_takes.py`. Dataset takes
live there / with the dataset, NOT in this repo. The 30 s PSRAM
record-and-dump path was REMOVED from this firmware (it lives only in the
dataset repo now); this firmware uses internal RAM only, no PSRAM.

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
- [x] Cleanup — 30 s PSRAM record-and-dump path removed from this firmware
      (`do_capture`, `dump_hex`, `rec_buf`, AUD_* markers, `scripts/capture.py`
      all deleted); BOOT = mel dump only, serial cmds `f`/`p` only.
- [~] Phase 3 — dataset collection IN PROGRESS (9 speakers × 3 conditions,
      ~920 "Jago Guru" utterances as 69×30 s takes; clip review ongoing via
      dataset repo `clip_cutter.py`; negatives = Speech Commands + MUSAN).
- [~] Phase 4 — Kaggle: GSC-35 pretrain hit 90.4% val (DS-CNN-S, 25.8K params);
      linear-probe fine-tune stalled (val ~0.82, DET badly missed bar:
      FRR 19% @ FAR 11% at thr 0.5). Notebook v2: stronger head, full
      unfreeze, wake augmentation (noise/shift/specaug), Keras 3 export fix.
  NOTE: Arm ML-zoo is ARCHIVED (read-only since Jul 2025) — still usable
  Apache-2.0, but pin the checkpoint commit; fallback = Keras rebuild from
  Hello Edge table, reserve = micro-wake-word trainer. No local GPU here;
  training runs on Kaggle. Feature extraction for training reuses
  `ref_logmel` (vendored into dataset repo `scripts/frontend_check.py`;
  keep copies in sync, re-run bit-match checker if params change).
  UPDATE run-2: v2 fine-tune reached val 88%, test recall 87-89% all classes,
  DET improved (FRR 11%@FAR 5.5% @thr0.5) but still misses bar. Notebook v3
  adds temperature scaling + full-int8 PTQ cell. Artifacts to pull down:
  jagoguru_int8.tflite, norm.npz, det_scaled.txt, threshold choice.
  UPDATE run-3: T=1.5 halves FAR (3.8%@thr0.5, 1.4%@thr0.7) but FRR stuck
  at 11% - model under-confident on true wake. §8 eval had batch-dim bug
  (export froze batch=1; fixed via resize_tensor_input). Next: hard-negative
  mining from high-firing GSC words, then focal loss if needed.
  UPDATE artifacts pulled: models/jagoguru_int8.tflite (51.7 KB, int8 61x40x1
  in / 1x3 out) + norm.npz (z-norm mu/sd) in dataset repo. Local int8 eval
  reproduces Kaggle exactly (0.8692). Local int8 DET (raw): FRR 10.6%@FAR
  6.7% @thr0.5; FRR 13.6%@FAR 3.3% @thr0.7. Start firmware threshold 0.7.
  NOTE: det_scaled.txt on disk differs from pasted run-3 table (different
  Kaggle run) - file is canonical for now; variance itself flags the
  single-speaker (sp09) test set as noisy. Still misses SIH bar.
- [ ] Phase 4 — train DS-CNN (transfer-learn ML-zoo, DET/FA-hr eval).
- [ ] Phase 5 — int8 quant + `.tflite` embed.  [ ] Phase 6 — on-device KWS.
- [ ] Phase 7 — pre-roll + WebSocket stream.  [ ] Phase 8 — faster-whisper
      server (laptop has GTX 2050 4 GB + i5/8 GB; use CUDA).
- [ ] Phase 9/10 — UI/state machine, SIH scorecard.

## Commands (IDF v6.1, `source ~/.espressif/tools/activate_idf_v6.1.sh` first)

```bash
idf.py set-target esp32s3 && idf.py build && idf.py -p /dev/ttyACM0 flash
idf.py -p /dev/ttyACM0 monitor        # buzzy at 921600 console baud
python scripts/frontend_check.py --port /dev/ttyACM0 --baud 921600
```
Serial cmds (UART port only; native-USB writes stall): `f` = 1 s mel dump,
`p` = power-spectrum debug dump. BOOT: tap = mel dump.

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
