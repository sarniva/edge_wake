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
  UPDATE run-A vs run-B: user keeps both runs (_1 suffix = earlier). Run A
  wins clearly - int8 acc 0.8917 vs 0.8692, DET raw thr0.7 FRR 12.9%@FAR
  2.4% (thr0.8: 13.6%/1.7%). models/ now holds run-A artifacts (verified).
  Lesson: identical-config retrains swing ~2 pts on this single-speaker
  test set - never trust one table; on-device FA/hr is the real metric.
  UPDATE v4 (hard-neg mining) REGRESSED - do NOT ship: int8 0.8728, DET
  thr0.5 FRR 13.6%/FAR 5.0%. Local model-vs-model: v4 misses every clip
  run-A misses (15/15 shared) +1 more - mining fixed zero wake misses.
  Forensics: missed clips are systematically QUIET (49% bins <-14dB vs 5%
  for hits). Fix = gain augmentation (0.3-1.5x), NOT more negatives.
  v5 recipe: warm-start run-A weights if downloadable (else GSC stem),
  gain aug, gentler mining (top500 x2), same schedule.
  UPDATE v4 on-device verdict: user-tested WORSE than run-A -> reverted on
  master (2cc2416), v4 commit preserved on branch model-v4 (note: no space,
  git-unfriendly otherwise). Run-A firmware reflashed + verified alive.
  v5 prompt saved at dataset repo V5_PROMPT.md (paste into fresh session).
  UPDATE v5/v6f/v7 (other agent, Kaggle): all fail to beat run-A on the real
  job. v5 DET worse both axes. v7 (focal path) breaks calibration
  (FRR 60%@thr0.7) and adds 6 NEW misses on previously-good clips - REJECT.
  v6f closest: int8 acc 0.9124 (best overall), wake recall 89.4% (fixes 2 of
  run-A's 15 misses, breaks 1), DET raw thr0.5 FRR 12.1%/FAR 3.6%; with T=0.5
  scaling: thr0.5 FRR 10.6%/FAR 5.6%, thr0.7 14.4%/2.5% - roughly tied with
  run-A, no clear win. All three plateau at FRR ~10-16%: the single-speaker
  sp09 test set (with systematically QUIET misses) is the ceiling, not the
  recipes. Tripwire machinery (FIXED/PERSISTENT/NEW) worked as designed.
  DECISION: run-A stays canonical on-device. Tie-break only via live
  Bengali FA/hr A/B (offered, not yet run). Next modeling lever if needed:
  gain augmentation (original v5 plan, never actually executed).
  UPDATE v6f-pt30 ON-DEVICE (mlab): user-tested BETTER than run-A on all 3 —
  fan-only 60 s -> 0 FA (run-A: 1); 5x Jago Guru @1m -> 5/5, no extras noted
  (run-A: 5/5 + 2 extra fires); Bengali YouTube @1m -> 3 fires/55 s (run-A:
  ~7). Verdict: hard-negative mining (1500 Bengali/Hindi windows) cuts the
  Bengali-FA rate by >half with no wake regression, despite the DET tripwire
  (0 fixed + 1 NEW `02_c09` on sp09). Lesson reinforced: single-speaker DET
  is noisy (±2 pts retrain swing); on-device FA/hr + wake recall is the real
  metric. v6f-pt30 (warm-start run-A + focal, old stem, 51824 B) now flashed
  for continued testing; run-A bytes kept at /tmp/model_data_runa.cc +
  git (revert: `git checkout -- main/model_data.cc`, rebuild, flash).
  NEXT: Path B (fresh-head + pt60 stem + focal + folded-cw) on Kaggle; if it
  wins DET *and* device, it takes the slot.
- [x] Phase 9 (v9 probe-verified) — 34 clips that FIRED v8 on a room speaker
  (phonetic-mined real words: joto-goru 0.949, coca-cola 0.934...) retrained
  as hard negatives (edgewake-v3/negpool-v3). SHIP PASS: int8 0.9569 (local
  repro == Kaggle), raw DET thr0.7 FRR 4.6%/FAR 1.4% (beats v8 6.1%/1.4%).
  14/15 fixed, 0 new (dud c02 persists). Branch model-v9, thr stays 0.7.
  Real verdict = Bengali YouTube 60 s re-test (SAME video as v8's 8 fires).
- [~] Threshold experiments (user-driven, on model-v9 branch).
  PRE-A BASELINE (revert target): KWS_QUIET_THR 0.7 / KWS_NOISY_THR 0.78 /
  margins 0.30/0.35 / 2-of-3 vote / VAD gate 1.5 s / debounce 1.5 s.
  Revert: `git revert 7d0eb7e 6051b6d` (option-A then option-B commits),
  or hand-restore the two #defines, rebuild, flash.
  - Option A (LIVE): quiet thr 0.80, noisy 0.88 (commit 7d0eb7e). Rationale:
    user-observed wakes 0.85-0.95 vs FAs <0.85 on small sample - but full
    logs show overlap (true 0.77 vs FA 0.945), so 0.85 was rejected as
    recall suicide; 0.80 is the reversible middle. Margins/vote unchanged.
  - Option B (NEXT): A + debounce 1.5 s -> 3.0 s (commit 6051b6d). Test
    words must be spaced >=4 s apart. Cuts repeat-fires, not distinct ones.
  - Option C (queued): revert to 0.7 + v10 data (probe survivors +
    YouTube captures). Decides between stricter gate vs smarter model.
  User order: A -> measure -> B -> measure -> C.
- [ ] Phase 7 — pre-roll + WebSocket stream (KICKED OFF 2026-09-10).
- [ ] Phase 4 — train DS-CNN (transfer-learn ML-zoo, DET/FA-hr eval).
- [x] Phase 6 (bring-up) — on-device KWS works: run-A int8 embedded via xxd,
      esp-tflite-micro 1.4.0 + ESP-NN + led_strip, 4 Hz full-window inference,
      thr 0.7 2/3 vote + 1.5 s debounce -> RGB LED + WAKE log. First light:
      idle p~0.01, WAKE p=0.76-0.99 on utterances. Arena 91.5/96 KB internal;
      ring 64 + fe 32 KB internal (alloc order: big blocks before I2S DMA).
  NOTE: resolver needs AddMean (converter emits MEAN for GAP, not AvgPool).
  OPEN: per-cycle cost fe 63.5 ms + invoke 110.5 ms = ~70% core (over budget;
  next = incremental frontend + invoke profiling). Run-A forensics closed:
  det_scaled_1.txt reproduced locally from its own artifacts (int8-vs-fp32
  jitter only) - run A healthy, canonical.
  FIXED false-fire storm: 10 WAKEs/75 s in a non-speaking room = per-window
  test FAR (~2%) becomes hundreds/hr continuous. Added VAD gate (WAKE needs
  speech within 1.5 s, suppressions logged) + kws? trace (p>=0.25) for live
  visibility. Post-gate: 0 fires/60 s, trace peaks ~0.59 on room noise.
  USER TESTS (margin rule live): fan-only 60 s -> 1 FA (w=0.73); 5x Jago Guru
  @1m -> 5/5 detected (p 0.94-0.99) + 2 extra fires; Bengali YouTube @1m ->
  ~7 fires/55 s (w 0.77-0.99, u~0.0). Root cause of Bengali FAs: 'jago' IS a
  Bengali word; GSC-English unknowns never taught otherwise. Fix = hard-
  negative mining with Bengali/Hindi speech, not thresholds (margin/thr
  can't stop 0.9+ confident misclassifications).
  DECISION v2 (firmware, no retrain): 5-frame sliding average + current-frame
  freshness (kills stale-vote fires like p=0.031) + noise-adaptive profile
  (silence-floor EMA, noisy thr 0.78/margin 0.35) + skip inference after 2 s
  silence (decays history). Research backing: ESPHome sliding-window+cutoff,
  adaptive thresholds, GraphemeAug TTS confusables (Sarvam script in dataset
  repo), interval-loss over focal.
  DECISION v2b (recall fix): 5-avg strangled true words (hot burst is only
  ~2 inferences: 0.99,0.99 then collapse; avg peaked 0.65 <0.7 gate). Now
  3-window 2-hot including current + relaxed current margin. Bengali-
  confident fires pass by design here - only retraining fixes those.
  HARD-NEG SOURCES (dataset repo ext/, all CC-BY-4.0, transcripts mined for
  guru/jaguar-like words): Kathbath bn test (2.8k clips/20 spk, 19 guru +
  jaguar hits), OpenSLR asr_bengali shards _0+_1+_2 (41k clips, 98 local
  transcript hits), Common Voice Hindi (1.7k clips + data.json, 6 hits).
  Full OpenSLR set is 16 shards - only pull more if v4 DET still leaks
  Bengali after mining with these three.
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
