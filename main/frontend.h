#pragma once
#include <stdint.h>

// Frozen frontend params — edge_wake KWS pipeline v1.
// scripts/frontend_check.py implements the BIT-IDENTICAL pipeline in numpy.
// If you change ANY constant here, change it there too and re-verify.
#define FE_SR       16000
#define FE_FRAME    512         // 32 ms analysis window
#define FE_HOP      256         // 16 ms stride (62.5 fps)
#define FE_NFFT     512
#define FE_NBINS    (FE_NFFT / 2 + 1)   // 257
#define FE_NMELS    40
#define FE_FMIN     20.0f
#define FE_FMAX     4000.0f
#define FE_PREEMPH  0.97f
#define FE_FLOOR    1e-10f      // log(max(e, FLOOR)), natural log

// 1 s of audio -> this many frames: (16000-512)/256 + 1 = 61 (floor)
#define FE_NFRAMES_1S 61

// One-time init: builds Hamming window + mel filterbank, inits esp-dsp FFT.
// Logs RAM cost. Must be called once before fe_frame().
void fe_init(void);

// One 512-sample int16 frame -> 40 log-mel floats. STATELESS (pure function):
// prev_in = raw sample immediately preceding pcm[0] in the ORIGINAL STREAM
// (pcm[-1]/32768, or 0.0 for the very first frame); prev_out = pcm[511]/32768
// for callers that walk a contiguous stream. Callers replaying overlapped
// frames MUST pass the true predecessor, not a running state (50% overlap
// would otherwise advance pre-emphasis 2x too fast - Phase-2 bug find).
void fe_frame(const int16_t *pcm, float prev_in, float *prev_out, float *mel_out);

// Number of frames for n_samples: 1 + floor((n-FRAME)/HOP), or 0 if too short.
int fe_nframes(int n_samples);

// Power spectrum of one frame (257 bins, |X|^2/N). Stateless like fe_frame.
void fe_power(const int16_t *pcm, float prev_in, float *prev_out, float *power_out);

// Debug: recompute one frame's FFT input vector x[] (post int-convert,
// pre-emphasis, windowing) into x_out. Stateless like fe_frame.
void fe_debug_x(const int16_t *pcm, float prev_in, float *prev_out, float *x_out);

// Debug: dump internal Hamming table (512 %a floats) for host comparison.
void fe_debug_window(float *w_out);

// FFT + power only, no stream state (pure function of x).
void fe_power_from_x(const float *x, float *power_out);
