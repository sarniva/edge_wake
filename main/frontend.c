// Log-mel frontend for edge_wake KWS pipeline v1.
// Mirrors scripts/frontend_check.py EXACTLY:
//   int16 -> /32768 -> pre-emph 0.97 (stateful) -> Hamming (symmetric)
//   -> 512-pt real DFT via esp-dsp complex FFT -> power/N, bins 0..256
//   -> 40 HTK-mel triangles (peak=1, floor() bins), 20..4000 Hz
//   -> natural log(max(e, 1e-10))
#include <math.h>
#include <string.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "dsps_fft2r.h"
#include "frontend.h"

// FFT backend: the S3-vectorized aes3 kernel vs the portable ANSI C kernel.
// Phase-2 finding: aes3 output disagrees with numpy in quiet bins (up to ~2
// log units) while ANSI is expected to match to float32 rounding. The 'p'
// debug dump + scripts/pow_check compare decide which one ships.
// 0 = aes3 (fast), 1 = ANSI (reference).
#ifndef FE_USE_ANSI_FFT
#define FE_USE_ANSI_FFT 0
#endif

static const char *TAG = "frontend";

static float s_window[FE_FRAME];
static float s_mel[FE_NMELS][FE_NBINS];   // ~41 KB, hot path -> internal RAM
static float s_work[FE_NFFT * 2] __attribute__((aligned(16)));
static float s_x[FE_FRAME];               // per-call scratch (NOT on stack)
static bool s_inited = false;
// NOTE: no pre-emphasis state lives here. Pre-emphasis needs the sample that
// precedes the frame in the original stream; with 50% frame overlap a running
// tail would advance 2x too fast on replay, so the caller passes prev_in
// explicitly (Phase-2 lesson).

static float hz_to_mel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
static float mel_to_hz(float m)  { return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f); }

void fe_init(void)
{
    // Symmetric Hamming: w[n] = 0.54 - 0.46*cos(2*pi*n/(FRAME-1))
    for (int n = 0; n < FE_FRAME; n++) {
        s_window[n] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * n / (FE_FRAME - 1));
    }
    // HTK mel bank, triangular filters peak-normalized to 1.
    float mlo = hz_to_mel(FE_FMIN), mhi = hz_to_mel(FE_FMAX);
    float pts[FE_NMELS + 2];
    for (int i = 0; i < FE_NMELS + 2; i++) {
        float hz = mel_to_hz(mlo + (mhi - mlo) * i / (FE_NMELS + 1));
        pts[i] = floorf((FE_NFFT + 1) * hz / FE_SR);   // fft bin, like librosa
    }
    memset(s_mel, 0, sizeof(s_mel));
    for (int m = 0; m < FE_NMELS; m++) {
        float lo = pts[m], ce = pts[m + 1], hi = pts[m + 2];
        if (ce <= lo || hi <= ce) continue;            // degenerate guard
        for (int k = (int)lo; k < (int)ce && k < FE_NBINS; k++) {
            if (k >= 0) s_mel[m][k] = (k - lo) / (ce - lo);
        }
        for (int k = (int)ce; k < (int)hi && k < FE_NBINS; k++) {
            if (k >= 0) s_mel[m][k] = (hi - k) / (hi - ce);
        }
    }
    // Peak of every filter must be exactly 1 (center bin hits ce).
    ESP_ERROR_CHECK(dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE));
    s_inited = true;
    ESP_LOGI(TAG, "ready: frame=%d hop=%d nfft=%d mels=%d %.0f-%.0fHz (bank %.1f KB)",
             FE_FRAME, FE_HOP, FE_NFFT, FE_NMELS, (double)FE_FMIN, (double)FE_FMAX,
             (double)sizeof(s_mel) / 1024.0);
    ESP_LOGI(TAG, "addrs: window=%p mel=%p work=%p(+%u) x=%p",
             (void *)s_window, (void *)s_mel, (void *)s_work,
             (unsigned)sizeof(s_work), (void *)s_x);
}

int fe_nframes(int n_samples)
{
    if (n_samples < FE_FRAME) return 0;
    return 1 + (n_samples - FE_FRAME) / FE_HOP;
}

void fe_debug_x(const int16_t *pcm, float prev_in, float *prev_out, float *x_out)
{
    float prev = prev_in;
    for (int n = 0; n < FE_FRAME; n++) {
        float s = (float)pcm[n] / 32768.0f;
        float y = s - FE_PREEMPH * prev;
        prev = s;
        x_out[n] = y * s_window[n];
    }
    if (prev_out) *prev_out = prev;
}

void fe_debug_window(float *w_out)
{
    for (int n = 0; n < FE_FRAME; n++) w_out[n] = s_window[n];
}

void fe_frame(const int16_t *pcm, float prev_in, float *prev_out, float *mel_out)
{
    float power[FE_NBINS];
    fe_power(pcm, prev_in, prev_out, power);
    // mel weighted, logged
    for (int m = 0; m < FE_NMELS; m++) {
        float e = 0.0f;
        const float *w = s_mel[m];
        for (int k = 0; k < FE_NBINS; k++) {
            e += w[k] * power[k];
        }
        if (e < FE_FLOOR) e = FE_FLOOR;
        mel_out[m] = logf(e);
    }
}

void fe_power(const int16_t *pcm, float prev_in, float *prev_out, float *power_out)
{
    float *x = s_x;   // static scratch, NOT stack (see top of file)
    fe_debug_x(pcm, prev_in, prev_out, x);
    fe_power_from_x(x, power_out);
}

void fe_power_from_x(const float *x, float *power_out)
{
    // complex FFT with zero imaginary part
    for (int n = 0; n < FE_NFFT; n++) {
        s_work[2 * n] = x[n];
        s_work[2 * n + 1] = 0.0f;
    }
#if FE_USE_ANSI_FFT
    dsps_fft2r_fc32_ansi_(s_work, FE_NFFT, dsps_fft_w_table_fc32);
    dsps_bit_rev_fc32_ansi(s_work, FE_NFFT);
#else
    dsps_fft2r_fc32(s_work, FE_NFFT);
    dsps_bit_rev_fc32(s_work, FE_NFFT);
#endif
    for (int k = 0; k < FE_NBINS; k++) {
        float re = s_work[2 * k], im = s_work[2 * k + 1];
        power_out[k] = (re * re + im * im) / FE_NFFT;
    }
}
