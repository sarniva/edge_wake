/*
 * edge_wake phase 6: continuous listening + log-mel frontend + on-device KWS.
 * INMP441 -> ESP32-S3 I2S -> 16 kHz mono PCM -> circular ring buffer
 * -> per-hop energy VAD + live VU + on-device log-mel (esp-dsp FFT)
 * -> run-A int8 DS-CNN-S (esp-tflite-micro + ESP-NN) at 4 Hz
 * -> 5-frame average + current-frame freshness + margin, adaptive
 *    quiet/noisy operating point, 1.5 s debounce -> RGB LED + WAKE log.
 *    Inference skipped (history decayed) after 2 s of silence.
 *
 * Wiring (INMP441 breakout -> ESP32-S3 DevKit):
 *   VDD -> 3V3        (NEVER 5V)
 *   GND -> GND
 *   L/R -> GND        (selects LEFT slot = mono source)
 *   SD  -> GPIO10     (I2S data in)
 *   WS  -> GPIO11     (word select / LRCK)
 *   SCK -> GPIO12     (bit clock)
 *
 * Why GPIO 10/11/12: free on S3 DevKit (not strapping 0/3/45/46,
 * not USB D-/D+ 19/20, not octal PSRAM 33-37), I2S-capable via GPIO matrix.
 *
 * Behavior:
 *  - boots, inits I2S STD RX 16 kHz / 32-bit / stereo (INMP441 is 24-bit in
 *    32-bit slots; we read stereo and keep LEFT slot only)
 *  - every 30 ms hop (480 samples): push into 2 s circular ring buffer,
 *    update energy VAD (enter/exit thresholds + 300 ms hangover), log
 *    SILENCE<->SPEECH transitions with timestamps
 *  - every 500 ms: VU line (RMS / peak / dBFS / VAD state / ring fill %)
 *  - 5 s after boot: ring self-check (newest hop re-read from ring must
 *    match what was written, incl. wraparound later) -> "ring self-check OK"
 *  - BOOT tap (or serial 'f'): fresh 1 s PCM -> 61x40 log-mel dump
 *    (FE_PCM_* + FE_MEL_* markers, %a exact floats) for
 *    scripts/frontend_check.py. (Long 30 s record-and-dump over PSRAM/UART
 *    used to live here too; it moved to the edge_wake_dataset repo, so this
 *    firmware uses no PSRAM at all.)
 *  - host script scripts/frontend_check.py verifies the S3 mel against numpy.
 *
 * Conversion note: INMP441 data is 24-bit left-justified in a 32-bit slot
 * (raw = data << 8). raw >> 16 would be "correct" 16-bit but very quiet
 * (-26 dBFS sensitivity). We use >> 12 (= +24 dB digital gain over raw)
 * with saturation so speech at ~30 cm sits around -12 dBFS. Tapping the
 * mic may still clip, which is fine. If normal speech clips constantly,
 * raise GAIN_SHIFT to 13 and rebuild.
 *
 * VAD tuning: thresholds below are a starting point (gain-shift dependent).
 * Watch VU lines in a quiet room: set VAD_EXIT_RMS ~2x the silence RMS and
 * VAD_ENTER_RMS ~2x VAD_EXIT_RMS, then rebuild.
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "led_strip.h"
#include "frontend.h"
#include "kws.h"
#include "streamer.h"

static const char *TAG = "mic_test";

// ---- pins ----
#define PIN_I2S_BCLK   GPIO_NUM_12
#define PIN_I2S_WS     GPIO_NUM_11
#define PIN_I2S_DIN    GPIO_NUM_10
#define PIN_BUTTON     GPIO_NUM_0

// ---- audio ----
#define SAMPLE_RATE       16000
#define GAIN_SHIFT        12          // >>12 (~+24 dB over raw): speech at 30 cm
                                      // sits ~-12 dBFS; tapping may still clip,
                                      // which is fine. Was 11 (clipped).
#define HOP_SAMPLES       480         // 30 ms per processing hop at 16 kHz
#define DMA_FRAMES        480         // one DMA buffer == one hop
#define DMA_DESC_NUM      6

// ---- ring buffer (the future wake-word pre-roll) ----
#define RING_SECONDS      2
#define RING_SAMPLES      (SAMPLE_RATE * RING_SECONDS)  // 32000 samples, 64 KB

// ---- VAD (energy gate with hysteresis + hangover) ----
// Calibrated 2026-09-04 from quiet-room log (traffic background):
// floor ~1500-3500 RMS -> exit ~2x floor min, enter ~2x exit.
// Speech at 30 cm is ~8000+ RMS, so it still trips reliably.
#define VAD_ENTER_RMS     6000        // hop RMS above this -> SPEECH
#define VAD_EXIT_RMS      3000        // hop RMS below this -> maybe SILENCE
#define VAD_HANGOVER_HOPS 10          // 300 ms of sub-exit hops before SILENCE

static i2s_chan_handle_t rx_chan = NULL;

static inline int16_t conv_s32_to_s16(int32_t s32)
{
    int32_t v = s32 >> GAIN_SHIFT;
    if (v > 32767)  v = 32767;
    if (v < -32768) v = -32768;
    return (int16_t)v;
}

static void i2s_init(void)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    chan_cfg.dma_frame_num = DMA_FRAMES;
    chan_cfg.dma_desc_num  = DMA_DESC_NUM;
    chan_cfg.auto_clear    = true;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_chan));

    i2s_std_config_t std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                        I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = PIN_I2S_BCLK,
            .ws   = PIN_I2S_WS,
            .dout = I2S_GPIO_UNUSED,
            .din  = PIN_I2S_DIN,
            .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
        },
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));
    ESP_LOGI(TAG, "I2S RX ready: %d Hz, BCLK=%d WS=%d DIN=%d",
             SAMPLE_RATE, PIN_I2S_BCLK, PIN_I2S_WS, PIN_I2S_DIN);
}

// Read exactly n_frames of stereo-32bit from I2S, convert LEFT slot to int16 mono.
// Returns number of mono samples written (0 on timeout).
static int read_mono_block(int16_t *out, int n_frames)
{
    // one frame = L(4B) + R(4B) = 8 bytes
    static int32_t *raw = NULL;
    static size_t raw_cap_frames = 0;
    if (raw_cap_frames < (size_t)n_frames) {
        free(raw);
        raw = malloc(n_frames * 2 * sizeof(int32_t));
        if (!raw) { ESP_LOGE(TAG, "no mem for raw block"); return 0; }
        raw_cap_frames = n_frames;
    }
    size_t bytes_want = n_frames * 2 * sizeof(int32_t);
    size_t bytes_read = 0;
    esp_err_t r = i2s_channel_read(rx_chan, (uint8_t *)raw, bytes_want,
                                   &bytes_read, pdMS_TO_TICKS(1000));
    if (r != ESP_OK || bytes_read == 0) return 0;
    int got_frames = bytes_read / (2 * sizeof(int32_t));
    for (int i = 0; i < got_frames; i++) {
        out[i] = conv_s32_to_s16(raw[2 * i]);   // LEFT slot (L/R pin = GND)
    }
    return got_frames;
}

// ---- ring buffer ----
static int16_t *ring_buf = NULL;
static uint32_t ring_write = 0;   // monotonic total samples ever pushed

static void ring_push(const int16_t *in, int n)
{
    uint32_t pos = ring_write % RING_SAMPLES;
    uint32_t first = RING_SAMPLES - pos < (uint32_t)n ? RING_SAMPLES - pos : n;
    memcpy(ring_buf + pos, in, first * sizeof(int16_t));
    if (first < (uint32_t)n) {
        memcpy(ring_buf, in + first, (n - first) * sizeof(int16_t));
    }
    ring_write += n;
}

// Copy n samples starting at absolute position start (oldest-first) into
// out. For the uplink pre-roll: walks the ring without a big staging
// buffer (48 KB would not fit internal RAM - see streamer note).
static void ring_slice(int16_t *out, uint32_t start, int n)
{
    for (int i = 0; i < n; i++) {
        out[i] = ring_buf[(start + i) % RING_SAMPLES];
    }
}

// Copy the newest n samples (n <= RING_SAMPLES) into out, oldest-first.
static void ring_newest(int16_t *out, int n)
{
    uint32_t start = ring_write - n;
    for (int i = 0; i < n; i++) {
        out[i] = ring_buf[(start + i) % RING_SAMPLES];
    }
}

static uint32_t ring_fill_pct(void)
{
    uint32_t f = ring_write < RING_SAMPLES ? ring_write : RING_SAMPLES;
    return (100 * f) / RING_SAMPLES;
}

// ---- KWS decision: short-window votes + freshness + adaptive point ----
// Live finding (2026-09): a true word burns hot for only ~2 inferences
// (~0.5 s: w=0.99,0.99 then collapse to ~0.27). A 5-frame average dilutes
// that to ~0.65 and strangles real detections, so the window is 3: fire
// needs the CURRENT frame hot plus at least one more hot frame in the last
// 3 (2-of-3 including current). The current-frame requirement kills
// stale-vote fires (observed: WAKE at p=0.031 carried by old votes).
// Bengali-confident fires (w~0.9,u~0.0, consecutive) are indistinguishable
// here by construction - that needs retraining with confusables.
#define KWS_WIN_N         3
// Option-A experiment (2026-09-09): quiet thr 0.70 -> 0.80 to cut
// YouTube FAs; noisy keeps its +0.08 offset (0.88). Margins, vote,
// VAD gate and cooldown UNCHANGED. Revert = restore 0.7f/0.78f.
#define KWS_QUIET_THR     0.8f
#define KWS_QUIET_MARGIN  0.3f
#define KWS_NOISY_THR     0.88f
#define KWS_NOISY_MARGIN  0.35f
#define KWS_NOISY_FLOOR   4000.0f  // silence-floor RMS above this = noisy room
// Option-B experiment (2026-09-09): debounce 1.5 s -> 3.0 s to cut
// repeat-fires on one sentence/Youtube event. NOTE: test words must be
// spaced >=4 s apart or the cooldown eats them. Revert = 1500000.
#define KWS_DEBOUNCE_US   3000000
#define KWS_PERIOD_US     250000   // 4 inferences/s
#define KWS_VAD_FRESH_US  1500000  // WAKE needs speech within last 1.5 s
#define KWS_IDLE_SKIP_US  2000000  // no speech this long -> skip inference
#define KWS_TRACE_P       0.25f    // log any posterior above this (visibility)

#define LED_GPIO          48       // DevKitC-1 addressable RGB LED
static led_strip_handle_t s_led = NULL;

static void led_set(uint8_t r, uint8_t g, uint8_t b)
{
    if (!s_led) return;
    led_strip_set_pixel(s_led, 0, r, g, b);
    led_strip_refresh(s_led);
}

static void led_init(void)
{
    led_strip_config_t cfg = {
        .strip_gpio_num = LED_GPIO,
        .max_leds = 1,
    };
    led_strip_rmt_config_t rmt = {
        .resolution_hz = 10 * 1000 * 1000,
    };
    if (led_strip_new_rmt_device(&cfg, &rmt, &s_led) != ESP_OK) {
        ESP_LOGW(TAG, "no addressable LED @ GPIO%d (logs still work)", LED_GPIO);
        s_led = NULL;
        return;
    }
    led_set(0, 16, 0);  // dim green = listening
}

// ---- VAD ----
typedef enum { VAD_SILENCE = 0, VAD_SPEECH = 1 } vad_state_t;
static vad_state_t vad_state = VAD_SILENCE;
static int vad_below_cnt = 0;

static int hop_rms(const int16_t *pcm, int n)
{
    long long sumsq = 0;
    for (int i = 0; i < n; i++) sumsq += (long long)pcm[i] * pcm[i];
    return (int)sqrt((double)sumsq / n);
}

// Returns true on a state transition.
static bool vad_update(int rms)
{
    if (vad_state == VAD_SILENCE) {
        if (rms >= VAD_ENTER_RMS) {
            vad_state = VAD_SPEECH;
            vad_below_cnt = 0;
            return true;
        }
    } else {
        if (rms < VAD_EXIT_RMS) {
            if (++vad_below_cnt >= VAD_HANGOVER_HOPS) {
                vad_state = VAD_SILENCE;
                vad_below_cnt = 0;
                return true;
            }
        } else {
            vad_below_cnt = 0;
        }
    }
    return false;
}

static void print_stats(const char *label, const int16_t *pcm, int n)
{
    long long sumsq = 0;
    long sum = 0;
    int peak = 0;
    for (int i = 0; i < n; i++) {
        int v = pcm[i];
        sum += v;
        sumsq += (long long)v * v;
        int a = v >= 0 ? v : -v;
        if (a > peak) peak = a;
    }
    double dc = (double)sum / n;
    double rms = sqrt((double)sumsq / n);
    double dbfs = (rms < 0.5) ? -96.0 : 20.0 * log10(rms / 32768.0);
    ESP_LOGI(TAG, "%s: n=%d dc=%+.1f rms=%.1f peak=%d (%.1f%% FS) %.1f dBFS",
             label, n, dc, rms, peak, 100.0 * peak / 32768.0, dbfs);
    if (peak < 200) {
        ESP_LOGW(TAG, "signal very quiet -> mic may be mute/disconnected, "
                      "or room silent. Tap the mic / speak close and re-capture.");
    } else if (peak >= 32760) {
        ESP_LOGW(TAG, "signal clipping -> loud or GAIN_SHIFT too small. "
                      "Try GAIN_SHIFT 13 and rebuild.");
    }
}

// CRC32 (zlib polynomial) over PCM: proves the host parsed the same bytes
// the S3 processed, ruling UART transport noise in or out definitively.
static uint32_t pcm_crc32(const int16_t *pcm, int n)
{
    uint32_t crc = 0xFFFFFFFFu;
    for (int i = 0; i < n; i++) {
        crc ^= (uint16_t)pcm[i];
        for (int b = 0; b < 16; b++) crc = (crc & 1) ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
    }
    return ~crc;
}

// Main-loop scratch: static, NOT stack (default main-task stack is small and
// printf/%a + dsps calls below it need headroom). Single task only.
static int16_t s_hop[HOP_SAMPLES];
static int16_t s_check[HOP_SAMPLES];
static int16_t s_trash[960];

static int16_t *fe_pcm = NULL;   // 1 s scratch for frontend dumps (32 KB)
static bool have_serial_cmd = false;

// Stateless-frontend helper: raw predecessor of frame f in fe_pcm
// (f == 0 -> stream start -> 0). Pass explicitly: with 50% overlap a
// running tail would advance pre-emphasis 2x too fast (Phase-2 bug find).
static inline float fe_prev_of(int f)
{
    return (f == 0) ? 0.0f : (float)fe_pcm[f * FE_HOP - 1] / 32768.0f;
}

// Forward declarations (defined further below).
static void fe_dump_1s(const char *why);
static void fe_pow_dump(const char *why);
static void stream_on_wake(int64_t now);

// Shared 61x40 inference window (~10 KB). Lives in PSRAM since Phase 7:
// WiFi static buffers need the internal RAM, and this window is touched
// 4x/s sequentially (cache-friendly) plus dumps - never concurrently,
// never by DMA. Allocated early in app_main with the other big blocks.
static float (*s_mel_win)[FE_NMELS] = NULL;

// KWS runtime state.
static bool s_kws_ok = false;
static float s_hist_w[KWS_WIN_N];  // recent wake posteriors (decision window)
static float s_hist_m[KWS_WIN_N];  // recent margins (decision window)
static int s_hist_idx = 0;
static float s_floor = 1500.0f;  // slow EMA of SILENCE hop RMS (room floor)
static bool s_noisy = false;     // room profile derived from s_floor
static int64_t s_last_wake_us = 0;
static int64_t s_last_inf_us = 0;
static int64_t s_last_speech_us = 0;  // freshest hop with VAD==SPEECH
static int s_inf_count = 0;
static uint64_t s_fe_us_sum = 0, s_inv_us_sum = 0;
static int64_t s_wake_until_us = 0;

static inline float kws_thr(void) { return s_noisy ? KWS_NOISY_THR : KWS_QUIET_THR; }
static inline float kws_margin(void) { return s_noisy ? KWS_NOISY_MARGIN : KWS_QUIET_MARGIN; }

// Long-silence bookkeeping: decay the smoothing history without spending
// DSP+invoke cycles. Keeps cadence and prevents stale hot votes surviving
// into the next utterance minutes later.
static void kws_idle_tick(void)
{
    s_hist_w[s_hist_idx] = 0.0f;
    s_hist_m[s_hist_idx] = 0.0f;
    s_hist_idx = (s_hist_idx + 1) % KWS_WIN_N;
}

// One full KWS cycle: newest 1 s from ring -> 61 mel frames -> infer ->
// vote -> WAKE action. Called at 4 Hz from the main loop.
static void kws_cycle(int64_t now)
{
    ring_newest(fe_pcm, FE_SR);
    int64_t t0 = esp_timer_get_time();
    for (int f = 0; f < FE_NFRAMES_1S; f++) {
        fe_frame(fe_pcm + f * FE_HOP, fe_prev_of(f), NULL, s_mel_win[f]);
    }
    int64_t t1 = esp_timer_get_time();
    int64_t inv_us = 0;
    float probs[KWS_N_CLASSES];
    float p = kws_infer(s_mel_win, probs, &inv_us);
    float p_unk = probs[1] > probs[0] ? probs[1] : probs[0];
    float margin = p - p_unk;
    s_fe_us_sum += (uint64_t)(t1 - t0);
    s_inv_us_sum += (uint64_t)(inv_us > 0 ? inv_us : 0);
    s_inf_count++;

    if (p < 0) return;  // invoke failed; error already logged
    s_hist_w[s_hist_idx] = p;
    s_hist_m[s_hist_idx] = margin;
    s_hist_idx = (s_hist_idx + 1) % KWS_WIN_N;
    // Window average is display/diagnosis; the FIRE rule below is what decides.
    float avg_w = 0, avg_m = 0;
    for (int i = 0; i < KWS_WIN_N; i++) {
        avg_w += s_hist_w[i];
        avg_m += s_hist_m[i];
    }
    avg_w /= KWS_WIN_N;
    avg_m /= KWS_WIN_N;
    float thr = kws_thr(), mthr = kws_margin();
    // Trace anything suspicious (max 4 lines/s): silence sits at ~0.01,
    // so anything here deserves a look. avg= shows the smoothed value that
    // actually decides, alongside the raw frame.
    if (p >= KWS_TRACE_P) {
        ESP_LOGI(TAG, "kws? w=%.3f u=%.3f s=%.3f avg=%.3f t=%.2fs", (double)p,
                 (double)probs[1], (double)probs[0], (double)avg_w,
                 (double)now / 1000000.0);
    }
    if (s_inf_count % 40 == 0) {
        ESP_LOGI(TAG, "kws #%d: p=%.3f fe=%.1fms inv=%.1fms (avg) prof=%s floor=%.0f",
                 s_inf_count, (double)p,
                 (double)s_fe_us_sum / s_inf_count / 1000.0,
                 (double)s_inv_us_sum / s_inf_count / 1000.0,
                 s_noisy ? "noisy" : "quiet", (double)s_floor);
    }
    // Fire needs: the CURRENT frame hot (freshness kills stale votes) plus
    // at least one more hot frame in the 3-window (a true word burns ~2).
    // A hot frame = thr + full margin; the current frame gets a relaxed
    // margin (word edges are ambiguous) but must clear the threshold.
    bool cur_hot = (p >= thr) && (margin >= mthr * 0.5f);
    int hits = cur_hot ? 1 : 0;
    for (int i = 0; i < KWS_WIN_N; i++) {
        int idx = (s_hist_idx + KWS_WIN_N - 1 - i) % KWS_WIN_N;
        if (i == 0) continue;  // slot just written = current frame, counted above
        if (s_hist_w[idx] >= thr && s_hist_m[idx] >= mthr) hits++;
        if (hits >= 2) break;
    }
    bool hot = cur_hot && (hits >= 2);
    if (!hot || now - s_last_wake_us <= KWS_DEBOUNCE_US) {
        if (s_wake_until_us && now > s_wake_until_us) {
            s_wake_until_us = 0;
            led_set(0, 16, 0);
        }
        return;
    }
    // VAD gate: per-window test FAR becomes hundreds of FA/hr on continuous
    // audio; genuine wake words always overlap speech energy. Suppressions
    // are logged (diagnostic gold for tuning the gate).
    if (now - s_last_speech_us > KWS_VAD_FRESH_US) {
        ESP_LOGI(TAG, "wake SUPPRESSED p=%.3f (no VAD speech for %.1fs)",
                 (double)p, (double)(now - s_last_speech_us) / 1000000.0);
        return;
    }
    s_last_wake_us = now;
    s_wake_until_us = now + 600000;
    led_set(64, 0, 0);
    ESP_LOGI(TAG, "WAKE w=%.3f avg=%.3f m=%.2f t=%.2fs",
             (double)p, (double)avg_w, (double)avg_m,
             (double)now / 1000000.0);
    stream_on_wake(now);
}

// ---- Phase-7 uplink: pre-roll + live stream after each WAKE ----
// Sends the 1.5 s BEFORE the wake word (already in the ring) plus live
// audio until 2 s of silence or 10 s max, then the laptop transcribes.
// KWS inference pauses while uploading (CPU + log clarity).
#define STREAM_PREROLL_SAMPLES 24000
#define STREAM_SIL_STOP_US     2000000
#define STREAM_MAX_US          10000000
#define STREAM_MIN_US          1000000
static bool s_streaming = false;
static int64_t s_stream_start_us = 0;
static int16_t s_tx[2048];  // ws frame scratch (4 KB internal)

static void stream_on_wake(int64_t now)
{
    if (s_streaming || !streamer_ready()) return;
    if (!streamer_begin()) return;
    // Pre-roll oldest-first in small frames (no 48 KB staging buffer).
    uint32_t avail = ring_write < RING_SAMPLES ? ring_write : RING_SAMPLES;
    uint32_t pre = avail < STREAM_PREROLL_SAMPLES ? avail : STREAM_PREROLL_SAMPLES;
    for (uint32_t off = 0; off < pre;) {
        int chunk = pre - off < 2048 ? pre - off : 2048;
        ring_slice(s_tx, ring_write - pre + off, chunk);
        if (!streamer_send_pcm(s_tx, chunk)) {
            streamer_end();
            return;  // KWS carries on; the laptop just misses one command
        }
        off += chunk;
    }
    s_streaming = true;
    s_stream_start_us = now;
    led_set(0, 0, 64);  // blue = uploading to laptop
    ESP_LOGI(TAG, "stream: pre-roll %.1fs sent, live...", (double)pre / 16000.0);
}

// Per-hop while streaming. Returns true while the stream continues.
static bool stream_feed(const int16_t *pcm, int n, int64_t now)
{
    bool ok = streamer_send_pcm(pcm, n);
    if (ok && now - s_stream_start_us < STREAM_MAX_US &&
        (now - s_stream_start_us < STREAM_MIN_US ||
         now - s_last_speech_us < STREAM_SIL_STOP_US)) {
        return true;
    }
    if (!ok) ESP_LOGW(TAG, "stream: send failed, aborting");
    streamer_end();
    s_streaming = false;
    led_set(0, 16, 0);
    ESP_LOGI(TAG, "stream: done (%.1fs)",
             (double)(now - s_stream_start_us) / 1000000.0);
    return false;
}

static void poll_serial_cmd(void)
{
    if (!have_serial_cmd) return;
    char c;
    int r = read(STDIN_FILENO, &c, 1);
    if (r != 1) return;
    if (c == 'f') {
        ESP_LOGI(TAG, "serial cmd 'f' -> frontend dump");
        fe_dump_1s("serial");
    } else if (c == 'p') {
        ESP_LOGI(TAG, "serial cmd 'p' -> power-spectrum debug dump");
        fe_pow_dump("serial");
    }
}

// Fresh 1 s of PCM -> log-mel frames -> exact-float dump for
// scripts/frontend_check.py. Prints FE_PCM_* then FE_MEL_* blocks.
static void fe_dump_1s(const char *why)
{
    const int want = FE_SR;   // 16000 samples
    ESP_LOGI(TAG, "=== frontend dump (%s): 1 s PCM + log-mel ===", why);
    for (int i = 0; i < 4; i++) read_mono_block(s_trash, 480);
    printf("FE_REC_START\n");

    int filled = 0;
    while (filled < want) {
        int chunk = want - filled > 960 ? 960 : want - filled;
        int got = read_mono_block(fe_pcm + filled, chunk);
        if (got <= 0) { ESP_LOGW(TAG, "fe i2s timeout at %d", filled); continue; }
        filled += got;
    }
    printf("FE_REC_END samples=%d\n", filled);
    print_stats("fe_pcm", fe_pcm, filled);

    printf("FE_PCM_START samples=%d\n", filled);
    printf("FE_PCM_CRC %08lx\n", (unsigned long)pcm_crc32(fe_pcm, filled));
    for (int i = 0; i < filled; i += 256) {
        int m = filled - i < 256 ? filled - i : 256;
        printf("FE_PCM:");
        for (int j = 0; j < m; j++) printf("%04x", (uint16_t)fe_pcm[i + j]);
        printf("\n");
    }
    printf("FE_PCM_END\n");

    int nf = fe_nframes(filled);
    printf("FE_MEL_START frames=%d mels=%d\n", nf, FE_NMELS);
    // Compute first (timed, no prints), dump after: the ms/frame number is
    // honest CPU cost, not UART time. Matches future KWS flow.
    int64_t t0 = esp_timer_get_time();
    for (int f = 0; f < nf; f++) {
        fe_frame(fe_pcm + f * FE_HOP, fe_prev_of(f), NULL, s_mel_win[f]);
    }
    int64_t t1 = esp_timer_get_time();
    for (int f = 0; f < nf; f++) {
        printf("FE_MEL:");
        for (int m = 0; m < FE_NMELS; m++) printf(" %a", (double)s_mel_win[f][m]);
        printf("\n");
    }
    printf("FE_MEL_END\n");
    ESP_LOGI(TAG, "fe: %d frames in %.2f ms (%.2f ms/frame)", nf,
             (t1 - t0) / 1000.0, (t1 - t0) / 1000.0 / nf);
    // Diagnosis replay: same reset + order as the mel loop above, so this x[]
    // is bit-identical to what the mel used. First 8 frames only.
    static float x_dbg[FE_FRAME];
    for (int f = 0; f < 8 && f < nf; f++) {
        fe_debug_x(fe_pcm + f * FE_HOP, fe_prev_of(f), NULL, x_dbg);
        printf("FE_X frame=%d:", f);
        for (int n = 0; n < FE_FRAME; n++) printf(" %a", (double)x_dbg[n]);
        printf("\n");
    }
    printf("FE_X_END\n");
    fflush(stdout);
    ESP_LOGI(TAG, "resuming continuous listening loop");
}

// Debug: capture 1 s, find loudest + quietest frames by total power, dump
// their 257-bin power spectra (%a exact). For scripts/pow_check.py —
// localizes frontend mismatches to FFT-vs-melbank without guessing.
static void fe_pow_dump(const char *why)
{
    const int want = FE_SR;
    ESP_LOGI(TAG, "=== power debug dump (%s) ===", why);
    for (int i = 0; i < 4; i++) read_mono_block(s_trash, 480);
    int filled = 0;
    while (filled < want) {
        int chunk = want - filled > 960 ? 960 : want - filled;
        int got = read_mono_block(fe_pcm + filled, chunk);
        if (got <= 0) continue;
        filled += got;
    }
    int nf = fe_nframes(filled);
    static float pow_scratch[FE_NBINS];
    printf("FE_PCM_START samples=%d\n", filled);
    printf("FE_PCM_CRC %08lx\n", (unsigned long)pcm_crc32(fe_pcm, filled));
    for (int i = 0; i < filled; i += 256) {
        int m = filled - i < 256 ? filled - i : 256;
        printf("FE_PCM:");
        for (int j = 0; j < m; j++) printf("%04x", (uint16_t)fe_pcm[i + j]);
        printf("\n");
    }
    printf("FE_PCM_END\n");
    int i_loud = 0, i_quiet = 0;
    float e_loud = -1.0f, e_quiet = 1e30f;
    for (int f = 0; f < nf; f++) {
        fe_power(fe_pcm + f * FE_HOP, fe_prev_of(f), NULL, pow_scratch);
        float tot = 0.0f;
        for (int k = 0; k < FE_NBINS; k++) tot += pow_scratch[k];
        if (tot > e_loud) { e_loud = tot; i_loud = f; }
        if (tot < e_quiet) { e_quiet = tot; i_quiet = f; }
    }
    printf("FE_POW_START frames=%d bins=%d loud=%d quiet=%d\n", nf, FE_NBINS, i_loud, i_quiet);
    {   // Window table first: proves/disproves the table itself in one shot.
        static float w_dump[FE_FRAME];
        fe_debug_window(w_dump);
        printf("FE_WIN n=%d:", FE_FRAME);
        for (int n = 0; n < FE_FRAME; n++) printf(" %a", (double)w_dump[n]);
        printf("\n");
    }
    // Second replay: same state trajectory, but also capture the FFT input
    // vector x[] for the two frames of interest (FE_X lines). x + power
    // together blame pre-FFT stages vs the FFT itself, no guessing.
    static float x_save[2][FE_FRAME];
    for (int f = 0; f < nf; f++) {
        int slot = (f == i_loud) ? 0 : (f == i_quiet) ? 1 : -1;
        if (slot >= 0) {
            fe_debug_x(fe_pcm + f * FE_HOP, fe_prev_of(f), NULL, x_save[slot]);
            fe_power_from_x(x_save[slot], pow_scratch);
            printf("FE_X frame=%d:", f);
            for (int n = 0; n < FE_FRAME; n++) printf(" %a", (double)x_save[slot][n]);
            printf("\n");
            printf("FE_POW frame=%d:", f);
        } else {
            fe_power(fe_pcm + f * FE_HOP, fe_prev_of(f), NULL, pow_scratch);
            if (f != i_loud && f != i_quiet) continue;
            printf("FE_POW frame=%d:", f);
        }
        for (int k = 0; k < FE_NBINS; k++) printf(" %a", (double)pow_scratch[k]);
        printf("\n");
    }
    printf("FE_POW_END\n");
    fflush(stdout);
    ESP_LOGI(TAG, "resuming continuous listening loop");
}

void app_main(void)
{
    ESP_LOGI(TAG, "edge_wake phase2 boot. heap internal=%u largest=%u",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));

    gpio_set_direction(PIN_BUTTON, GPIO_MODE_INPUT);
    gpio_set_pull_mode(PIN_BUTTON, GPIO_PULLUP_ONLY);

    // Serial single-char commands: 'f' = 1 s frontend dump, 'p' = power dump.
    // Non-blocking; if the UART VFS refuses, buttons remain the fallback.
    int fl = fcntl(STDIN_FILENO, F_GETFL, 0);
    if (fl >= 0 && fcntl(STDIN_FILENO, F_SETFL, fl | O_NONBLOCK) == 0) {
        have_serial_cmd = true;
    }
    ESP_LOGI(TAG, "serial commands %s", have_serial_cmd ? "ON ('f'/'p')" : "OFF (buttons only)");

    // Big buffers FIRST (unfragmented heap).
    // Ring 64 KB -> PSRAM since Phase 7: WiFi/LWIP statics fragment internal
    // RAM (largest free block ~31 KB < 64 KB, boot dies at "cannot allocate
    // ring"). Traffic is tiny (1 KB/30 ms in, 32 KB copies 4x/s out,
    // sequential = cache-friendly); the I2S DMA buffers stay internal.
    ring_buf = heap_caps_malloc(RING_SAMPLES * sizeof(int16_t),
                                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!ring_buf) { ESP_LOGE(TAG, "cannot allocate %u-byte ring (PSRAM?)", (unsigned)(RING_SAMPLES * 2)); return; }
    // 1 s frontend scratch: internal (hot path).
    fe_pcm = heap_caps_malloc(FE_SR * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!fe_pcm) fe_pcm = malloc(FE_SR * sizeof(int16_t));
    if (!fe_pcm) { ESP_LOGE(TAG, "cannot allocate fe buffer"); return; }
    // 61x40 mel window: PSRAM (see decl note - WiFi owns internal now).
    s_mel_win = heap_caps_malloc(FE_NFRAMES_1S * sizeof(*s_mel_win), MALLOC_CAP_SPIRAM);
    if (!s_mel_win) { ESP_LOGE(TAG, "cannot allocate mel window (PSRAM?)"); return; }
    ESP_LOGI(TAG, "ring %u KB + fe 32 KB internal, largest free now %u",
             (unsigned)(RING_SAMPLES * 2 / 1024),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));

    i2s_init();
    fe_init();
    led_init();
    s_kws_ok = kws_init();
    ESP_LOGI(TAG, "kws %s", s_kws_ok ? "ARMED (3win thr 0.80/0.88 adaptive, 4 Hz)" : "OFF (init failed)");
    bool uplink = streamer_init();
    ESP_LOGI(TAG, "uplink %s", uplink ? "READY (pre-roll + live to laptop)"
                                      : "OFFLINE (KWS-only; wifi retries in bg)");

    int btn_prev = 1;
    int64_t t_boot = esp_timer_get_time();
    int64_t last_vu = t_boot;
    bool selfcheck_done = false;
    int vu_peak_max = 0;

    ESP_LOGI(TAG, "listening... VAD flips on speech; BOOT = mel dump; serial 'f'/'p' same");
    while (1) {
        poll_serial_cmd();
        int64_t now = esp_timer_get_time();
        // KWS at 4 Hz (full-window recompute for bring-up; incremental later).
        // Paused while uploading a command (stream owns the mic + CPU then).
        // Long silence skips DSP+invoke (~170 ms saved per skipped cycle) and
        // decays the smoothing history so nothing stale survives into the
        // next utterance. First 30 s after boot always run (proves pipeline).
        if (s_kws_ok && !s_streaming && now - s_last_inf_us >= KWS_PERIOD_US &&
            ring_write >= (uint32_t)FE_SR) {
            s_last_inf_us = now;
            if (now - t_boot > 30000000 &&
                now - s_last_speech_us > KWS_IDLE_SKIP_US) {
                kws_idle_tick();
            } else {
                kws_cycle(now);
            }
            now = esp_timer_get_time();
        }
        int got = read_mono_block(s_hop, HOP_SAMPLES);
        if (got != HOP_SAMPLES) {
            ESP_LOGW(TAG, "short hop: %d/%d (I2S underrun?)", got, HOP_SAMPLES);
            if (got <= 0) { vTaskDelay(pdMS_TO_TICKS(5)); continue; }
        }
        ring_push(s_hop, got);

        // Gate VAD until the DMA pipeline has settled: the first ~10 hops
        // after boot contain startup garbage (I2S/DMA priming, auto_clear)
        // and must not latch a bogus SPEECH at t=0.03 s.
        int rms = hop_rms(s_hop, got);
        if (ring_write > 10 * (uint32_t)HOP_SAMPLES && vad_update(rms)) {
            ESP_LOGI(TAG, "VAD: -> %s (rms=%d, t=%.2fs)",
                     vad_state == VAD_SPEECH ? "SPEECH" : "SILENCE",
                     rms, (esp_timer_get_time() - t_boot) / 1000000.0);
        }
        // Freshness source for the KWS VAD gate (updated every hop, not
        // just on transitions, so long utterances don't go stale).
        if (vad_state == VAD_SPEECH) {
            s_last_speech_us = now;
        } else {
            // Room-floor estimate: slow EMA of SILENCE hop RMS only (speech
            // must not pollute it). ~15 s time constant at 33 hops/s.
            s_floor += 0.002f * ((float)rms - s_floor);
            bool noisy = s_floor > KWS_NOISY_FLOOR;
            if (noisy != s_noisy) {
                s_noisy = noisy;
                ESP_LOGI(TAG, "room profile -> %s (silence floor %.0f)",
                         noisy ? "NOISY thr=0.88" : "quiet thr=0.80",
                         (double)s_floor);
            }
        }
        // Phase-7: while uploading, every hop goes to the laptop (stop on
        // 2 s silence / 10 s max inside stream_feed).
        if (s_streaming) {
            stream_feed(s_hop, got, now);
            now = esp_timer_get_time();
        }

        now = esp_timer_get_time();
        if (now - last_vu > 500000) {
            last_vu = now;
            long long sumsq = 0; int peak = 0;
            for (int i = 0; i < got; i++) {
                sumsq += (long long)s_hop[i] * s_hop[i];
                int a = s_hop[i] >= 0 ? s_hop[i] : -s_hop[i];
                if (a > peak) peak = a;
            }
            if (peak > vu_peak_max) vu_peak_max = peak;
            double r = sqrt((double)sumsq / got);
            double db = (r < 0.5) ? -96.0 : 20.0 * log10(r / 32768.0);
            ESP_LOGI(TAG, "VU rms=%.0f peak=%d (max %d) %.1f dBFS vad=%s ring=%u%%",
                     r, peak, vu_peak_max, db,
                     vad_state == VAD_SPEECH ? "SPEECH" : "silence",
                     (unsigned)ring_fill_pct());
        }

        // One-shot integrity proof: the hop just pushed must read back
        // identically as the ring's newest samples (runs after 5 s uptime,
        // i.e. after the 2 s ring has wrapped at least once).
        if (!selfcheck_done && now - t_boot > 5000000 && got == HOP_SAMPLES) {
            selfcheck_done = true;
            ring_newest(s_check, HOP_SAMPLES);
            if (memcmp(s_check, s_hop, HOP_SAMPLES * sizeof(int16_t)) == 0) {
                ESP_LOGI(TAG, "ring self-check OK (%u samples stored, wrap-tested)",
                         (unsigned)ring_write);
            } else {
                ESP_LOGE(TAG, "ring self-check FAILED (w=%u)", (unsigned)ring_write);
            }
        }

        int btn = gpio_get_level(PIN_BUTTON);
        if (btn_prev == 1 && btn == 0) {
            vTaskDelay(pdMS_TO_TICKS(50));  // debounce
            if (gpio_get_level(PIN_BUTTON) != 0) { btn_prev = 1; continue; }
            fe_dump_1s("button");
            while (gpio_get_level(PIN_BUTTON) == 0) vTaskDelay(pdMS_TO_TICKS(20));
            btn_prev = 1;
            last_vu = esp_timer_get_time();
            continue;
        }
        btn_prev = btn;
    }
}
