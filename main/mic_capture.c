/*
 * edge_wake phase 2: continuous listening + log-mel frontend.
 * INMP441 -> ESP32-S3 I2S -> 16 kHz mono PCM -> circular ring buffer
 * -> per-hop energy VAD + live VU + on-device log-mel (esp-dsp FFT).
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
 *  - BOOT short-press (or serial 'f'): fresh 1 s PCM -> 61x40 log-mel dump
 *    (FE_PCM_* + FE_MEL_* markers, %a exact floats) for
 *    scripts/frontend_check.py. Hold BOOT 1.5 s (or serial 'r'): fresh
 *    30 s record (480000 samples, 960 KB PSRAM buffer; PSRAM OCTAL @ 80 MHz
 *    required, see sdkconfig.defaults), stats + hex dump between
 *    AUD_DUMP_START / AUD_DUMP_END, framed by REC_START / REC_END.
 *    Console runs at 921600 baud so the ~1.9 MB dump takes ~30 s.
 *  - host scripts: capture.py converts a 30 s dump to sample.wav (sends
 *    'r' itself); frontend_check.py verifies the S3 mel against numpy.
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
#include "frontend.h"

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
#define REC_SECONDS       30
#define REC_SAMPLES       (SAMPLE_RATE * REC_SECONDS)   // 480000 samples, 960 KB

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

// Hex dump with markers for scripts/capture.py. Uses plain printf (no ESP_LOG
// prefix) so the host parser stays trivial. 256 samples per line.
static void dump_hex(const int16_t *pcm, int n)
{
    printf("AUD_DUMP_START samples=%d sr=%d bits=16 ch=1 shift=%d\n",
           n, SAMPLE_RATE, GAIN_SHIFT);
    const int per_line = 256;
    for (int i = 0; i < n; i += per_line) {
        int m = n - i < per_line ? n - i : per_line;
        printf("AUD_DATA:");
        for (int j = 0; j < m; j++) {
            printf("%04x", (uint16_t)pcm[i + j]);
        }
        printf("\n");
    }
    printf("AUD_DUMP_END\n");
    fflush(stdout);
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

static int16_t *rec_buf = NULL;
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
static void do_capture(const char *why);
static void fe_dump_1s(const char *why);
static void fe_pow_dump(const char *why);

static void poll_serial_cmd(void)
{
    if (!have_serial_cmd) return;
    char c;
    int r = read(STDIN_FILENO, &c, 1);
    if (r != 1) return;
    if (c == 'f') {
        ESP_LOGI(TAG, "serial cmd 'f' -> frontend dump");
        fe_dump_1s("serial");
    } else if (c == 'r') {
        ESP_LOGI(TAG, "serial cmd 'r' -> 30 s capture");
        do_capture("serial");
    } else if (c == 'p') {
        ESP_LOGI(TAG, "serial cmd 'p' -> power-spectrum debug dump");
        fe_pow_dump("serial");
    }
}

static void do_capture(const char *why)
{
    ESP_LOGI(TAG, "=== capture (%s): recording %d s @ %d Hz mono ===",
             why, REC_SECONDS, SAMPLE_RATE);
    // flush DMA pipeline so the recording starts fresh
    for (int i = 0; i < 4; i++) read_mono_block(s_trash, 480);
    printf("REC_START seconds=%d\n", REC_SECONDS);

    int filled = 0;
    int64_t t0 = esp_timer_get_time();
    while (filled < REC_SAMPLES) {
        int want = REC_SAMPLES - filled;
        if (want > 960) want = 960;
        int got = read_mono_block(rec_buf + filled, want);
        if (got <= 0) {
            ESP_LOGW(TAG, "i2s read timeout at %d/%d", filled, REC_SAMPLES);
            continue;
        }
        filled += got;
    }
    int64_t t1 = esp_timer_get_time();
    ESP_LOGI(TAG, "recorded %d samples in %.2f s", filled,
             (t1 - t0) / 1000000.0);
    printf("REC_END samples=%d\n", filled);
    print_stats("capture", rec_buf, filled);
    dump_hex(rec_buf, filled);
    ESP_LOGI(TAG, "dump done. Run: python3 scripts/capture.py "
                  "--port /dev/ttyACM0 --baud 921600  (or parse existing log)");
    ESP_LOGI(TAG, "resuming continuous listening loop");
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
    static float mel_all[FE_NFRAMES_1S][FE_NMELS];
    int64_t t0 = esp_timer_get_time();
    for (int f = 0; f < nf; f++) {
        fe_frame(fe_pcm + f * FE_HOP, fe_prev_of(f), NULL, mel_all[f]);
    }
    int64_t t1 = esp_timer_get_time();
    for (int f = 0; f < nf; f++) {
        printf("FE_MEL:");
        for (int m = 0; m < FE_NMELS; m++) printf(" %a", (double)mel_all[f][m]);
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
    ESP_LOGI(TAG, "edge_wake phase2 boot. heap internal=%u",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

    gpio_set_direction(PIN_BUTTON, GPIO_MODE_INPUT);
    gpio_set_pull_mode(PIN_BUTTON, GPIO_PULLUP_ONLY);

    // Serial single-char commands: 'f' = 1 s frontend dump, 'r' = 30 s capture.
    // Non-blocking; if the UART VFS refuses, buttons remain the fallback.
    int fl = fcntl(STDIN_FILENO, F_GETFL, 0);
    if (fl >= 0 && fcntl(STDIN_FILENO, F_SETFL, fl | O_NONBLOCK) == 0) {
        have_serial_cmd = true;
    }
    ESP_LOGI(TAG, "serial commands %s", have_serial_cmd ? "ON ('f'/'r')" : "OFF (buttons only)");

    i2s_init();
    fe_init();

    // Ring lives in internal RAM (hot path, DMA-adjacent, future pre-roll).
    ring_buf = heap_caps_malloc(RING_SAMPLES * sizeof(int16_t),
                                MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!ring_buf) { ESP_LOGE(TAG, "cannot allocate %u-byte ring", (unsigned)(RING_SAMPLES * 2)); return; }
    // 30 s dataset buffer: PSRAM if present, else internal (won't fit 960 KB).
    rec_buf = heap_caps_malloc(REC_SAMPLES * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!rec_buf) rec_buf = heap_caps_malloc(REC_SAMPLES * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!rec_buf) rec_buf = malloc(REC_SAMPLES * sizeof(int16_t));
    if (!rec_buf) { ESP_LOGE(TAG, "cannot allocate %u-byte rec buffer", (unsigned)(REC_SAMPLES * 2)); return; }
    // 1 s frontend scratch: internal (hot path).
    fe_pcm = heap_caps_malloc(FE_SR * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!fe_pcm) fe_pcm = malloc(FE_SR * sizeof(int16_t));
    if (!fe_pcm) { ESP_LOGE(TAG, "cannot allocate fe buffer"); return; }
    ESP_LOGI(TAG, "ring %u KB internal, rec buffer %u KB @ %p",
             (unsigned)(RING_SAMPLES * 2 / 1024), (unsigned)(REC_SAMPLES * 2 / 1024), rec_buf);

    int btn_prev = 1;
    int64_t t_boot = esp_timer_get_time();
    int64_t last_vu = t_boot;
    bool selfcheck_done = false;
    int vu_peak_max = 0;

    ESP_LOGI(TAG, "listening... VAD flips on speech; BOOT short-press = mel dump, hold 1.5 s = 30 s capture; serial 'f'/'r' same");
    while (1) {
        poll_serial_cmd();
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

        int64_t now = esp_timer_get_time();
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
            int64_t t_press = esp_timer_get_time();
            vTaskDelay(pdMS_TO_TICKS(50));  // debounce
            if (gpio_get_level(PIN_BUTTON) != 0) { btn_prev = 1; continue; }
            // Distinguish tap vs hold. I2S overflows while we wait — harmless,
            // both actions flush the pipeline first.
            bool hold = false;
            while (gpio_get_level(PIN_BUTTON) == 0) {
                if (esp_timer_get_time() - t_press > 1500000) { hold = true; break; }
                vTaskDelay(pdMS_TO_TICKS(20));
            }
            if (hold) {
                do_capture("button-hold");
            } else {
                fe_dump_1s("button");
            }
            while (gpio_get_level(PIN_BUTTON) == 0) vTaskDelay(pdMS_TO_TICKS(20));
            btn_prev = 1;
            last_vu = esp_timer_get_time();
            continue;
        }
        btn_prev = btn;
    }
}
