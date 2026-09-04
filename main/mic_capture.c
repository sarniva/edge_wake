/*
 * edge_wake stage 1: INMP441 -> ESP32-S3 I2S -> 16 kHz mono PCM -> buffer -> serial dump
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
 *  - prints live VU (RMS / peak / dBFS) every 500 ms so you can see the
 *    mic is alive without any script
 *  - 2 s after boot: records 30 s (480000 samples, 960 KB) into a PSRAM
 *    buffer (30 s does NOT fit internal RAM; PSRAM OCTAL @ 80 MHz required,
 *    see sdkconfig.defaults), prints stats + hex dump between
 *    AUD_DUMP_START / AUD_DUMP_END markers, framed by REC_START / REC_END
 *    so you know exactly which window was recorded (speak between them).
 *    Console runs at 921600 baud so the ~1.9 MB dump takes ~30 s.
 *  - press BOOT button (GPIO0) any time to re-record + re-dump
 *  - host script scripts/capture.py converts the dump to sample.wav
 *
 * Conversion note: INMP441 data is 24-bit left-justified in a 32-bit slot
 * (raw = data << 8). raw >> 16 would be "correct" 16-bit but very quiet
 * (-26 dBFS sensitivity). We use >> 12 (= +24 dB digital gain over raw)
 * with saturation so speech at ~30 cm sits around -12 dBFS. Tapping the
 * mic may still clip, which is fine. If normal speech clips constantly,
 * raise GAIN_SHIFT to 13 and rebuild.
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"

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
#define DMA_FRAMES        480         // 30 ms per DMA buffer at 16 kHz
#define DMA_DESC_NUM      6
#define REC_SECONDS       30
#define REC_SAMPLES       (SAMPLE_RATE * REC_SECONDS)   // 480000 samples, 960 KB

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
                      "Try GAIN_SHIFT 12/13 and rebuild.");
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

static int16_t *rec_buf = NULL;

static void do_capture(const char *why)
{
    ESP_LOGI(TAG, "=== capture (%s): recording %d s @ %d Hz mono ===",
             why, REC_SECONDS, SAMPLE_RATE);
    // flush DMA pipeline so the recording starts fresh
    int16_t trash[480];
    for (int i = 0; i < 4; i++) read_mono_block(trash, 480);
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
    ESP_LOGI(TAG, "press BOOT button for next capture");
}

void app_main(void)
{
    ESP_LOGI(TAG, "edge_wake mic_test boot. heap internal=%u",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

    gpio_set_direction(PIN_BUTTON, GPIO_MODE_INPUT);
    gpio_set_pull_mode(PIN_BUTTON, GPIO_PULLUP_ONLY);

    i2s_init();

    // 960 KB capture buffer: PSRAM if present, else internal (fits only the
    // old 5 s test; 30 s requires PSRAM, else you get an explicit error)
    size_t buf_bytes = REC_SAMPLES * sizeof(int16_t);
    rec_buf = heap_caps_malloc(buf_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!rec_buf) rec_buf = heap_caps_malloc(buf_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!rec_buf) rec_buf = malloc(buf_bytes);
    if (!rec_buf) { ESP_LOGE(TAG, "cannot allocate %u bytes", (unsigned)buf_bytes); return; }
    ESP_LOGI(TAG, "capture buffer %u bytes @ %p", (unsigned)buf_bytes, rec_buf);

    vTaskDelay(pdMS_TO_TICKS(2000));
    do_capture("boot");

    // live VU + button re-trigger loop
    int16_t vu[480];
    int btn_prev = 1;
    int64_t last_vu = esp_timer_get_time();
    while (1) {
        int got = read_mono_block(vu, 480);
        int64_t now = esp_timer_get_time();
        if (got > 0 && now - last_vu > 500000) {
            last_vu = now;
            long long sumsq = 0; int peak = 0;
            for (int i = 0; i < got; i++) {
                sumsq += (long long)vu[i] * vu[i];
                int a = vu[i] >= 0 ? vu[i] : -vu[i];
                if (a > peak) peak = a;
            }
            double rms = sqrt((double)sumsq / got);
            double db = (rms < 0.5) ? -96.0 : 20.0 * log10(rms / 32768.0);
            ESP_LOGI(TAG, "VU rms=%.0f peak=%d %.1f dBFS (speak/tap to see movement)",
                     rms, peak, db);
        }
        int btn = gpio_get_level(PIN_BUTTON);
        if (btn_prev == 1 && btn == 0) {
            vTaskDelay(pdMS_TO_TICKS(50));  // debounce
            if (gpio_get_level(PIN_BUTTON) == 0) {
                do_capture("button");
                // wait for release
                while (gpio_get_level(PIN_BUTTON) == 0) vTaskDelay(pdMS_TO_TICKS(20));
            }
        }
        btn_prev = btn;
        vTaskDelay(pdMS_TO_TICKS(5));
    }
}
