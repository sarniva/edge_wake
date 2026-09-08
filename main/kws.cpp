// On-device KWS inference: run-A int8 DS-CNN-S via esp-tflite-micro.
//
// Input contract (must match training exactly):
//   61x40 log-mel float (our frozen frontend) -> z-norm with KWS_NORM_MU/SD
//   (norm.npz, per-mel) -> int8 quantize with the model's own input
//   scale/zero-point -> Invoke -> dequantize output[WAKE].
// Model bytes live in flash (model_data.cc); arena is internal RAM.
#include <cmath>
#include <cstdint>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "kws.h"
#include "norm.h"

extern const unsigned char jagoguru_int8_tflite[];
extern const unsigned int jagoguru_int8_tflite_len;

static const char *TAG = "kws";

// 110 KB start; kws_init logs actual use so we can shrink to fit (<256 KB
// RAM budget: ring 64 KB + arena + stacks must stay internal).
static constexpr int kArenaSize = 96 * 1024;
alignas(16) static uint8_t s_arena[kArenaSize];

namespace {
tflite::MicroInterpreter *s_interp = nullptr;
TfLiteTensor *s_input = nullptr;
TfLiteTensor *s_output = nullptr;
float s_in_scale = 0;
int s_in_zp = 0;
float s_out_scale = 0;
int s_out_zp = 0;
}  // namespace

bool kws_init(void) {
    const tflite::Model *model = tflite::GetModel(jagoguru_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "model schema %d != supported %d", model->version(),
                 TFLITE_SCHEMA_VERSION);
        return false;
    }

    // Only the ops this graph needs (Invoke errors name any missing one).
    // NOTE: the converter emits MEAN (not AveragePool) for the GAP layer.
    static tflite::MicroMutableOpResolver<9> resolver;
    if (resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddMean() != kTfLiteOk ||
        resolver.AddAveragePool2D() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddQuantize() != kTfLiteOk ||
        resolver.AddDequantize() != kTfLiteOk) {
        ESP_LOGE(TAG, "resolver add failed");
        return false;
    }

    static tflite::MicroInterpreter interp(model, resolver, s_arena, kArenaSize);
    s_interp = &interp;
    if (s_interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed (arena %d?)", kArenaSize);
        return false;
    }

    s_input = s_interp->input(0);
    s_output = s_interp->output(0);
    if (s_input->dims->size != 4 || s_input->dims->data[1] != 61 ||
        s_input->dims->data[2] != 40 || s_input->type != kTfLiteInt8 ||
        s_output->dims->data[1] != KWS_N_CLASSES ||
        s_output->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "unexpected tensor geometry");
        return false;
    }
    s_in_scale = s_input->params.scale;
    s_in_zp = s_input->params.zero_point;
    s_out_scale = s_output->params.scale;
    s_out_zp = s_output->params.zero_point;

    ESP_LOGI(TAG,
             "ready: in 61x40x1 int8 (scale %.4f zp %d), out 1x%d int8, "
             "arena %u/%d bytes",
             (double)s_in_scale, s_in_zp, KWS_N_CLASSES,
             (unsigned)s_interp->arena_used_bytes(), kArenaSize);
    return true;
}

float kws_infer(const float mel[61][40], int64_t *invoke_us) {
    if (!s_interp) return -1.0f;
    // z-norm (training stats) + quantize into the input tensor
    int8_t *dst = s_input->data.int8;
    for (int f = 0; f < 61; f++) {
        for (int m = 0; m < 40; m++) {
            float v = (mel[f][m] - KWS_NORM_MU[m]) / KWS_NORM_SD[m];
            int q = (int)lroundf(v / s_in_scale) + s_in_zp;
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            dst[f * 40 + m] = (int8_t)q;
        }
    }
    int64_t t0 = esp_timer_get_time();
    if (s_interp->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return -1.0f;
    }
    int64_t t1 = esp_timer_get_time();
    if (invoke_us) *invoke_us = t1 - t0;
    int8_t qw = s_output->data.int8[KWS_CLASS_WAKE];
    return (qw - s_out_zp) * s_out_scale;  // ≈ softmax posterior 0..1
}
