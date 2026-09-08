#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Wake-word class index in the int8 model output (silence/unknown/wake).
#define KWS_CLASS_WAKE 2
#define KWS_N_CLASSES  3

// One-time init: maps model, builds resolver, allocates tensors.
// Logs arena usage. Returns false on any failure (check serial log).
bool kws_init(void);

// Run one inference on a 61x40 log-mel frame buffer (row-major float).
// Applies training z-norm + int8 quantization internally.
// Returns wake posterior 0..1 (or -1.0 on invoke failure).
// invoke_us (optional, may be NULL) receives Invoke() time in microseconds.
float kws_infer(const float mel[61][40], int64_t *invoke_us);

#ifdef __cplusplus
}
#endif
