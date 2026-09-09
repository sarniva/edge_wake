#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Phase-7 uplink: WiFi station + websocket PCM stream to the ASR laptop.
//
// Protocol (see scripts/asr_server/server.py - the two MUST match):
//   -> text   {"type":"start","sr":16000,"ch":1}
//   -> binary int16le PCM chunks (any size, 960 B typical = one 30 ms hop)
//   -> text   {"type":"end"}
//   <- text   {"type":"result","text":"..."}  (logged, not acted on yet)
//
// All sends are best-effort with short timeouts: if the network is down,
// the chip keeps doing on-device KWS and just logs the failure. Streaming
// never blocks the mic loop for more than ~100 ms per call.

// Connect WiFi + websocket client. Returns true if the uplink is READY
// (joined AP and ws connected). False = keep KWS-only mode; retry happens
// automatically in the background on disconnect events.
bool streamer_init(void);

// True when a stream can start right now.
bool streamer_ready(void);

// Begin an utterance: sends {"type":"start"}. Returns false if not ready.
bool streamer_begin(void);

// One PCM chunk (int16 mono @16k). Returns false on send failure
// (caller should abort the stream).
bool streamer_send_pcm(const int16_t *pcm, size_t n_samples);

// Finish an utterance: sends {"type":"end"}.
void streamer_end(void);

#ifdef __cplusplus
}
#endif
