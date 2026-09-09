// WiFi + ASR-server credentials for Phase-7 streaming.
//
// HOW TO CHANGE LATER (user asked): edit THE OTHER file,
//   main/wifi_config.h  (same two lines), then rebuild + flash:
//     idf.py build && idf.py -p /dev/ttyACM0 flash
// wifi_config.h is git-ignored and NEVER committed - your password
// cannot leak into git. This file is just the shape template.
#pragma once

#define WIFI_SSID   "YOUR_HOME_WIFI_NAME"
#define WIFI_PASS   "YOUR_HOME_WIFI_PASSWORD"

// Laptop running scripts/asr_server/server.py (its IP on the home net).
#define ASR_WS_URI  "ws://192.168.1.50:8765"
