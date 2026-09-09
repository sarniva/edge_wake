// Phase-7 uplink: WiFi station + websocket PCM to the ASR laptop.
//
// Design notes (tight-RAM machine):
// - WiFi/LWIP/ws buffers live in PSRAM (sdkconfig SPIRAM_USE_MALLOC);
//   KWS hot path (ring/arena/fe) stays internal, untouched.
// - Sends happen on the main task (same task as KWS): short 100 ms
//   timeouts so a dead server can never stall the mic loop.
// - Reconnect is event-driven: WiFi retries the AP, ws client is
//   (re)created on got-IP and destroyed on disconnect.
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_websocket_client.h"
#include "nvs_flash.h"

#include "streamer.h"
#include "wifi_config.h"

static const char *TAG = "stream";

#define WIFI_GOT_IP_BIT BIT0
static EventGroupHandle_t s_wifi_ev = NULL;
static esp_websocket_client_handle_t s_ws = NULL;
static bool s_ws_up = false;
static int s_retry = 0;

static void ws_event_cb(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    (void)arg; (void)base; (void)data;
    if (id == WEBSOCKET_EVENT_CONNECTED) {
        s_ws_up = true;
        ESP_LOGI(TAG, "ws connected -> %s", ASR_WS_URI);
    } else if (id == WEBSOCKET_EVENT_DISCONNECTED) {
        s_ws_up = false;
        ESP_LOGI(TAG, "ws disconnected");
    } else if (id == WEBSOCKET_EVENT_ERROR) {
        ESP_LOGW(TAG, "ws error");
    }
}

static void wifi_event_cb(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    (void)arg;
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_ws) { esp_websocket_client_destroy(s_ws); s_ws = NULL; }
        s_ws_up = false;
        xEventGroupClearBits(s_wifi_ev, WIFI_GOT_IP_BIT);
        // Gentle retry: the KWS loop keeps working offline meanwhile.
        if (s_retry < 20) {
            s_retry++;
            ESP_LOGI(TAG, "wifi down, retry %d (ssid=%s)", s_retry, WIFI_SSID);
            vTaskDelay(pdMS_TO_TICKS(2000));
            esp_wifi_connect();
        } else {
            ESP_LOGW(TAG, "wifi giving up after 20 tries (KWS-only mode)");
        }
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *ev = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "wifi up: " IPSTR, IP2STR(&ev->ip_info.ip));
        s_retry = 0;
        xEventGroupSetBits(s_wifi_ev, WIFI_GOT_IP_BIT);
        if (!s_ws) {
            esp_websocket_client_config_t cfg = {
                .uri = ASR_WS_URI,
                .reconnect_timeout_ms = 5000,
                .network_timeout_ms = 5000,
            };
            s_ws = esp_websocket_client_init(&cfg);
            if (s_ws) {
                esp_websocket_register_events(s_ws, WEBSOCKET_EVENT_ANY,
                                              ws_event_cb, NULL);
                esp_websocket_client_start(s_ws);
            } else {
                ESP_LOGE(TAG, "ws init failed");
            }
        }
    }
}

bool streamer_init(void)
{
    esp_err_t r = nvs_flash_init();
    if (r == ESP_ERR_NVS_NO_FREE_PAGES || r == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        r = nvs_flash_init();
    }
    if (r != ESP_OK) { ESP_LOGE(TAG, "nvs init failed"); return false; }

    s_wifi_ev = xEventGroupCreate();
    if (!s_wifi_ev) return false;
    if (esp_netif_init() != ESP_OK) return false;
    if (esp_event_loop_create_default() != ESP_OK) return false;
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t icfg = WIFI_INIT_CONFIG_DEFAULT();
    if (esp_wifi_init(&icfg) != ESP_OK) { ESP_LOGE(TAG, "wifi init failed"); return false; }
    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_cb, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event_cb, NULL);

    wifi_config_t wcfg = { 0 };
    strncpy((char *)wcfg.sta.ssid, WIFI_SSID, sizeof(wcfg.sta.ssid) - 1);
    strncpy((char *)wcfg.sta.password, WIFI_PASS, sizeof(wcfg.sta.password) - 1);
    wcfg.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    if (esp_wifi_set_mode(WIFI_MODE_STA) != ESP_OK) return false;
    if (esp_wifi_set_config(WIFI_IF_STA, &wcfg) != ESP_OK) return false;
    if (esp_wifi_start() != ESP_OK) return false;

    // Wait briefly for the first join so the boot log shows the verdict;
    // failure here is NOT fatal (KWS works offline, retries continue).
    EventBits_t b = xEventGroupWaitBits(s_wifi_ev, WIFI_GOT_IP_BIT,
                                        pdFALSE, pdFALSE, pdMS_TO_TICKS(8000));
    if (!(b & WIFI_GOT_IP_BIT)) {
        ESP_LOGW(TAG, "no wifi yet (ssid=%s) - KWS-only mode, retrying in bg", WIFI_SSID);
        return false;
    }
    // IP is up; give the ws handshake a moment too.
    for (int i = 0; i < 20 && !s_ws_up; i++) vTaskDelay(pdMS_TO_TICKS(100));
    ESP_LOGI(TAG, "uplink %s", s_ws_up ? "READY" : "ws pending (KWS-only for now)");
    return s_ws_up;
}

bool streamer_ready(void)
{
    return s_ws != NULL && s_ws_up && esp_websocket_client_is_connected(s_ws);
}

bool streamer_begin(void)
{
    if (!streamer_ready()) return false;
    const char *msg = "{\"type\":\"start\",\"sr\":16000,\"ch\":1}";
    int n = esp_websocket_client_send_text(s_ws, msg, strlen(msg),
                                           pdMS_TO_TICKS(1000));
    if (n < 0) { ESP_LOGW(TAG, "start send failed"); return false; }
    ESP_LOGI(TAG, "stream >>> start");
    return true;
}

bool streamer_send_pcm(const int16_t *pcm, size_t n_samples)
{
    if (!streamer_ready()) return false;
    int n = esp_websocket_client_send_bin(s_ws, (const char *)pcm,
                                          n_samples * sizeof(int16_t),
                                          pdMS_TO_TICKS(100));
    return n >= 0;
}

void streamer_end(void)
{
    if (s_ws && s_ws_up) {
        const char *msg = "{\"type\":\"end\"}";
        esp_websocket_client_send_text(s_ws, msg, strlen(msg),
                                       pdMS_TO_TICKS(1000));
    }
    ESP_LOGI(TAG, "stream >>> end");
}
