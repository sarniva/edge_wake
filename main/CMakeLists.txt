idf_component_register(SRCS "mic_capture.c" "frontend.c" "kws.cpp" "model_data.cc" "streamer.c"
                        INCLUDE_DIRS "."
                        REQUIRES esp_driver_i2s esp_driver_gpio freertos esp_system esp_timer heap log
                                 esp_wifi esp_event esp_netif nvs_flash esp_websocket_client
                        PRIV_REQUIRES espressif__esp-dsp espressif__esp-tflite-micro espressif__led_strip)
