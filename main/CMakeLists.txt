idf_component_register(SRCS "mic_capture.c"
                       INCLUDE_DIRS "."
                       REQUIRES esp_driver_i2s esp_driver_gpio freertos esp_system esp_timer heap log)
