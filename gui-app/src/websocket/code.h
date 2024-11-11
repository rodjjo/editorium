#pragma once

#include <string>
#include <functional>
#include "images/image.h"
#include <nlohmann/json.hpp>


namespace editorium
{
    namespace ws
    {
        using json = nlohmann::json;

        typedef struct {
            int x;
            int y;
            int x2;
            int y2;
        } box_t;

        typedef struct {
            std::vector<editorium::image_ptr_t> images;
            std::vector<std::string> texts;
            std::vector<box_t> boxes;
            std::string server_error;
        } api_payload_t;

        typedef std::function<void()> callback_t;
        typedef std::function<void(const std::string& id, const api_payload_t & response)> listener_t;

        json to_input(const api_payload_t &payload);
        std::shared_ptr<api_payload_t> execute(const std::string& task_type,  const json &inputs,  const json &config);

        void run_websocket();
        void stop_websocket();
    } // namespace py
    
} // namespace editorium
