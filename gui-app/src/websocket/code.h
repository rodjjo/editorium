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
        } api_payload_t;

        typedef std::function<std::string(const std::string& task_type, const json &inputs, const json & config)> request_cb_t;
        typedef std::function<void(request_cb_t &request_cb)> callback_t;
        typedef std::function<bool(const std::string& id, const api_payload_t & response)> listener_t;
        void execute(callback_t cb);
        size_t add_listener(listener_t listener);
        void remove_listener(size_t id);
        void run_websocket();
        void stop_websocket();
    } // namespace py
    
} // namespace dfe
