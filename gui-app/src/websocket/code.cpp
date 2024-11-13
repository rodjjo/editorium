#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <memory>
#include <set>
#include <tiny_websockets/client.hpp>

#include <FL/Fl.H>

#include "windows/progress_ui.h"
#include "websocket/uuid.h"
#include "websocket/code.h"
#include "misc/utils.h"
#include "misc/dialogs.h"
#include "misc/config.h"

namespace editorium
{
    namespace ws
    {

        using namespace websockets;

        namespace
        {
            bool running = true;
            std::mutex callback_mutex;
            std::mutex listener_mutex;
            callback_t current_callback;
            std::shared_ptr<WebsocketsClient> ws_client;
            std::shared_ptr<std::thread> ws_thread;
            std::shared_ptr<std::thread> ws_callback;
            listener_t listener;
            std::string current_task_id;
            std::set<std::string> reported_task_ids;
            std::chrono::_V2::system_clock::time_point last_report_time;
            std::chrono::_V2::system_clock::time_point last_check_time;
            size_t miss_count = 0;
            std::string ws_address = "ws://localhost:5001/";
        }

        json to_input(const api_payload_t &payload) {
            json inputs;
            inputs["images"] = json(json::value_type::array());
            for (const auto & image : payload.images) {
                inputs["images"].push_back(image->toJson());
            }
            inputs["texts"] = json(json::value_type::array());
            for (const auto & text : payload.texts) {
                inputs["texts"].push_back(text);
            }
            inputs["boxes"] = json(json::value_type::array());
            for (const auto & box : payload.boxes) {
                json box_json;
                box_json["x"] = box.x;
                box_json["y"] = box.y;
                box_json["x2"] = box.x2;
                box_json["y2"] = box.y2;
                inputs["boxes"].push_back(box_json);
            }
            return inputs;
        }

        void response_callback(const std::string& id, const json& response) {
            api_payload_t payload;
            if (response.contains("error")) {
                printf("Server error: %s\n", response["error"].get<std::string>().c_str());
                payload.server_error = response["error"];
            } else {
                if (response.contains("images"))
                    payload.images = newImageList(response["images"]);
                if (response.contains("texts"))
                    payload.texts = response["texts"];
                if (response.contains("boxes")) {
                    for (const auto & item : response["boxes"]) {
                        box_t box;
                        box.x = item["x"];
                        box.y = item["y"];
                        box.x2 = item["x2"];
                        box.y2 = item["y2"];
                        payload.boxes.push_back(box);
                    }
                }
            }
            std::unique_lock<std::mutex> lk(listener_mutex);
            if (listener) 
                listener(id, payload);
        }

        void replace_callback(callback_t callback)
        {
            while (!callback_mutex.try_lock()) {
                Fl::wait(0.015);
            }
            current_callback = callback;
            callback_mutex.unlock();
        }

        void wait_callback(bool wait_report=false)
        {
            bool should_continue = true;
            bool locked = false;
            while (should_continue)
            {
                callback_mutex.lock();
                if (!current_callback)
                {
                    should_continue = false;
                }
                callback_mutex.unlock();
                if (should_continue)
                {
                    Fl::wait(0.015);
                }

                if (wait_report) {
                     if (should_cancel()) {
                        should_continue = false;
                        continue;
                    }
                    std::unique_lock<std::mutex> lk(listener_mutex);
                    auto last_report_was = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - last_report_time).count();

                    if (last_report_was > 8) {
                        printf("Missed report for %lu seconds\n", std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - last_report_time).count());
                        should_continue = false;
                    } else {
                        auto last_check_was = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - last_check_time).count();
                        if (last_check_was > 1) {
                            if (reported_task_ids.find(current_task_id) == reported_task_ids.end()) {
                                miss_count++;
                            } else {
                                miss_count = 0;
                            }
                            last_check_time = std::chrono::high_resolution_clock::now();
                            if (miss_count > 5) {
                                printf("The task %s is lost\n", current_task_id.c_str());
                                should_continue = false;
                            }
                        }
                    }

                }
            }
        }

        std::shared_ptr<api_payload_t> execute(const std::string& task_type, const json &inputs, const json &config) {
            show_progress_window();

            auto result = std::make_shared<api_payload_t>();
            std::string id = UUID::generate_uuid();
            std::shared_ptr<bool> interrupted = std::make_shared<bool>(false);

            auto callback_send = [id, task_type, inputs, config, result, interrupted]() {
                printf("[callback_send] Executing task: %s, id: %s \n", task_type.c_str(), id.c_str());
                json req;
                req["task_type"] = std::string("api-") + task_type;
                req["input"] = inputs;
                req["config"] = config;
                req["id"] = id;
                std::atomic<bool> result_set = false;

                { // scope to set the listener
                    std::unique_lock<std::mutex> lk(listener_mutex);
                    listener = [uuid{id}, &result, &result_set] (const std::string& id, const api_payload_t & response) {
                        if (id == uuid) {
                            *result = response;
                            result_set.store(true);
                        }
                    };
                }

                printf("[callback_send] Sending request: %s, id: %s \n", task_type.c_str(), id.c_str());
                ws_client->send(req.dump());
                while (running && !(*interrupted)) {
                    if (result_set.load() || !ws_client->available()) {
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }

                { // scope to reset the listener
                    std::unique_lock<std::mutex> lk(listener_mutex);
                    listener = listener_t();
                }
            };

            wait_callback();
            
            {
                std::unique_lock<std::mutex> lk(listener_mutex);
                current_task_id = id;
                reported_task_ids.clear();
                miss_count = 0;
                last_report_time = std::chrono::high_resolution_clock::now();
                last_check_time = last_report_time;
            }

            replace_callback(callback_send);

            wait_callback(true);

            *interrupted = true;

            { // scope to reset the listener
                std::unique_lock<std::mutex> lk(listener_mutex);
                listener = listener_t();
                if (should_cancel()) {
                    printf("Cancelling task: %s \n", id.c_str());
                    json cancel_payload;
                    cancel_payload["cancel_task_id"] = id;
                    ws_client->send(cancel_payload.dump());
                }
            }
            
            hide_progress_window();

            if (result->boxes.empty() && result->images.empty() && result->texts.empty()) {
                if (!result->server_error.empty()) {
                    show_error(result->server_error.c_str());
                }
                return std::shared_ptr<api_payload_t>();
            }

            return result;
        }

        void execute_next_cb()
        {
            callback_t cb;
            { // locking area
                std::unique_lock<std::mutex> lk(callback_mutex);
                cb = current_callback;
            } // locking area

            if (!cb) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                return;
            }

            cb();

            { // locking area
                std::unique_lock<std::mutex> lk(callback_mutex);
                current_callback = callback_t();
            } // locking area
        }

        std::string send_request(const std::string& task_type,  const json &inputs,  const json &config)
        {
            if (!ws_client) {
                return std::string();
            }
            json req;
            req["task_type"] = task_type;
            req["inputs"] = inputs;
            req["config"] = config;
            req["id"] = UUID::generate_uuid();
            ws_client->send(req.dump());
            return req["id"];
        }

        void run_ws_client() {
            ws_client->connect(ws_address);

            ws_client->onMessage([&](WebsocketsClient&, WebsocketsMessage message){
                try {
                    json response = json::parse(message.data());
                    if (response.contains("id") && response.contains("result")) {
                        printf("Received response, id: %s \n", response["id"].get<std::string>().c_str());
                        response_callback(response["id"], response["result"]);
                    } else if (response.contains("current_task") && response.contains("pending_tasks")) {
                        auto cur_task = response["current_task"];
                        std::unique_lock<std::mutex> lk(listener_mutex);
                        last_report_time = std::chrono::high_resolution_clock::now();
                        reported_task_ids.clear();
                        reported_task_ids.insert(cur_task["id"]);
                        for (const auto & task : response["pending_tasks"]) {
                            reported_task_ids.insert(task["id"]);
                        }
                        if (cur_task["id"] == current_task_id) {
                            if (cur_task.contains("progress_title")) {
                                set_progress_text(cur_task["progress_title"]);
                            }

                            if (cur_task.contains("progress_percent")) {
                                float progress = cur_task["progress_percent"];
                                set_progress((size_t)progress, 100);
                            }
                        }
                    } else {
                        puts("Received unexpected message!");
                    }
                } catch (std::exception e) {
                    puts("Error parsing message!");
                }
            });

            if (ws_client->available()) {
                puts("Connected to websocket server!");
            }

            while (running) {
                if (!ws_client->available()) {
                    puts("Client not connected! trying to reconnect...");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    ws_client->connect(ws_address);
                    if (ws_client->available()) {
                        puts("Connected to websocket server!");
                    }
                    continue;
                }
                ws_client->poll();
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }

        }

        void run_websocket()
        {
            ws_address = get_config()->server_url();
            running = true;
            if (ws_callback) {
                return;
            }
            ws_client.reset(new WebsocketsClient());
            ws_thread.reset(new std::thread(run_ws_client));

            ws_callback.reset(new std::thread([] () {
                try {
                    while (running) {
                        execute_next_cb();
                    }
                } catch (std::exception e) {
                    return std::string("errored!");
                }
                return std::string("done!");
            }));
        }

        void stop_websocket()
        {
             if (ws_callback) {
                puts("Stopping websocket...");
                running = false;
                ws_callback->join();
                ws_thread->join();
                puts("websocket stopped...");
            }
        }
    } // namespace py

} // namespace editorium
