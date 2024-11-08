#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <memory>
#include <tiny_websockets/client.hpp>

#include <FL/Fl.H>


#include "websocket/uuid.h"
#include "websocket/code.h"
#include "misc/utils.h"

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
        }

        json to_input(const api_payload_t &payload) {
            json inputs;
            inputs["images"] = json::array();
            for (const auto & image : payload.images) {
                inputs["images"].push_back(image->toJson());
            }
            inputs["texts"] = payload.texts;
            inputs["boxes"] = json::array();
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
            std::unique_lock<std::mutex> lk(listener_mutex);
            if (listener) 
                listener(id, payload);
        }

        void replace_callback(callback_t callback)
        {
            while (!callback_mutex.try_lock()) {
                Fl::wait(0.033);
            }
            current_callback = callback;
            callback_mutex.unlock();
        }

        void wait_callback()
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
                    Fl::wait(0.033);
                }
            }
        }

        std::shared_ptr<api_payload_t> execute(const std::string& task_type, const json &inputs, const json &config, bool *canceled_checker) {
            std::shared_ptr<api_payload_t> result;
            std::string id = UUID::generate_uuid();
            auto callback_send = [id, task_type, inputs, config, &result, canceled_checker]() {
                printf("[callback_send] Executing task: %s, id: %s \n", task_type.c_str(), id.c_str());
                json req;
                json default_input;
                default_input["default"] = inputs;
                req["task_type"] = std::string("api-") + task_type;
                req["input"] = default_input;
                req["config"] = config;
                req["id"] = id;
                std::atomic<bool> result_set = false;

                { // scope to set the listener
                    std::unique_lock<std::mutex> lk(listener_mutex);
                    listener = [uuid{id}, &result, &result_set] (const std::string& id, const api_payload_t & response) {
                        if (id == uuid) {
                            result = std::make_shared<api_payload_t>(response);
                            result_set.store(true);
                        }
                    };
                }

                printf("[callback_send] Sending request: %s, id: %s \n", task_type.c_str(), id.c_str());
                ws_client->send(req.dump());
                while (true) {
                    if (result_set.load() || !ws_client->available() || (canceled_checker && *canceled_checker)) {
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(33));
                }

                { // scope to reset the listener
                    std::unique_lock<std::mutex> lk(listener_mutex);
                    listener = listener_t();
                }
            };


            wait_callback();
            printf("[ws:execute] Executing task: %s, id: %s \n", task_type.c_str(), id.c_str());
            replace_callback(callback_send);
            printf("[ws:execute] Waiting task: %s, id: %s \n", task_type.c_str(), id.c_str());
            wait_callback();

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
                std::this_thread::sleep_for(std::chrono::milliseconds(33));
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
            std::string address = "ws://localhost:5001/";
            ws_client->connect(address);

            ws_client->onMessage([&](WebsocketsClient&, WebsocketsMessage message){
                try {
                    json response = json::parse(message.data());
                    if (response.contains("id") && response.contains("result")) {
                        printf("Received response, id: %s \n", response["id"].get<std::string>().c_str());
                        response_callback(response["id"], response["result"]);
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
                    ws_client->connect(address);
                    continue;
                }
                ws_client->poll();
            }

        }

        void run_websocket()
        {
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

} // namespace dfe
