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
            listener_t listener;
        }

        void response_callback(const json& response) {
            if (!response.contains("id") || !response.contains("images") || !response.contains("texts") || !response.contains("boxes")) {
                return;
            }
            std::string id = response["id"];
            api_payload_t payload;
            payload.images = newImageList(response["images"]);
            payload.texts = response["texts"];
            for (const auto & item : response["boxes"]) {
                box_t box;
                box.x = item["x"];
                box.y = item["y"];
                box.x2 = item["x2"];
                box.y2 = item["y2"];
                payload.boxes.push_back(box);
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

        std::shared_ptr<api_payload_t> execute(const std::string& task_type, const json &inputs, const json &config, bool &canceled_checker) {
            std::shared_ptr<api_payload_t> result;
            auto callback_send = [task_type, inputs, config, &result, &canceled_checker]() {
                json req;
                auto uuid = UUID::generate_uuid();
                req["task_type"] = task_type;
                req["input"] = inputs;
                req["config"] = config;
                req["id"] = uuid;
                std::atomic<bool> result_set = false;

                { // scope to set the listener
                    std::unique_lock<std::mutex> lk(listener_mutex);
                    listener = [uuid, &result, &result_set] (const std::string& id, const api_payload_t & response) {
                        if (id == uuid) {
                            result = std::make_shared<api_payload_t>(response);
                            result_set.store(true);
                        }
                    };
                }

                ws_client->send(req.dump());
                while (!canceled_checker) {
                    if (result_set.load() || !ws_client->available()) {
                        break;
                    }
                    ws_client->poll();
                }

                { // scope to reset the listener
                    std::unique_lock<std::mutex> lk(listener_mutex);
                    listener = listener_t();
                }
            };

            wait_callback();
            replace_callback(callback_send);
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
                ws_client->poll();
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

        void run_websocket()
        {
            if (ws_thread) {
                return;
            }
            ws_thread.reset(new std::thread([] () {
                try {
                    std::string address = "ws://localhost:5001/";
                    ws_client.reset(new WebsocketsClient());
                    ws_client->connect(address);

                    ws_client->onMessage([&](WebsocketsClient&, WebsocketsMessage message){
                        json response = json::parse(message.data());
                        response_callback(response);
                    });

                    if (ws_client->available()) {
                        puts("Connected to websocket server!");
                    }

                    running = true;
                    while (running) {
                        if (!ws_client->available()) {
                            puts("Client not connected! trying to reconnect...");
                            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                            ws_client->connect(address);
                            continue;
                        }
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
             if (ws_thread) {
                puts("Stopping websocket...");
                running = false;
                ws_thread->join();
                puts("websocket stopped...");
            }
        }
    } // namespace py

} // namespace dfe
