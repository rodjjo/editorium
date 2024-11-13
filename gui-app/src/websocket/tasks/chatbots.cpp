#include "windows/progress_ui.h"
#include "websocket/code.h"
#include "chatbots.h"

namespace editorium {
namespace ws {
namespace chatbots {

std::string chat_bot(const chatbot_request_t& request) {
    std::string result;

    json config;
    config["repo_id"] = request.repo_id;
    config["model_name"] = request.model_name;
    config["template"] = request.template_str;
    config["context"] = request.context;
    config["prompt"] = request.prompt;
    config["max_new_tokens"] = request.max_new_tokens;
    config["temperature"] = request.temperature;
    config["top_p"] = request.top_p;
    config["top_k"] = request.top_k;
    config["repetition_penalty"] = request.repetition_penalty;
    config["response_after"] = request.response_after;

    json inputs;
    enable_progress_window(progress_chatbot);
    auto response = execute("chatbot", inputs, config);

    if (response && response->texts.size() > 0) {
        result = response->texts[0];
    }

    return result;
}

std::string chat_bot_vision(const vision_chat_request_t& request) {
    std::string result;

    json config;
    config["repo_id"] = request.repo_id;
    config["prompt"] = request.prompt;
    config["system_prompt"] = request.system_prompt;
    config["temperature"] = request.temperature;

    api_payload_t payload;
    payload.images = {request.image};

    json inputs;
    inputs["default"] = to_input(payload);

    enable_progress_window(progress_chatbot_vision);
    auto response = execute("chatvision", inputs, config);

    if (response && response->texts.size() > 0) {
        result = response->texts[0];
    }
    
    return result;
}


} // namespace upscalers
} // namespace ws
} // namespace editorium

