#include "windows/progress_ui.h"
#include "websocket/code.h"
#include "misc/config.h"
#include "chatbots.h"

namespace editorium {
namespace ws {
namespace chatbots {

std::string chat_bot(const chatbot_request_t& request) {
    std::string result;

    json config;
    config["repo_id"] = get_config()->chat_bot_repo_id();
    config["model_name"] = get_config()->chat_bot_model_name();
    config["template"] = get_config()->chat_bot_template();
    config["context"] = request.context;
    config["prompt"] = request.prompt;
    config["max_new_tokens"] = get_config()->chat_bot_max_new_tokens();
    config["temperature"] = get_config()->chat_bot_temperature();
    config["top_p"] = get_config()->chat_bot_top_p();
    config["top_k"] = get_config()->chat_bot_top_k();
    config["repetition_penalty"] = get_config()->chat_bot_repetition_penalty();
    config["response_after"] = get_config()->chat_bot_response_after();

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
    config["repo_id"] = get_config()->chat_vision_repo_id();
    config["prompt"] = request.prompt;
    config["system_prompt"] = request.system_prompt;
    config["temperature"] = get_config()->chat_vision_temperature();

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

