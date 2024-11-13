#pragma once

#include <vector>
#include <string>
#include "images/image.h"

namespace editorium {
namespace ws {
namespace chatbots {

typedef struct {
    std::string context;
    std::string prompt;
} chatbot_request_t;

typedef struct {
    std::string prompt;
    std::string system_prompt;
    image_ptr_t image;
} vision_chat_request_t;

std::string chat_bot(const chatbot_request_t& request);
std::string chat_bot_vision(const vision_chat_request_t& request);

} // namespace chatbots
} // namespace ws
} // namespace editorium

