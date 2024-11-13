#pragma once

#include <vector>
#include <string>
#include "images/image.h"

namespace editorium {
namespace ws {
namespace chatbots {

typedef struct {
    std::string repo_id = "TheBloke/Nous-Hermes-13B-GPTQ";
    std::string model_name = "model";
    std::string template_str = "### Instruction:\n{context}\n### Input:\n{input}\n### Response:\n";
    std::string context;
    std::string prompt;
    int max_new_tokens = 512;
    float temperature = 1;
    float top_p = 1;
    int top_k = 0;
    float repetition_penalty = 1;
    std::string response_after = "";
} chatbot_request_t;

typedef struct {
    std::string repo_id = "openbmb/MiniCPM-Llama3-V-2_5-int4";
    std::string prompt;
    std::string system_prompt;
    float temperature = 0.7;
    image_ptr_t image;
} vision_chat_request_t;

std::string chat_bot(const chatbot_request_t& request);
std::string chat_bot_vision(const vision_chat_request_t& request);

} // namespace chatbots
} // namespace ws
} // namespace editorium

