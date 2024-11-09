#include <nlohmann/json.hpp>

#include "websocket/code.h"
#include "websocket/tasks/diffusion.h"



namespace editorium {
namespace ws {
namespace diffusion {

std::pair<json, json> create_sd15_diffusion_request(const diffusion_request_t &request) {
    std::pair<json, json> result;

    json config;
    config["model_name"] = request.model_name;
    config["prompt"] = request.prompt;
    config["negative_prompt"] = request.negative_prompt;
    config["use_lcm"] = request.use_lcm;
    config["scheduler_name"] = "EulerAncestralDiscreteScheduler";
    config["use_float16"] = request.use_float16;
    config["free_lunch"] = request.free_lunch;
    config["seed"] = request.seed;
    config["cfg"] = request.cfg;
    config["steps"] = request.steps;
    config["width"] = request.width;
    config["height"] = request.height;
    config["strength"] = request.image_strength;
    config["batch_size"] = request.batch_size;
    config["inpaint_mode"] = request.inpaint_mode;
    config["mask_dilate_size"] = request.mask_dilate_size;
    config["mask_blur_size"] = request.mask_blur_size;

    api_payload_t images;
    images.images = request.images;
    api_payload_t masks;
    masks.images = request.masks;
    api_payload_t loras;

    char buffer[128] = "";
    for (size_t i = 0; i < request.loras.size(); i++) {
        sprintf(buffer, ":%0.3f", request.loras[i].first);
        loras.texts.push_back(request.loras[i].second + buffer);
    }
    
    json inputs;
    
    inputs["images"] = to_input(images);
    inputs["masks"] = to_input(masks);
    inputs["loras"] = to_input(loras);
    
    for (size_t i = 0; i < 6 && i < request.ip_adapters.size(); i++) {
        sprintf(buffer, "ip_adapter_scale_%lu", i + 1);    
        config[buffer] = request.ip_adapters[i].first.second;

        json ip_adapter;
        ip_adapter["adapter_model"] = request.ip_adapters[i].first.first;
        ip_adapter["image"] = request.ip_adapters[i].second->toJson();
        json data;
        data["data"] = ip_adapter;

        sprintf(buffer, "ip_adapter_%lu", i + 1);
        inputs[buffer] = data;
    }
    
    for (size_t i = 0; i < 5 && i < request.controlnets.size(); i++) {
        json controlnet;
        controlnet["repo_id"] = "";
        controlnet["control_type"] = request.controlnets[i].first.first;
        controlnet["strength"] = request.controlnets[i].first.second;
        controlnet["image"] = request.controlnets[i].second->toJson();
        json data;
        data["data"] = controlnet;
        sprintf(buffer, "controlnet_%lu", i + 1);
        inputs[buffer] = data;
    }

    result.first = inputs;
    result.second = config;
    
    return result;
}

std::vector<editorium::image_ptr_t> run_diffusion(const diffusion_request_t &request) {
    std::vector<editorium::image_ptr_t> result;

    std::string task_name;
    std::pair<json, json> request_data;

    if (request.model_type == "sd15") {
        task_name = "sd15";
        request_data = create_sd15_diffusion_request(request);
    }

    if (!task_name.empty()) {
        auto response = execute(task_name, request_data.first, request_data.second);
        if (response) {
            result = response->images;
        }
    }

    return result;
}


std::vector<editorium::image_ptr_t> run_preprocessor(const std::string& type, std::vector<editorium::image_ptr_t> images) {
    std::vector<editorium::image_ptr_t> result;

    api_payload_t payload;
    payload.images = images;

    json config;
    config["control_type"] = type;

    json inputs;
    inputs["default"]= to_input(payload);

    auto response = execute(type, inputs, config);

    if (response) {
        result = response->images;
    }

    return result;
}

    
} // namespace models
} // namespace ws
} // namespace editorium
