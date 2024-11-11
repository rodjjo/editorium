#include <FL/fl_ask.H>
#include <nlohmann/json.hpp>

#include "websocket/code.h"
#include "websocket/tasks/diffusion.h"
#include "windows/progress_ui.h"
#include "misc/config.h"

namespace editorium {
namespace ws {
namespace diffusion {


architecture_features_t get_architecture_features(const std::string &architecture) {
    architecture_features_t result;
    if (architecture == "sd15") {
        result.controlnet_count = 4;
        result.ip_adapter_count = 2;
        result.controlnet_types = {"canny", "depth", "pose", "scribble", "segmentation", "lineart", "mangaline", "inpaint"};
        result.ip_adapter_types = {"plus-face", "full-face", "plus", "common", "light", "vit"};
        result.support_inpaint = true;
    } else if (architecture == "sdxl") {
        result.controlnet_count = 1;
        result.ip_adapter_count = 2;
        result.controlnet_types = {"pose", "canny", "depth"};
        result.ip_adapter_types = {"plus-face", "plus", "common"};
        result.support_inpaint = true;
    } else if (architecture == "flux") {
        result.controlnet_count = 1;
        result.ip_adapter_count = 2;
        result.controlnet_types = {"pose", "canny", "depth"};
        result.ip_adapter_types = {};
        result.support_inpaint = true;
    } else if (architecture == "sd35") {
        result.controlnet_count = 0;
        result.ip_adapter_count = 0;
        result.controlnet_types = {};
        result.ip_adapter_types = {};
        result.support_inpaint = true;
    } else if (architecture == "omnigen") {
        result.controlnet_count = 0;
        result.ip_adapter_count = 0;
        result.controlnet_types = {};
        result.ip_adapter_types = {};
        result.support_inpaint = false;
    }
    return result;

}


std::vector<std::pair<std::string, std::string> > list_architectures() {
    std::vector<std::pair<std::string, std::string> > result;
    result = {
        {"sd15", "Stable Diffusion 1.5"},
        {"sdxl", "Stable Diffusion XL"},
        {"sd35", "Stable Diffusion 3.5"},
        {"flux", "Flux 1.0"},
        {"omnigen", "Omnigen"},
    };
    return result;
}   

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
    loras.texts = request.loras;
    
    json inputs;
    if (request.images.size() > 0) {
        inputs["image"] = to_input(images);
    }
    if (request.masks.size() > 0) {
        inputs["mask"] = to_input(masks);
    }
    if (request.loras.size() > 0) {
        inputs["loras"] = to_input(loras);
    }

    char buffer[128] = "";
    for (size_t i = 0; i < 6 && i < request.ip_adapters.size(); i++) {
        sprintf(buffer, "ip_adapter_scale_%lu", i + 1);    
        config[buffer] = request.ip_adapters[i].first.second;

        json ip_adapter;
        ip_adapter["adapter_model"] = request.ip_adapters[i].first.first;
        ip_adapter["image"] = request.ip_adapters[i].second->toJson();
        json data;
        data["data"] = ip_adapter;

        sprintf(buffer, "adapter_%lu", i + 1);
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

std::pair<json, json> create_sdxl_diffusion_request(const diffusion_request_t &request) {
    std::pair<json, json> result;

    if (get_config()->sdxl_base_model().empty()) {
        fl_alert("%s", "Please set the base model for SDXL in the settings");
        return result;
    }

    json config;

    float lora_scale = 1.0;
    std::string lora_name;
    if (!request.loras.empty() && request.loras[0].size() < 1024) {
        char lora_name_cstr[1024];
        if (sscanf(request.loras[1].c_str(), "%s:%f", lora_name_cstr, &lora_scale) == 2) {
            lora_name = lora_name_cstr;
        }
    }

    float controlnet_scale = 1.0;
    image_ptr_t controlnet_image;
    std::string controlnet_type;
    if (!request.controlnets.empty()) {
        controlnet_scale = request.controlnets[0].first.second;
        controlnet_image = request.controlnets[0].second;
        controlnet_type = request.controlnets[0].first.first;
    }

    float ip_adapter_scale = 0.6;
    if (!request.ip_adapters.empty()) {
        ip_adapter_scale = request.ip_adapters[0].first.second;
    }

    config["prompt"] = request.prompt;
    config["negative_prompt"] = request.negative_prompt;
    config["model_name"] = get_config()->sdxl_base_model();
    config["cfg"] = request.cfg;
    config["height"] = request.height;
    config["width"] = request.width;
    config["steps"] = request.steps;
    config["seed"] = request.seed;
    config["inpaint_mode"] = request.inpaint_mode;
    config["mask_dilate_size"] = request.mask_dilate_size;
    config["mask_blur_size"] = request.mask_blur_size;
    config["unet_model"] = request.model_name;
    config["lora_repo_id"] = lora_name;
    config["lora_scale"] = lora_scale;
    config["controlnet_conditioning_scale"] = controlnet_scale;
    if (controlnet_image) {
        config["controlnet_type"] = controlnet_type;
    }
    config["strength"] = request.image_strength;
    config["ip_adapter_scale"] = ip_adapter_scale;

    api_payload_t images;
    images.images = request.images;
    api_payload_t masks;
    masks.images = request.masks;

    json inputs;
    if (request.images.size() > 0) {
        inputs["image"] = to_input(images);
    }
    if (request.masks.size() > 0) {
        inputs["mask"] = to_input(masks);
    }
    
    if (controlnet_image) {
        api_payload_t control_image;
        control_image.images = {controlnet_image};
        inputs["control_image"] = to_input(control_image);
    }

    for (size_t i = 0; i < 2 && i < request.ip_adapters.size(); i++) {
        json ip_adapter;
        ip_adapter["adapter_model"] = request.ip_adapters[i].first.first;
        ip_adapter["image"] = request.ip_adapters[i].second->toJson();
        json data;
        data["data"] = ip_adapter;

        char buffer[128] = "";
        sprintf(buffer, "ip_adapter_%lu", i + 1);
        inputs[buffer] = data;
    }

    result.first = inputs;
    result.second = config;

    return result;
}

std::pair<json, json> create_flux_diffusion_request(const diffusion_request_t &request) {
    std::pair<json, json> result;

    if (get_config()->flux_base_model().empty()) {
        fl_alert("%s", "Please set the base model for Flux in the settings");
        return result;
    }

    float lora_scale = 1.0;
    std::string lora_name;
    if (!request.loras.empty() && request.loras[0].size() < 1024) {
        char lora_name_cstr[1024];
        if (sscanf(request.loras[1].c_str(), "%s:%f", lora_name_cstr, &lora_scale) == 2) {
            lora_name = lora_name_cstr;
        }
    }

    float controlnet_scale = 1.0;
    image_ptr_t controlnet_image;
    std::string controlnet_type;
    if (!request.controlnets.empty()) {
        controlnet_scale = request.controlnets[0].first.second;
        controlnet_image = request.controlnets[0].second;
        controlnet_type = request.controlnets[0].first.first;
    }

    json config;

    config["prompt"] = request.prompt;
    config["model_name"] = get_config()->flux_base_model();
    config["cfg"] = request.cfg;
    config["height"] = request.height;
    config["width"] = request.width;
    config["steps"] = request.steps;
    config["max_sequence_length"] = 512;
    config["seed"] = request.seed;
    config["inpaint_mode"] = request.inpaint_mode;
    config["mask_dilate_size"] = request.mask_dilate_size;
    config["mask_blur_size"] = request.mask_blur_size;
    config["transformer2d_model"] = request.model_name;
    config["lora_repo_id"] = lora_name;
    config["lora_scale"] = lora_scale;

    if (controlnet_image) {
        config["controlnet_type"] = controlnet_type;
        config["controlnet_conditioning_scale"] = controlnet_scale;
    }
    
    api_payload_t images;
    images.images = request.images;
    api_payload_t masks;
    masks.images = request.masks;

    json inputs;
    if (request.images.size() > 0) {
        inputs["image"] = to_input(images);
    }
    if (request.masks.size() > 0) {
        inputs["mask"] = to_input(masks);
    }
    
    if (controlnet_image) {
        api_payload_t control_image;
        control_image.images = {controlnet_image};
        inputs["control_image"] = to_input(control_image);
    }

    result.first = inputs;
    result.second = config;

    return result;
}

std::pair<json, json> create_sd35_diffusion_request(const diffusion_request_t &request) {
    std::pair<json, json> result;

    if (get_config()->sd35_base_model().empty()) {
        fl_alert("%s", "Please set the base model for SD35 in the settings");
        return result;
    }

    json config;

    float lora_scale = 1.0;
    std::string lora_name;
    if (!request.loras.empty() && request.loras[0].size() < 1024) {
        char lora_name_cstr[1024];
        if (sscanf(request.loras[1].c_str(), "%s:%f", lora_name_cstr, &lora_scale) == 2) {
            lora_name = lora_name_cstr;
        }
    }

    /*
    float controlnet_scale = 1.0;
    image_ptr_t controlnet_image;
    std::string controlnet_type;
    if (!request.controlnets.empty()) {
        controlnet_scale = request.controlnets[0].first.second;
        controlnet_image = request.controlnets[0].second;
        controlnet_type = request.controlnets[0].first.first;
    }
    */

    config["prompt"] = request.prompt;
    config["model_name"] = get_config()->sd35_base_model();
    config["cfg"] = request.cfg;
    config["height"] = request.height;
    config["width"] = request.width;
    config["steps"] = request.steps;
    config["max_sequence_length"] = 512;
    config["seed"] = request.seed;
    config["inpaint_mode"] = request.inpaint_mode;
    config["mask_dilate_size"] = request.mask_dilate_size;
    config["mask_blur_size"] = request.mask_blur_size;
    config["transformer2d_model"] = request.model_name;
    config["lora_repo_id"] = lora_name;
    config["lora_scale"] = lora_scale;

    /*
    if (controlnet_image) {
        config["controlnet_type"] = controlnet_type;
        config["controlnet_conditioning_scale"] = controlnet_scale;
    }
    */

    api_payload_t images;
    images.images = request.images;
    api_payload_t masks;
    masks.images = request.masks;

    json inputs;
    if (request.images.size() > 0) {
        inputs["image"] = to_input(images);
    }
    if (request.masks.size() > 0) {
        inputs["mask"] = to_input(masks);
    }

    //if (controlnet_image) {
        //api_payload_t control_image;
        //control_image.images = {controlnet_image};
        //inputs["control_image"] = to_input(control_image);
    //}

    result.first = inputs;
    result.second = config;

    return result;
}

std::pair<json, json> create_omnigen_diffusion_request(const diffusion_request_t &request) {
    std::pair<json, json> result;

    json config;

    config["prompt"] = request.prompt;
    config["cfg"] = request.cfg;
    config["height"] = request.height;
    config["width"] = request.width;
    config["steps"] = request.steps;
    config["seed"] = request.seed;

    api_payload_t images;
    images.images = request.images;
    api_payload_t masks;
    masks.images = request.masks;

    json inputs;
    if (request.images.size() > 0) {
        inputs["image"] = to_input(images);
    }

    /*
    if (request.masks.size() > 0) {
        inputs["mask"] = to_input(masks);
    }
    */

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
    } else if (request.model_type == "sdxl") {
        task_name = "sdxl";
        request_data = create_sdxl_diffusion_request(request);
    } else if (request.model_type == "flux") {
        task_name = "flux";
        request_data = create_flux_diffusion_request(request);
    } else if (request.model_type == "sd35") {
        task_name = "sd35";
        request_data = create_sd35_diffusion_request(request);
    } else if (request.model_type == "omnigen") {
        task_name = "omnigen";
        request_data = create_omnigen_diffusion_request(request);
    }

    if (!task_name.empty()) {
        enable_progress_window(progress_generation);
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

    enable_progress_window(progress_preprocessor);
    auto response = execute("image-preprocessor", inputs, config);

    if (response) {
        result = response->images;
    }

    return result;
}

std::vector<editorium::image_ptr_t> run_seg_ground_dino(const std::string& tags, std::vector<editorium::image_ptr_t> images) {
    std::vector<editorium::image_ptr_t> result;

    json config;
    config["prompt"] = tags;
    config["model_name_segmentation"] = "facebook/sam-vit-base";
    config["model_name_detection"] = "IDEA-Research/grounding-dino-tiny";
    config["margin"] = 5;
    config["selection_type"] = "detected-square";

    api_payload_t payload;
    payload.images = images;

    json inputs;
    inputs["default"] = to_input(payload);

    enable_progress_window(progress_segmentation);
    auto response = execute("sam-dino-segmentation", inputs, config);

    if (response) {
        result = response->images;
        for (size_t i = 0; i < result.size(); i++) {
            result[i] = result[i]->black_white_into_rgba_mask();
        }
    }
    
    return result;
}

std::vector<editorium::image_ptr_t> run_seg_sapiens(const std::string& tags, std::vector<editorium::image_ptr_t> images) {
    std::vector<editorium::image_ptr_t> result;

    json config;
    config["classes"] = tags;
    config["margin"] = 5;
    config["selection_type"] = "detected-square";

    api_payload_t payload;
    payload.images = images;

    json inputs;
    inputs["default"] = to_input(payload);

    enable_progress_window(progress_segmentation);
    auto response = execute("sapiens-segmentation", inputs, config);

    if (response) {
        result = response->images;
        for (size_t i = 0; i < result.size(); i++) {
            result[i] = result[i]->black_white_into_rgba_mask();
        }
    }

    return result;
}

    
} // namespace models
} // namespace ws
} // namespace editorium
