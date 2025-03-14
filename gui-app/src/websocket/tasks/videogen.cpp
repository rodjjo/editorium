#include <FL/fl_ask.H>
#include <nlohmann/json.hpp>

#include "websocket/code.h"
#include "websocket/tasks/videogen.h"
#include "windows/progress_ui.h"
#include "misc/config.h"



namespace editorium {
namespace ws {
namespace video_gen {


std::pair<json, json> create_ltx_video_gen_request(const ltx_video_gen_request_t &request) {
    std::pair<json, json> result;

    int seed = (rand() % 1000000) + 1;
    if (request.seed >= 0) {
        seed = request.seed;
    }

    json config;
    config["prompt"] = request.prompt;
    config["negative_prompt"] = request.negative_prompt;
    config["lora_path"] = request.lora_path;
    config["lora_rank"] = request.lora_rank;
    config["num_inference_steps"] = request.num_inference_steps;
    config["guidance_scale"] = request.guidance_scale;
    config["num_videos_per_prompt"] = request.num_videos_per_prompt;
    config["seed"] = seed;
    config["width"] = request.width;
    config["height"] = request.height;
    config["num_frames"] = request.num_frames;
    config["frame_rate"] = request.frame_rate;
    config["strength"] = request.strength;
    config["intermediate_start"] = request.intermediate_start;
    config["intermediate_strength"] = request.intermediate_strength;
    config["stg_skip_layers"] = request.stg_skip_layers;
    config["stg_mode"] = request.stg_mode;
    config["stg_scale"] = request.stg_scale;
    config["stg_rescale"] = request.stg_rescale;
    config["image_cond_noise_scale"] = request.image_cond_noise_scale;
    config["decode_timestep"] = request.decode_timestep;
    config["decode_noise_scale"] = request.decode_noise_scale;
    config["save_path"] = request.save_path;
 

    api_payload_t first_frame_images;
    if (request.first_frame) {
        first_frame_images.images.push_back(request.first_frame);
    }
    api_payload_t last_frame_images;
    if (request.last_frame) {
        last_frame_images.images.push_back(request.last_frame);
    }
    api_payload_t intermediate_frame_images;
    if (!request.intermediate_frames.empty()) {
        intermediate_frame_images.images = std::vector<editorium::image_ptr_t>(request.intermediate_frames.begin(), request.intermediate_frames.end());
    }
    
    json inputs;
    if (first_frame_images.images.size() > 0) {
        inputs["first_frame"] = to_input(first_frame_images);
    }
    if (last_frame_images.images.size() > 0) {
        inputs["last_frame"] = to_input(last_frame_images);
    }
    if (intermediate_frame_images.images.size() > 0) {
        inputs["middle_frames"] = to_input(intermediate_frame_images);
    }

    result.first = inputs;
    result.second = config;
    
    return result;
}
    
bool run_ltx_video_gen(const ltx_video_gen_request_t &request) {
    std::pair<json, json> request_data = create_ltx_video_gen_request(request);
    enable_progress_window(progress_generation_video);
    execute("ltxvideo", request_data.first, request_data.second);
    return !should_cancel();
}

}
}
}