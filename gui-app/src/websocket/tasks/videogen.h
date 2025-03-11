#pragma once

#include <vector>
#include <string>
#include <set>

#include "images/image.h"

namespace editorium {
namespace ws {
namespace video_gen {

typedef struct {
    std::string prompt;
    std::string negative_prompt;
    std::string lora_path;
    int lora_rank = 128;
    int num_inference_steps = 50;
    float guidance_scale = 5.0;
    int num_videos_per_prompt = 1;
    int seed = -1;
    float strength = 0.8;
    int width = 704;
    int height = 480;
    int num_frames = 121;
    int frame_rate = 25;
    std::string stg_skip_layers = "19";
    std::string stg_mode = "attention_values";
    float stg_scale = 1.0;
    float stg_rescale = 0.7;
    float image_cond_noise_scale = 0.15;
    float decode_timestep = 0.025;
    float decode_noise_scale = 0.0125;

    std::string save_path;
    editorium::image_ptr_t first_frame;
    editorium::image_ptr_t last_frame;
} ltx_video_gen_request_t;

bool run_ltx_video_gen(const ltx_video_gen_request_t &request);

}
}
}
