#pragma once

#include <vector>
#include <string>
#include <set>
#include "images/image.h"

namespace editorium {
namespace ws {
namespace diffusion {

typedef std::pair<std::pair<std::string, float>, editorium::image_ptr_t> control_image_t;

typedef struct {
    std::string model_type;
    std::string model_name;
    std::string prompt;
    std::string negative_prompt;
    std::string scheduler;
    bool use_lcm = false;
    bool free_lunch = false;
    bool use_float16 = true;
    unsigned char steps = 4;
    float cfg = 3.0; 
    float image_strength = 0.75;
    int seed = -1;
    unsigned int width = 512;
    unsigned int height = 512;
    unsigned int batch_size = 1;
    float ipadapter_strength = 1.0;
    unsigned int mask_dilate_size = 3;
    unsigned int mask_blur_size = 3;
    std::string inpaint_mode = "original";
    std::vector<editorium::image_ptr_t> images;
    std::vector<editorium::image_ptr_t> masks;
    std::vector<control_image_t> controlnets;  // mode, scale, image
    std::vector<control_image_t> ip_adapters;  // mode, image
    std::vector<std::string> loras; // a string that contains lora name and strength separated by ':'
} diffusion_request_t;

typedef struct {
    int controlnet_count = 0;
    int ip_adapter_count = 0;
    std::set<std::string> controlnet_types;
    std::set<std::string> ip_adapter_types;
    bool support_inpaint = false;
} architecture_features_t;

std::vector<std::pair<std::string, std::string> > list_architectures();
std::vector<editorium::image_ptr_t> run_diffusion(const diffusion_request_t &request);
std::vector<editorium::image_ptr_t> run_preprocessor(const std::string& type, std::vector<editorium::image_ptr_t> images);
std::vector<editorium::image_ptr_t> run_seg_ground_dino(const std::string& tags, std::vector<editorium::image_ptr_t> images); 
std::vector<editorium::image_ptr_t> run_seg_sapiens(const std::string& tags, std::vector<editorium::image_ptr_t> images); 
architecture_features_t get_architecture_features(const std::string &architecture);

} // namespace diffusion
} // namespace ws
} // namespace editorium
