#pragma once

#include <vector>
#include "images/image.h"

namespace editorium {
namespace ws {
namespace upscalers {

std::vector<editorium::image_ptr_t> upscale_gfpgan(float scale, float face_weight, bool restore_background, std::vector<editorium::image_ptr_t> images);
std::vector<editorium::image_ptr_t> correct_colors(std::vector<editorium::image_ptr_t> images, std::vector<editorium::image_ptr_t> originals); 

} // namespace upscalers
} // namespace ws
} // namespace editorium
