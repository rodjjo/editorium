#pragma once

namespace editorium {
namespace ws {
namespace upscalers {

std::vector<editorium::image_ptr_t> upscale_gfpgan(float scale, float face_weight, bool restore_background, std::vector<editorium::image_ptr_t> images);

} // namespace upscalers
} // namespace ws
} // namespace editorium
