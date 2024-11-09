#pragma once

#include <vector>
#include <string>
#include "images/image.h"

namespace editorium {
namespace ws {
namespace filesystem {
    image_ptr_t load_image(const std::string &path);
    void save_image(const std::string &path, image_ptr_t image, bool png_format);
} // namespace filesystem
} // namespace ws
} // namespace editorium
