#pragma once

#include <vector>
#include <string>
#include "images/image.h"

namespace editorium
{
    namespace ws {
        image_ptr_t load_image(const std::string &path);
        void save_image(const std::string &path, image_ptr_t image, bool png_format);
        std::vector<std::string> list_models(const std::string& model_type, bool list_loras);
    }
} // namespace editorium
