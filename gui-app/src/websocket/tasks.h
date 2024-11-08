#pragma once

#include "images/image.h"

namespace editorium
{
    namespace ws {
        image_ptr_t load_image(const std::string &path);
        void save_image(const std::string &path, image_ptr_t image, bool png_format);
    }
} // namespace editorium
