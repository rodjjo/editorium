#pragma once

#include <vector>
#include <string>
#include "images/image.h"

namespace editorium {
namespace ws {
namespace models {
    std::vector<std::string> list_models(const std::string& model_type, bool list_loras);
} // namespace models
} // namespace ws
} // namespace editorium
