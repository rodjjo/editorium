#pragma once

#include <nlohmann/json.hpp>

namespace editorium {

using json = nlohmann::json;

json parse_workflow(const std::string &path);

}
