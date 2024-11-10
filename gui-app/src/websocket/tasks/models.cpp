#include <fstream>
#include "base64/base64.h"
#include "websocket/tasks/models.h"
#include "websocket/code.h"
#include "windows/progress_ui.h"

namespace editorium
{
namespace ws {
namespace models {

    std::vector<std::string> list_models(const std::string& model_type, bool list_loras) {
        json config;
        config["list_lora"] = list_loras;
        config["model_type"] = model_type;
        enable_progress_window(progress_list_models);
        auto result = execute("list-models", json(), config);
        if (!result) {
            return std::vector<std::string>();
        }
        return result->texts;
    }

} // namespace models
} // namespace ws
} // namespace editorium
