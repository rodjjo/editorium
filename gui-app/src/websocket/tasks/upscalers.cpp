#include <fstream>
#include "base64/base64.h"
#include "websocket/tasks/models.h"
#include "websocket/code.h"
#include "windows/progress_ui.h"
#include "upscalers.h"


namespace editorium {
namespace ws {
namespace upscalers {

std::vector<editorium::image_ptr_t> upscale_gfpgan(float scale, float face_weight, bool restore_background, std::vector<editorium::image_ptr_t> images) {
    std::vector<editorium::image_ptr_t> result;

    json config;
    config["scale"] = scale;
    config["face_weight"] = face_weight;
    config["restore_background"] = restore_background;

    api_payload_t payload;
    payload.images = images;

    json inputs;
    inputs["default"] = to_input(payload);

    enable_progress_window(progress_upscaler);
    auto response = execute("gfpgan-upscaler", inputs, config);

    if (response) {
        result = response->images;
    }

    return result;
}

std::vector<editorium::image_ptr_t> correct_colors(std::vector<editorium::image_ptr_t> images, std::vector<editorium::image_ptr_t> originals) {
    std::vector<editorium::image_ptr_t> result;

    json config;

    api_payload_t payload;
    payload.images = images;

    json inputs;
    inputs["default"] = to_input(payload);
    payload.images = originals;
    inputs["original"] = to_input(payload);

    enable_progress_window(progress_correct_colors);
    auto response = execute("correct-colors", inputs, config);

    if (response) {
        result = response->images;
    }

    return result;
}



} // namespace upscalers
} // namespace ws
} // namespace editorium

