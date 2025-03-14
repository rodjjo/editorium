#include <fstream>
#include "base64/base64.h"
#include "websocket/tasks/filesystem.h"
#include "websocket/code.h"
#include "windows/progress_ui.h"
#include "video/video.h"

namespace editorium {
namespace ws {
namespace filesystem {
    image_ptr_t load_image(const std::string &path) {
        /*
        // vs::open_file is corrupting memory, maybe;

        auto video = vs::open_file(path.c_str());
        if (video->error() || !video->buffer()) {
            return nullptr;
        }
        return std::make_shared<RawImage>(video->buffer(), video->w(), video->h(), img_rgb, false); */
        std::ifstream file(path, std::ios::binary);
        std::vector<char> buffer(std::istreambuf_iterator<char>(file), {});
        auto encoded = base64_encode((const unsigned char *)buffer.data(), buffer.size());
        
        api_payload_t payload;
        payload.texts = {encoded};
        auto inputs = to_input(payload);
        json default_input;
        default_input["default"] = inputs;
        enable_progress_window(progress_opening_file);
        auto result = execute("base64image", default_input, json());

        if (!result || result->images.empty()) {
            return nullptr;
        }

        return result->images.front();
    }

    void save_image(const std::string &path, image_ptr_t image, bool png_format) {
        printf("Saving image: %s\n", path.c_str());
        api_payload_t payload;
        payload.images.push_back(image);
        auto inputs = to_input(payload);

        json config;
        config["png_format"] = png_format;
        json default_input;
        default_input["default"] = inputs;
        enable_progress_window(progress_saving_file);
        auto result = execute("image2base64", default_input, config);

        if (!result || result->texts.empty()) {
            return;
        }

        std::string base64 = result->texts.front();
        auto decoded = base64_decode(base64.c_str(), base64.size());
        std::ofstream file(path, std::ios::binary);
        file.write((const char *)decoded.first.get(), decoded.second);
    }

    std::list<image_ptr_t> grab_frames(const std::string &path, size_t max_frames) {
        auto video = vs::open_file(path.c_str());
        if (video->error() || !video->buffer()) {
            return {};
        }
        std::list<image_ptr_t> result;
        size_t current_count = 0;
        while (current_count < max_frames) {
            auto img = std::make_shared<RawImage>(video->buffer(), video->w(), video->h(), img_rgb, false);
            result.push_back(img);
            current_count++;
            if (video->position() + 1 >= video->count()) {
                break;
            }
            video->next();
        }
        
        return result;
    }

} // namespace filesystem
} // namespace ws
} // namespace editorium
