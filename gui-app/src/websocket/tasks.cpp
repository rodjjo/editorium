#include <fstream>
#include "base64/base64.h"
#include "tasks.h"
#include "websocket/code.h"

namespace editorium
{
namespace ws {
    
    image_ptr_t load_image(const std::string &path) {
        printf("Loading image: %s\n", path.c_str());
        std::ifstream file(path, std::ios::binary);
        std::vector<char> buffer(std::istreambuf_iterator<char>(file), {});
        auto encoded = base64_encode((const unsigned char *)buffer.data(), buffer.size());
        
        api_payload_t payload;
        payload.texts = {encoded};
        auto inputs = to_input(payload);

        auto result = execute("base64image", inputs, json());

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

        auto result = execute("img2base64", inputs, config);

        if (!result || result->texts.empty()) {
            return;
        }

        std::string base64 = result->texts.front();
        auto decoded = base64_decode(base64.c_str(), base64.size());
        std::ofstream file(path, std::ios::binary);
        file.write((const char *)decoded.first.get(), decoded.second);
    }
}
} // namespace editorium
