#include <string>
#include "images/image_palette.h"
#include "misc/dialogs.h"

namespace editorium
{

namespace {
    const int max_image_palette_count = 10;
    std::vector<std::pair<image_ptr_t, bool> > image_palette;
}

image_ptr_t get_image_palette(size_t index) {
    if (index < image_palette.size()) {
        return image_palette[index].first;
    }
    return nullptr;
}

size_t get_image_palette_count() {
    return image_palette.size();
}

void add_image_palette(const image_ptr_t& img) {
    if (image_palette.size() >= max_image_palette_count) {
        bool image_removed = false;
        // remove the first not pinned image
        for (size_t i = 0; i < image_palette.size(); i++) {
            if (!image_palette[i].second) {
                image_palette.erase(image_palette.begin() + i);
                image_removed = true;
                break;
            }
        }
        // if all images are pinned, remove the first one
        if (!image_removed) {
            image_palette.erase(image_palette.begin());
        }
    }
    image_palette.push_back(std::make_pair(img, false));
    show_info("You added an image to the image palette");
}

void pin_image_palette(size_t index) {
    size_t pinned_count = 0;
    // only bookmark max_image_palette_count - 1 images
    for (size_t i = 0; i < image_palette.size(); i++) {
        if (image_palette[i].second) {
            pinned_count++;
        }
    }
    if (pinned_count < max_image_palette_count - 1 && index < image_palette.size()) {
        image_palette[index].second = true;
    } else if (pinned_count >= max_image_palette_count - 1) {
        std::string message = "You can pin only " + std::to_string(max_image_palette_count - 1) + " images";
        show_error(message.c_str());
    }
}

void unpin_image_palette(size_t index) {
    if (index < image_palette.size()) {
        image_palette[index].second = false;
    }
}

void clear_image_palette() {
    image_palette.clear();
}

bool is_pinned_at_image_palette(size_t index) {
    if (index < image_palette.size()) {
        return image_palette[index].second;
    }
    return false;
}

void remove_image_palette(RawImage *img) {
    for (size_t i = 0; i < image_palette.size(); i++) {
        if (image_palette[i].first.get() == img) {
            image_palette.erase(image_palette.begin() + i);
            break;
        }
    }
}
    
} // namespace editorium
