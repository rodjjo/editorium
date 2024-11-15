#include <string>
#include "images/bookmarked_images.h"
#include "misc/dialogs.h"

namespace editorium
{

namespace {
    const int max_bookmarked_images = 10;
    std::vector<std::pair<image_ptr_t, bool> > bookmarked_images;
}

image_ptr_t get_bookmarked_image(size_t index) {
    if (index < bookmarked_images.size()) {
        return bookmarked_images[index].first;
    }
    return nullptr;
}

size_t get_bookmarked_images_count() {
    return bookmarked_images.size();
}

void add_bookmarked_image(const image_ptr_t& img) {
    if (bookmarked_images.size() >= max_bookmarked_images) {
        bool image_removed = false;
        // remove the first not pinned image
        for (size_t i = 0; i < bookmarked_images.size(); i++) {
            if (!bookmarked_images[i].second) {
                bookmarked_images.erase(bookmarked_images.begin() + i);
                image_removed = true;
                break;
            }
        }
        // if all images are pinned, remove the first one
        if (!image_removed) {
            bookmarked_images.erase(bookmarked_images.begin());
        }
    }
    bookmarked_images.push_back(std::make_pair(img, false));
}

void pin_bookmarked_image(size_t index) {
    size_t pinned_count = 0;
    // only bookmark max_bookmarked_images - 1 images
    for (size_t i = 0; i < bookmarked_images.size(); i++) {
        if (bookmarked_images[i].second) {
            pinned_count++;
        }
    }
    if (pinned_count < max_bookmarked_images - 1 && index < bookmarked_images.size()) {
        bookmarked_images[index].second = true;
    } else if (pinned_count >= max_bookmarked_images - 1) {
        std::string message = "You can pin only " + std::to_string(max_bookmarked_images - 1) + " images";
        show_error(message.c_str());
    }
}

void unpin_bookmarked_image(size_t index) {
    if (index < bookmarked_images.size()) {
        bookmarked_images[index].second = false;
    }
}

void clear_bookmarked_images() {
    bookmarked_images.clear();
}

void remove_bookmarked_image(RawImage *img) {
    for (size_t i = 0; i < bookmarked_images.size(); i++) {
        if (bookmarked_images[i].first.get() == img) {
            bookmarked_images.erase(bookmarked_images.begin() + i);
            break;
        }
    }
}
    
} // namespace editorium
