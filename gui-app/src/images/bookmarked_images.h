#pragma once

#include <vector>
#include "images/image.h"

namespace editorium
{


image_ptr_t get_bookmarked_image(size_t index);
size_t get_bookmarked_images_count();
void add_bookmarked_image(const image_ptr_t& img);
void pin_bookmarked_image(size_t index);
void unpin_bookmarked_image(size_t index);
void clear_bookmarked_images();
void remove_bookmarked_image(RawImage *img);
    
} // namespace editorium
