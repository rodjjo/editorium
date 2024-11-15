#pragma once

#include <vector>
#include "images/image.h"

namespace editorium
{


image_ptr_t get_image_palette(size_t index);
size_t get_image_palette_count();
void add_image_palette(const image_ptr_t& img);
void pin_image_palette(size_t index);
void unpin_image_palette(size_t index);
bool is_pinned_at_image_palette(size_t index);
void clear_image_palette();
void remove_image_palette(RawImage *img);
    
} // namespace editorium
