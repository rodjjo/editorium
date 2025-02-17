#pragma once

#include <vector>
#include <memory>
#include <string>
#include <stdint.h>

#include <nlohmann/json.hpp>

namespace editorium {

using json = nlohmann::json;

typedef enum {
    img_gray_8bit,
    img_rgb,
    img_rgba,
    // keep img_format_count at the end
    img_format_count
} image_format_t;


class RawImage;
typedef std::shared_ptr<RawImage> image_ptr_t;

class RawImage {
 public:
    RawImage(const unsigned char *buffer, uint32_t w, uint32_t h, image_format_t format, bool fill_transparent=true);
    virtual ~RawImage();
    const unsigned char *buffer();
    image_format_t format();
    uint32_t h();
    uint32_t w();
    size_t getVersion();
    void incVersion();
    void pasteFill(RawImage *image);
    void pasteFrom(int x, int y, float zoom, RawImage *image);
    void pasteAtClearFirst(int x, int y, RawImage *image);
    void pasteAt(int x, int y, RawImage *image);
    void pasteAt(int x, int y, RawImage *mask, RawImage *image);
    void pasteAt(int x, int y, int w, int h, RawImage *image);
    void fuseAt(int x, int y, RawImage *image);
    image_ptr_t pasteAtNoBackground(int x, int y, RawImage *mask, RawImage *image);
    void pasteInvertMask(RawImage *image);
    void rectangle(int x, int y, int w, int h, uint8_t color[4], float fill_opacity=0);
    image_ptr_t duplicate();
    image_ptr_t removeBackground(bool white);
    image_ptr_t black_white_into_rgba_mask();
    image_ptr_t rgba_mask_into_black_white(bool invert_colors=false);
    image_ptr_t to_rgb_mask();
    image_ptr_t to_rgba();
    image_ptr_t to_rgb();
    image_ptr_t resizeCanvas(uint32_t x, uint32_t y);
    image_ptr_t resizeImage(uint32_t x, uint32_t y);
    image_ptr_t resizeImage(uint32_t size);
    image_ptr_t resizeInTheCenter(uint32_t x, uint32_t y);
    image_ptr_t getCrop(uint32_t x, uint32_t y, uint32_t w, uint32_t h);
    image_ptr_t fit1024();
    image_ptr_t resizeLeft(int value);
    image_ptr_t resizeRight(int value);
    image_ptr_t resizeTop(int value);
    image_ptr_t resizeBottom(int value);
    image_ptr_t blur(int size);
    image_ptr_t erode(int size);
    image_ptr_t dilate(int size);
    image_ptr_t flip(bool vertically);
    image_ptr_t resize_min_area_using_alpha();
    image_ptr_t rotate();
    image_ptr_t invert_mask();
    image_ptr_t negative_mask();
    image_ptr_t create_mask_from_alpha_channel();
    image_ptr_t create_modification_mask(RawImage *modified_image);
    json toJson();
    
    void clear(uint8_t r, uint8_t g, uint8_t b, uint8_t a);

    bool getColor(int x, int y, uint8_t *r, uint8_t *g, uint8_t *b, uint8_t *a);

    void drawCircleColor(int x, int y, int radius, uint8_t color[4], uint8_t bgcolor[4], bool clear);
    void drawCircle(int x, int y, int radius, bool clear);
    void fillWithMask(int x, int y, RawImage *mask);
    void clear_pixels(uint32_t x, uint32_t y, uint32_t w, uint32_t h);

 private:
    unsigned char *buffer_;
    size_t buffer_len_;
    uint32_t w_;
    uint32_t h_;
    image_format_t format_;
    size_t version_;
};


image_ptr_t newImage(uint32_t w, uint32_t h, bool enable_alpha);
image_ptr_t newImage(const json& value);
std::vector<image_ptr_t> newImageList(const json& value);

typedef image_ptr_t image_ptr_t;
typedef RawImage RawImage;

}  // namespace editorium
