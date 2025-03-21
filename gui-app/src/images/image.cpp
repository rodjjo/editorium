#include <vector>
#include <stdio.h>
#include <string>
#include <exception>

#include <CImg.h>

#include "base64/base64.h"
#include "images/image.h"

using namespace cimg_library;

namespace editorium {

namespace {
    const int format_channels[img_format_count] = {
        1,  // img_gray_8bit
        3,  // img_rgb
        4  // img_rgba
    };

    uint8_t white_color_rgba[4] {
        255, 255, 255, 255
    };

    uint8_t black_color_rgba[4] {
        0, 0, 0, 255
    };
    
    uint8_t no_color_rgba[4] {
        0, 0, 0, 0
    };
}  // unnamed namespace

RawImage::RawImage(const unsigned char *buffer, uint32_t w, uint32_t h, image_format_t format, bool fill_transparent) {
    format_ = format;
    buffer_len_ = w * h;
    w_ = w;
    h_ = h;
    switch (format_) {
        case img_rgb:
            buffer_len_ *= 3;
            break;
        case img_rgba:
            buffer_len_ *= 4;
            break;
    }
    buffer_ = (unsigned char *)malloc(buffer_len_);
    if (buffer) {
        memcpy(buffer_, buffer, buffer_len_);
    } else {
        if (format_ == img_rgba && fill_transparent) {
            memset(buffer_, 0, buffer_len_);
        } else {
            memset(buffer_, 255, buffer_len_);
        }
    }
    version_ = (size_t) buffer_; // randomize the version
}

RawImage::~RawImage() {
    free(buffer_);
}

const unsigned char *RawImage::buffer() {
    return buffer_;
}

image_format_t RawImage::format() {
    return format_;
}

uint32_t RawImage::h() {
    return h_;
}

uint32_t RawImage::w() {
    return w_;
}

size_t RawImage::getVersion() {
    return version_;
}

void RawImage::incVersion() {
    ++version_;
}

image_ptr_t RawImage::duplicate() {
    return std::make_shared<RawImage>(
        buffer_, w_, h_, format_
    );
}

image_ptr_t RawImage::black_white_into_rgba_mask() {
    image_ptr_t result;
    result.reset(new RawImage(NULL, w_, h_, img_rgba, false));
    
    int src_channels = format_channels[format_];

    unsigned char *d = result->buffer_;
    unsigned char *s = this->buffer_;
    if (src_channels == 4) { // if source is rgba, use the alpha channel instead of red channel
        s = s + 3;  
    }
    unsigned char color = 0;
    for (int i = 0; i < result->buffer_len_; i += 4) {
        color = *s ? 255 : 0;
        *d = color; ++d;
        *d = color; ++d;
        *d = color; ++d;
        *d = color; s += src_channels;
        ++d;
    }

    return result;
}

image_ptr_t RawImage::rgba_mask_into_black_white(bool invert) {
    image_ptr_t result;
    result.reset(new RawImage(NULL, w_, h_, img_rgb, false));
    
    int src_channels = format_channels[format_];

    unsigned char *d = result->buffer_;
    unsigned char *s = this->buffer_;
    
    if (src_channels == 4) { // if source is rgba, use the alpha channel instead of red channel
        s = s + 3;  
    }

    unsigned char color = 0;
    unsigned char color_1 = invert ?  0 : 255;
    unsigned char color_2 = invert ?  255 : 0;
    for (size_t i = 0; i < result->buffer_len_; i += 3) {
        color = *s ? color_1 : color_2;
        *d = color; ++d;
        *d = color; ++d;
        *d = color; ++d;
        s += src_channels;
    }

    return result;
}

image_ptr_t RawImage::removeBackground(bool white) {
    image_ptr_t r;
    r.reset(new RawImage(NULL, w_, h_, img_rgba, false));
    int src_channels = format_channels[format_];
  
    CImg<unsigned char> src(buffer_, src_channels, w_, h_, 1, true);
    CImg<unsigned char> img(r->buffer_, 4, w_, h_, 1, true);
    
    src.permute_axes("yzcx");
    img.permute_axes("yzcx");
    
    img.draw_image(0, 0, src);
    
    if (white) {
        img.fill("if(i0>=255&&i1==255&&i2==255,0,i)", true);
    } else {
        img.fill("if(i0!=0||i1!=0||i2!=0,i,0)", true);
    }
   
    img.permute_axes("cxyz");
    src.permute_axes("cxyz");

    return r;
}

image_ptr_t RawImage::invert_mask() {
    image_ptr_t r;
    r.reset(new RawImage(NULL, w_, h_, img_rgba, false));
    int src_channels = format_channels[format_];
  
    CImg<unsigned char> src(buffer_, src_channels, w_, h_, 1, true);
    CImg<unsigned char> img(r->buffer_, 4, w_, h_, 1, true);
    
    src.permute_axes("yzcx");
    img.permute_axes("yzcx");
    img.draw_image(0, 0, src);
    img.fill("if(i0>=255&&i1==255&&i2==255,0,255)", true);
    img.permute_axes("cxyz");
    src.permute_axes("cxyz");

    return r;
}

image_ptr_t RawImage::to_rgb_mask() {
    image_ptr_t r;

    r.reset(new RawImage(NULL, w_, h_, img_rgb));

    int src_channels = format_channels[format_];
   
    CImg<unsigned char> src(buffer_, src_channels, w_, h_, 1, true);
    CImg<unsigned char> img(r->buffer_, 3, w_, h_, 1, true);
    CImg<unsigned char> tmp(src, false);

    tmp.permute_axes("yzcx");
    img.permute_axes("yzcx");
    if (src_channels == 4) {
        tmp.fill("if(i3==0,255,i)", true); // remove the transparency
    }
    img.draw_image(0, 0, tmp); 
    // replace white per black and black per white
    img.fill("if((i0==i1&&i0==i2)&&(i0==255||i0==0),if(i0==0,255,0),i)", true);
    img.permute_axes("cxyz");

    return r;
}

void RawImage::drawCircleColor(int x, int y, int radius, uint8_t color[4], uint8_t bgcolor[4], bool clear) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= w_) x = w_ - 1;
    if (y >= h_) y = h_ - 1;
    int src_channels = format_channels[format_];
    CImg<unsigned char> img(buffer_, src_channels, w_, h_, 1, true);
    img.permute_axes("yzcx");
    if (clear) {
        img.draw_circle(x, y, radius, bgcolor);
    } else {
        img.draw_circle(x, y, radius, color);
    }
    img.permute_axes("cxyz");
    incVersion();
}

void RawImage::drawCircle(int x, int y, int radius, bool clear) {
    if (clear) {
        if (format_ == img_rgba)
            drawCircleColor(x, y, radius, black_color_rgba, no_color_rgba, true);
        else
            drawCircleColor(x, y, radius, black_color_rgba, white_color_rgba, true);
    } else {
        drawCircleColor(x, y, radius, black_color_rgba, white_color_rgba, false);
    }
}

void RawImage::fillWithMask(int x, int y, RawImage *mask) {
    auto image = this->duplicate();
    auto same_mask = mask->resizeCanvas(image->w(), image->h());
    CImg<unsigned char> msk(same_mask->buffer(), format_channels[same_mask->format()], same_mask->w(), same_mask->h(), 1, true);
    CImg<unsigned char> src(image->buffer(), format_channels[image->format()], image->w(), image->h(), 1, true);
    src.permute_axes("yzcx");   
    msk.permute_axes("yzcx");
    msk.fill("if(i0==i1&&i1==i2&&i2==255,255,0)", true);
    CImg<uint8_t> kernel(30, 30);
    msk.dilate(kernel);
    src.draw_fill(x, y, black_color_rgba);
    msk.permute_axes("cxyz");
    src.permute_axes("cxyz");
    pasteAt(0, 0, same_mask.get(), image.get());
}

 void RawImage::pasteFill(RawImage *image) {
    CImg<unsigned char> src(image->buffer(), format_channels[image->format()], image->w(), image->h(), 1, true);
    CImg<unsigned char> img(this->buffer(), format_channels[this->format()], this->w(), this->h(), 1, true);
    src.permute_axes("yzcx");
    img.permute_axes("yzcx");
    img.draw_image(0, 0, src.get_resize(image->w(), image->h()));
    img.permute_axes("cxyz");
    src.permute_axes("cxyz");
}

void RawImage::pasteAt(int x, int y, RawImage *image) {
    CImg<unsigned char> src(image->buffer(), format_channels[image->format()], image->w(), image->h(), 1, true);
    CImg<unsigned char> img(this->buffer(), format_channels[this->format()], this->w(), this->h(), 1, true);
    src.permute_axes("yzcx");
    img.permute_axes("yzcx");
    if (image->format() == img_rgba) {
        img.draw_image(x, y, 0, 0, src, src.get_shared_channel(3), 1, 255);
    } else {
        img.draw_image(x, y, src);
    }
    img.permute_axes("cxyz");
    src.permute_axes("cxyz");
}

void RawImage::pasteAtClearFirst(int x, int y, RawImage *image) {
    this->rectangle(x, y, image->w(), image->h(), white_color_rgba, 1.0f);
    pasteAt(x, y, image);
}

void RawImage::pasteAt(int x, int y, RawImage *mask, RawImage *image) {
    CImg<unsigned char> src(image->buffer(), format_channels[image->format()], image->w(), image->h(), 1, true);
    CImg<unsigned char> msk(mask->buffer(), format_channels[mask->format()], mask->w(), mask->h(), 1, true);
    CImg<unsigned char> img(this->buffer(), format_channels[this->format()], this->w(), this->h(), 1, true);
    src.permute_axes("yzcx");
    img.permute_axes("yzcx");
    msk.permute_axes("yzcx");
    if (mask->format() == img_rgba) {
        img.draw_image(x, y, 0, 0, src, msk.get_channel(3), 1, 255);
    } else {
        img.draw_image(x, y, 0, 0, src, msk, 1, 255);
    }
    msk.permute_axes("cxyz");
    img.permute_axes("cxyz");
    src.permute_axes("cxyz");
}

void RawImage::pasteAt(int x, int y, int w, int h, RawImage *image) {
    CImg<unsigned char> src(image->buffer(), format_channels[image->format()], image->w(), image->h(), 1, true);
    CImg<unsigned char> img(this->buffer(), format_channels[this->format()], this->w(), this->h(), 1, true);
    src.permute_axes("yzcx");
    img.permute_axes("yzcx");
    if (image->format() == img_rgba) {
        auto resized = src.get_resize(w, h);
        img.draw_image(x, y, 0, 0, resized, resized.get_shared_channel(3), 1, 255);
    } else {
        img.draw_image(x, y, src.get_resize(w, h));
    }
    img.permute_axes("cxyz");
    src.permute_axes("cxyz");
}

void RawImage::rectangle(int x, int y, int w, int h, uint8_t color[4], float fill_opacity) {
    int src_channels = format_channels[format_];
    CImg<unsigned char> img(buffer_, src_channels, w_, h_, 1, true);
    img.permute_axes("yzcx");
    img.draw_line(x, y, x + w, y, color); // top line
    img.draw_line(x, y + h, x + w, y + h, color); // bottom line
    img.draw_line(x, y, x, y + h, color); // left line
    img.draw_line(x + w, y, x + w, y + h, color); // right line
    if (fill_opacity > 0.001) {
        img.draw_rectangle(x, y, x + w, y + h, color, fill_opacity);
    }
    img.permute_axes("cxyz");
    incVersion();   
}

void RawImage::pasteInvertMask(RawImage *image) {
    // the current image is a mask
    // we-re going to draw the image over the mask, but invert the pixels
    CImg<unsigned char> src(image->buffer(), format_channels[image->format()], image->w(), image->h(), 1, true);
    CImg<unsigned char> img(this->buffer(), format_channels[this->format()], this->w(), this->h(), 1, true);
    src.permute_axes("yzcx");
    img.permute_axes("yzcx");
    auto resized = src.get_resize(w(), h());
    img.draw_image(0, 0, 0, 0, resized, img.get_shared_channel(3), 1, 255);
    img.permute_axes("cxyz");
    src.permute_axes("cxyz");

    if (this->format() == img_rgba) {
        unsigned char *p = this->buffer_;
        for (int i = 0; i < this->buffer_len_; i += 4) {
            *p = 255 - *p; ++p;
            *p = 255 - *p; ++p;
            *p = 255 - *p; ++p;
            ++p;
        }
    }
}

void RawImage::fuseAt(int x, int y, RawImage *image) {
    if (image->format() != img_rgba) {
        return;
    }

    int ww = this->w();
    int hh = this->h();
    int cx = 0;
    int cy = 0;
    if (x >= ww || y >= hh) {
        printf("No Image to past 1\n");
        return;
    }
    ww = image->w();
    hh = image->h();
    if (x < 0) {
        ww = ww - (-x);
        cx = -x;
        x = 0;
    }
    if (y < 0) {
        hh = hh - (-y);
        cy = -y;
        y = 0;
    }
    if (cx + ww > image->w()) {
        ww = image->w() - cx;
    }
    if (cy + hh > image->w()) {
        hh = image->h() - cy;
    }
    if (x + ww > this->w()) {
        ww = this->w() - x;
    }
    if (y + hh > this->h()) {
        hh = this->h() - y;
    }
    if (ww < 1 || hh < 1) {
        printf("No Image to past 2\n");
        return;
    }
    
    printf("Image paste at %d x %d\n", x, y);
    auto crop = image->getCrop(cx, cy, ww, hh);
    if (this->format() != img_rgba) {
        pasteAt(x, y, crop.get());    
        image->clear_pixels(cx, cy, ww, hh);
    } else {
        auto mask = this->getCrop(x, y, ww, hh);
        mask = mask->black_white_into_rgba_mask();
        pasteAt(x, y, mask.get(), crop.get());
        image->clear_pixels(cx, cy, ww, hh);
        mask = mask->invert_mask();
        image->pasteAt(cx, cy, mask.get(), crop.get());
    }
}

image_ptr_t RawImage::to_rgb() {
    if (format_ == img_rgb) {
        return duplicate();
    }
    image_ptr_t r;
    r.reset(new RawImage(NULL, w_, h_, img_rgb, false));
    r->pasteAt(0, 0, this);
    return r;
}

image_ptr_t RawImage::resize_min_area_using_alpha() {
    /*
        Resize the image to its minimal size considering the alpha channel.
    */
    if (format_ != img_rgba) {
        return duplicate();
    }

    unsigned char *src = buffer_;
    int min_y = h_;
    int min_x = w_;
    int max_x = 0;
    int max_y = 0;
    for (int y = 0; y < h_; y++) {
        for (int x = 0; x < w_; x++) {
            if (src[3] != 0) { // alpha != 0
                if (min_x > x) {
                    min_x = x;
                }
                if (min_y > y) {
                    min_y = y;
                }
                if (max_x < x) {
                    max_x = x;
                }
                if (max_y < y) {
                    max_y = y;
                }
            }
            src += 4;
        }
    }

    if (max_x == 0 || max_y == 0) {
        return duplicate();
    }

    int ww = max_x - min_x;
    int hh = max_y - min_y;
    image_ptr_t r;
    r.reset(new RawImage(NULL, ww, hh, img_rgba, false));
    unsigned char *source = buffer_;
    unsigned char *target = r->buffer_;
    int source_stride = w_ * 4;
    int target_stride = ww * 4;
    source += (source_stride * min_y) + min_x * 4;
    for (int y = min_y; y < max_y; y++) {
        memcpy(target, source, target_stride);
        source += source_stride;
        target += target_stride;
    }
    return r;
}

image_ptr_t RawImage::pasteAtNoBackground(int x, int y, RawImage *mask, RawImage *image) {
    auto white = newImage(image->w(), image->h(), true);
    white->clear(255, 255, 255, 255);
    auto black = newImage(image->w(), image->h(), true);
    black->clear(0, 0, 0, 255);
    white->pasteAt(x, y, mask, black.get());
    white = white->removeBackground(true);
    white->pasteAt(x, y, mask, image);
    return white;
}

void RawImage::pasteFrom(int x, int y, float zoom, RawImage *image) {
    int w = this->w();
    int h = this->h();
    if (zoom < 0.001) {
        zoom = 0.001;
    }

    w /= zoom;
    h /= zoom;

    // keep the area inside the source image    
    if (w <= 0 || h <= 0 || x >= image->w() || y >= image->h())  {
        return;
    }
    if (x + w > image->w()) {
        w = image->w() - x;
    }
    if (y + h > image->h()) {
        h = image->h() - y;
    }
    memset(this->buffer_, 255, this->buffer_len_); // turn this image white
    if (h < 0 || w < 0) {
        return;
    }
    CImg<unsigned char> src(image->buffer(), format_channels[image->format()], image->w(), image->h(), 1, true);
    CImg<unsigned char> self(this->buffer(), format_channels[this->format()], this->w(), this->h(), 1, true);

    src.permute_axes("yzcx");
    self.permute_axes("yzcx");
    float ratio, invert_w, invert_h;

    if (w > h) {
        ratio = h / (float)w;
        invert_w = w * zoom;
        invert_h = invert_w * ratio;
    } else {
        ratio = w / (float)h;
        invert_h = h * zoom;
        invert_w = invert_h * ratio;
    }
    self.draw_image(0, 0, src.get_crop(x, y, x + w, y + h).get_resize(invert_w, invert_h));
    if (invert_w < this->w()) {
        self.draw_rectangle(invert_w, 0, this->w(), this->h(), no_color_rgba);
    }
    if (invert_h < this->h()) {
        self.draw_rectangle(0, invert_h, this->w(), this->h(), no_color_rgba);
    }
    self.permute_axes("cxyz");
    src.permute_axes("cxyz");
}

image_ptr_t RawImage::resizeCanvas(uint32_t x, uint32_t y) {
    image_ptr_t result(new RawImage(NULL, x, y, this->format(), false));
    CImg<unsigned char> src(this->buffer(), format_channels[this->format()], this->w(), this->h(), 1, true);
    CImg<unsigned char> self(result->buffer(), format_channels[result->format()], result->w(), result->h(), 1, true);
    src.permute_axes("yzcx");
    self.permute_axes("yzcx");
    self.draw_image(0, 0, src);
    self.permute_axes("cxyz");
    src.permute_axes("cxyz");
    return result;
}

image_ptr_t  RawImage::resizeInTheCenter(uint32_t x, uint32_t y) {
    image_ptr_t result(new RawImage(NULL, x, y, this->format(), this->format() == img_rgba));

    bool ref_x = false;
    float scale;

    if (x > y) {
        ref_x = true;
        scale = y / (float) x;   // 300 / 400
    } else {
        scale = x / (float) y;  // 400 / 300
    }

    int new_x, new_y;

    if (ref_x) {
        new_x = x;
        new_y = new_x * scale;   // 0.75
        if (new_y > y) {
            scale = y / (float) new_y;
        } else {
            scale = 1.0;
        }
    } else {
        new_y = y;
        new_x = new_y * scale;
        if (new_x > x) {
            scale = x / (float) new_x;
        } else {
            scale = 1.0;
        }
    }

    new_x = new_x * scale;
    new_y = new_y * scale;

    int sx = (x - new_x) / 2;
    int sy = (y - new_y) / 2;
    
    auto resized = this->resizeImage(new_x, new_y);

    result->pasteAt(sx, sy, resized.get());

    return result;
}

image_ptr_t RawImage::resizeImage(uint32_t x, uint32_t y) {
    image_ptr_t result(new RawImage(NULL, x, y, this->format(), false));
    CImg<unsigned char> src(this->buffer(), format_channels[this->format()], this->w(), this->h(), 1, true);
    CImg<unsigned char> self(result->buffer(), format_channels[result->format()], result->w(), result->h(), 1, true);
    src.permute_axes("yzcx");
    self.permute_axes("yzcx");
    self.draw_image(0, 0, src.get_resize(x, y));
    self.permute_axes("cxyz");
    src.permute_axes("cxyz");
    return result;
}

image_ptr_t RawImage::resizeImage(uint32_t size) {
    if (this->w() > this->h()) {
        return resizeImage(size, (int)((this->h() / (float)this->w()) * size));
    } else {
        return resizeImage((int)((this->w() / (float)this->h()) * size), size);
    }
}

image_ptr_t RawImage::getCrop(uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    image_ptr_t result(new RawImage(NULL, w, h, this->format(), false));
    CImg<unsigned char> src(this->buffer(), format_channels[this->format()], this->w(), this->h(), 1, true);
    CImg<unsigned char> self(result->buffer(), format_channels[result->format()], result->w(), result->h(), 1, true);
    src.permute_axes("yzcx");
    self.permute_axes("yzcx");
    self.draw_image(0, 0, src.get_crop(x, y, x + w, y + h));
    self.permute_axes("cxyz");
    src.permute_axes("cxyz");
    return result;
}

void RawImage::clear_pixels(uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    if (this->format() != img_rgba) {
        return;
    }
    CImg<unsigned char> self(this->buffer(), format_channels[this->format()], this->w(), this->h(), 1, true);
    self.permute_axes("yzcx");
    self.draw_rectangle(x, y, x + w, y + h, no_color_rgba);
    self.permute_axes("cxyz");
}

image_ptr_t RawImage::blur(int size) {
    image_ptr_t result = this->duplicate();
    CImg<unsigned char> self(result->buffer(), format_channels[result->format()], result->w(), result->h(), 1, true);
    self.permute_axes("yzcx");
    self.blur(size, size, 0.0, 1, true);
    self.permute_axes("cxyz");
    return result;
}

bool RawImage::getColor(int x, int y, uint8_t *r, uint8_t *g, uint8_t *b, uint8_t *a) {
    if (x < 0 || y < 0 || x > w() || y > h()) {
        return false;
    }
    CImg<unsigned char> self(this->buffer(), format_channels[this->format()], this->w(), this->h(), 1, true);
    self.permute_axes("yzcx");

    *r = *self.data(x, y, 0, 0);
    *g = *self.data(x, y, 0, 1);
    *b = *self.data(x, y, 0, 2);

    if (format_ == img_rgba) {
        *a = *self.data(x, y, 0, 3);
    } else {
        *a = 255;
    }

    self.permute_axes("cxyz");
    return true;
}

image_ptr_t RawImage::erode(int size) {
    image_ptr_t result = this->duplicate();
    CImg<unsigned char> self(result->buffer(), format_channels[result->format()], result->w(), result->h(), 1, true);
    self.permute_axes("yzcx");
    self.erode(size);
    self.permute_axes("cxyz");
    return result;
}

image_ptr_t RawImage::dilate(int size) {
    image_ptr_t result = this->duplicate();
    CImg<unsigned char> self(result->buffer(), format_channels[result->format()], result->w(), result->h(), 1, true);
    self.permute_axes("yzcx");
    self.dilate(size);
    self.permute_axes("cxyz");
    return result;
}


image_ptr_t RawImage::flip(bool vertically) {
    image_ptr_t result = this->duplicate();
    CImg<unsigned char> self(result->buffer(), format_channels[result->format()], result->w(), result->h(), 1, true);
    self.permute_axes("yzcx");
    if (vertically) {
        self.mirror("y");
    } else {
        self.mirror("x");
    }
    self.permute_axes("cxyz");
    return result;
}

image_ptr_t RawImage::rotate() {
    image_ptr_t result = this->duplicate();
    CImg<unsigned char> self(result->buffer(), format_channels[result->format()], result->w(), result->h(), 1, true);
    self.permute_axes("yzcx");
    self.rotate(90);
    self.permute_axes("cxyz");
    result->w_ = this->h_;
    result->h_ = this->w_;
    return result;
}


image_ptr_t RawImage::fit1024() {
    int nx, ny;
    if (this->h() > this->w()) {
        ny = 1024;
        nx = (int)((this->w() / (float)this->h()) * 1024.0);
    } else {
        nx = 1024;
        ny = (int)((this->h() / (float)this->w()) * 1024.0);
    }
    printf("Image resized from %dx%d to %dx%d\n", this->w(), this->h(), nx, ny);
    return this->resizeImage(nx, ny);
}

image_ptr_t RawImage::to_rgba() {
    // convert the image to RGBA
    if (format_ == img_rgba) {
        return duplicate();
    }
    auto img = std::make_shared<RawImage>(
        (const unsigned char *) NULL, this->w(), this->h(), img_rgba, false
    );
    img->pasteAt(0, 0, this);
    return img;
}

image_ptr_t RawImage::negative_mask() {
    /*
        Return a negative mask.
    */
    if (format_ != img_rgba) {
        return std::make_shared<RawImage>(
            (const unsigned char *) NULL, this->w(), this->h(), img_rgba, false
        );
    }
    auto img = this->duplicate();
    unsigned char *p = img->buffer_;
    unsigned char *r, *g, *b, *a;
    for (int i = 0; i < img->buffer_len_; i += 4) {
        r = p; ++p;
        g = p; ++p;
        b = p; ++p;
        a = p; ++p;
        *r = 255;
        *g = 255;
        *b = 255;
        *a = (255 - *a);
    }

    return img;
}

image_ptr_t RawImage::create_mask_from_alpha_channel() {
    image_ptr_t r;
    if (this->format() != img_rgba) {
        return r;
    }
    r.reset(new RawImage(NULL, this->w(), this->h(), img_rgba, false));
    bool masked = false;
    unsigned char *p = r->buffer_;
    unsigned char *s = this->buffer_;
    s += 3; // use the alpha channel 
    for (int i = 0; i < r->buffer_len_; i += 4) {
        if (!*s) {
            *p = 255; ++p;
            *p = 255; ++p;
            *p = 255; ++p;
            *p = 255; ++p;
            masked = true;
        } else {
            *p = 255; ++p;
            *p = 255; ++p;
            *p = 255; ++p;
            *p = 0; ++p;
        }
        s += 4;
    }
    if (!masked) {
        return image_ptr_t();
    }
    return r->dilate(4);
}

image_ptr_t RawImage::create_modification_mask(RawImage *modified_image) {
    image_ptr_t r;
    if (this->w() != modified_image->w() || this->h() != modified_image->h() || this->format() != modified_image->format()) {
        return r;
    }
    r.reset(new RawImage(NULL, this->w(), this->h(), img_rgba, false));
    unsigned char *p = r->buffer_;
    unsigned char *s = this->buffer_;
    unsigned char *m = modified_image->buffer_;
    size_t len = r->buffer_len_;
    for (size_t i = 0; i < len; i += 4) {
        if (*s != *m || *(s + 1) != *(m + 1) || *(s + 2) != *(m + 2)) {
            *p = 255; ++p;
            *p = 255; ++p;
            *p = 255; ++p;
            *p = 0; ++p;
        } else {
            *p = 255; ++p;
            *p = 255; ++p;
            *p = 255; ++p;
            *p = 255; ++p;
        }
        s += 4;
        m += 4;
    }
    return r;
}

image_ptr_t RawImage::resizeLeft(int value) {
    auto img = std::make_shared<RawImage>(
        (const unsigned char *) NULL, this->w() + value, this->h(), img_rgba, false
    );
    img->pasteAt(value, 0, this);
    return img;
}

image_ptr_t RawImage::resizeRight(int value) {
    return resizeCanvas(this->w() + value, this->h());
}

image_ptr_t RawImage::resizeTop(int value) {
    auto img = std::make_shared<RawImage>(
        (const unsigned char *) NULL, this->w(), this->h() + value, img_rgba, false
    );
    img->pasteAt(0, value, this);
    return img;
}

image_ptr_t RawImage::resizeBottom(int value) {
    return resizeCanvas(this->w(), this->h() + value);
}

void RawImage::clear(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    int src_channels = format_channels[format_];
    CImg<unsigned char> img(buffer_, src_channels, w_, h_, 1, true);
    img.permute_axes("yzcx");
    uint8_t color[] = {r, g, b, a};
    img.draw_rectangle(0, 0, w_, h_, color);
    img.permute_axes("cxyz");
    incVersion();
}

json RawImage::toJson() {
    json result;
    result["width"] = w_;
    result["height"] = h_;
    result["mode"] = format_ == img_rgb ? "RGB" : "RGBA";
    result["data"] = base64_encode((const unsigned char *) buffer_, buffer_len_);
    return result;
}

image_ptr_t newImage(uint32_t w, uint32_t h, bool enable_alpha) {
    auto r = std::make_shared<RawImage>(
        (const unsigned char *) NULL, w, h, enable_alpha ? img_rgba : img_rgb
    );
    r->clear(255, 255, 255, enable_alpha ? 0 : 255);
    return r;
}

image_ptr_t newImage(const json& value) {
    if (value.is_null()) {
        return image_ptr_t();
    }
    if (!value.contains("data") || !value.contains("width") || !value.contains("height") || !value.contains("mode")) {
        return image_ptr_t();
    }
    auto img_mode = value["mode"].get<std::string>();
    auto format = img_gray_8bit;
    if (img_mode == "RGB")
        format = img_rgb;
    else if (img_mode == "RGBA")
        format = img_rgba;
    const auto &data = value["data"].get<std::string>();
    const auto &width = value["width"].get<uint32_t>();
    const auto &height = value["height"].get<uint32_t>();
    size_t decoded_size = 0;
    auto decoded = base64_decode(data.c_str(), data.size());
    if (decoded.first == nullptr || decoded.second != width * height * format_channels[format]) {
        return image_ptr_t();
    }
    return std::make_shared<RawImage>(decoded.first.get(), width, height, format);
}

std::vector<image_ptr_t> newImageList(const json& value) {
    std::vector<image_ptr_t> result;
    if (value.is_null() || !value.is_array()) {
        return result;
    }
    for (const auto &item : value) {
        auto img = newImage(item);
        if (img) {
            result.push_back(img);
        }
    }
    return result;
}

} // namespace editorium

