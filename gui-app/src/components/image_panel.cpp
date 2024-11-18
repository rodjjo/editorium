#include <cmath>
#include <GL/gl.h>
#include <FL/gl.h>
#include <FL/Fl.H>
#include <FL/fl_ask.H>

#include "misc/config.h"
#include "misc/utils.h"
#include "components/image_panel.h"
#include "messagebus/messagebus.h"
#include "websocket/code.h"
#include "websocket/tasks.h"
#include "windows/sapiens_ui.h"

namespace editorium
{
    namespace {
        const int gl_format[img_format_count] = {
            GL_LUMINANCE,
            GL_RGB,
            GL_RGBA
        };

        std::map<fl_uintptr_t, void*> window_user_data;

        uint8_t red_color[4] = {255, 0, 0, 255};
        uint8_t gray_color[4] = {128, 128, 128, 255};
        uint8_t pink_color[4] = {255, 128, 128, 255};
        
        int g_mouse_delta = 0;

#ifdef _WIN32
    HHOOK _hook;
    LRESULT MouseHook(
        int    nCode,
        WPARAM wParam,
        LPARAM lParam
    ) {
        if (wParam == WM_MOUSEWHEEL) {
            LPMOUSEHOOKSTRUCTEX minfo = (LPMOUSEHOOKSTRUCTEX) lParam;
            g_mouse_delta = (short)HIWORD(minfo->mouseData);
        }
        return CallNextHookEx(_hook, nCode, wParam, lParam);
    }
#endif
    bool hook_enabled = false;
    void register_mouse_hook() {
        if (hook_enabled) {
            return;
        }
        hook_enabled = true;
#ifdef _WIN32
    _hook = SetWindowsHookExA(WH_MOUSE, MouseHook, NULL, GetCurrentThreadId());
#endif
        }

    }

    Layer::Layer(ViewSettings *parent, const char *path) : parent_(parent), image_(ws::filesystem::load_image(path)) {
        if (image_) {
            w_ = image_->w();
            h_ = image_->h();
        }
    }

    Layer::Layer(ViewSettings *parent, int w, int h, bool transparent) : parent_(parent), image_(newImage(w, h, transparent))  {
        if (image_) {
            w_ = image_->w();
            h_ = image_->h();
        }
    }

    Layer::Layer(ViewSettings *parent, image_ptr_t image) : parent_(parent), image_(image->duplicate()) {
        if (image_) {
            w_ = image_->w();
            h_ = image_->h();
        }
    }

    void Layer::refresh(bool force) {
        parent_->refresh(force);
    }

    Layer::~Layer() {
    }

    std::shared_ptr<Layer> Layer::duplicate() {
        auto l = std::make_shared<Layer>(parent_, this->image_->duplicate());
        l->x(x());
        l->y(y());
        l->w(w());
        l->h(h());
        return l;
    }

    RawImage *Layer::getImage() {
        return image_.get();
    }

    int Layer::x() {
        return x_;
    }

    int Layer::y() {
        return y_;
    }

    int Layer::w() {
        return w_;
    }

    int Layer::h() {
        return h_;
    }

    bool Layer::selected() {
        return parent_->selected_layer() == this && parent_->layer_count() > 1;
    }

    bool Layer::visible() {
        return visible_;
    }

    void Layer::visible(bool value) {
        visible_ = value;
        parent_->refresh(true);
    }

    bool Layer::pinned() {
        return pinned_;
    }

    void Layer::pinned(bool value) {
        pinned_ = value;
    }

    void Layer::focusable(bool value) {
        focusable_ = value;
    }

    bool Layer::focusable() {
        return focusable_;
    }

    const char *Layer::name() {
        return name_.c_str();
    }

    void Layer::name(const char *value) {
        name_ = value;
    }

    void Layer::set_modified() {
        ++version_;
    }

    void Layer::x(int value) {
        if (pinned_) {
            return;
        }
        if (value != x_) {
            x_ = value;
            ++version_;
            refresh();
        }
    }

    void Layer::y(int value) {
        if (pinned_) {
            return;
        }
        if (value != y_) {
            y_ = value;
            ++version_;
            refresh();
        }
    }

    void Layer::w(int value) {
        if (pinned_) {
            return;
        }
        if (value != w_) {
            w_ = value;
            ++version_;
            refresh();
        }
    }

    void Layer::h(int value) {
        if (pinned_) {
            return;
        }
        if (value != h_) {
            h_ = value;
            ++version_;
            refresh();
        }
    }

    float Layer::scale_x() {
        if (image_) {
            return (float)image_->w() / (float)w_;
        }
        return 1.0;
    }

    float Layer::scale_y() {
        if (image_) {
            return (float)image_->h() / (float)h_;
        }
        return 1.0;
    }


    int Layer::version() {
        return version_;
    }

    void Layer::replace_image(image_ptr_t new_image) {
        if (new_image && image_) {
            int diff_x = image_->w() - new_image->w();
            int diff_y = image_->h() - new_image->h();
            x(x() + diff_x / 2);
            y(y() + diff_y / 2);
            w(new_image->w());
            h(new_image->h());
            image_ = new_image;
            refresh();
        }
    }

    void Layer::restore_size() {
        if (image_) {
            w_ = image_->w();
            h_ = image_->h();
            ++version_;
            refresh(true);
        }
    }

    void Layer::scale_size(bool up) {
        if (image_) {
            if (image_->w() * 4.0 < w_ || image_->h() * 4.0 < h_)  {
                w_ = image_->w() * 4.0;
                h_ = image_->h() * 4.0;
            }
            if (image_->w() * 0.10 > w_ || image_->h() * 0.10 > h_) {
                w_ = image_->w() * 0.10;
                h_ = image_->h() * 0.10;
            }
            float five_x = w_ * 0.05;
            float five_y = h_ * 0.05;
            if (up) {
                w_ += five_x;
                h_ += five_y;
                x_ -= five_x / 2.0;
                y_ -= five_y / 2.0;
            } else {
                w_ -= five_x;
                h_ -= five_y;
                x_ += five_x / 2.0;
                y_ += five_y / 2.0;
            }
            ++version_;
            refresh(true);
        }
    }

    ViewSettings::ViewSettings(ImagePanel *parent): parent_(parent), cache_() {

    }

    ViewSettings::~ViewSettings() {

    }

    size_t ViewSettings::layer_count() {
        return layers_.size();
    }

    Layer* ViewSettings::add_layer(std::shared_ptr<Layer> l, bool in_front) {
        if (l->getImage()) {
            if (strlen(l->name()) == 0) {
                char buffer[128] = "";
                sprintf(buffer, "Layer %03d", (int)name_index_);
                l->name(buffer);
            }
            name_index_ += 1;
            if (in_front) {
                layers_.insert(layers_.begin(), l);
            } else {
                layers_.push_back(l);
            }
            selected_ = l.get();
            refresh(true);
            publish_event(parent_, event_layer_count_changed, NULL);
            publish_event(parent_, event_layer_selected, selected_);
            cache_.clear_hits();
            return l.get();
        }
        return NULL;
    }

    Layer* ViewSettings::add_layer(const char *path) {
        return add_layer(std::make_shared<Layer>(this, path));
    }

    Layer* ViewSettings::add_layer(int w, int h, bool transparent) {
        return add_layer(std::make_shared<Layer>(this, w, h, transparent));
    }

    Layer* ViewSettings::add_layer(image_ptr_t image, bool in_front) {
        return add_layer(std::make_shared<Layer>(this, image), in_front);
    }

    Layer* ViewSettings::at(size_t position) {
        return layers_.at(position).get();
    }

    void ViewSettings::duplicate_selected() {
        if (selected_) {
            add_layer(selected_->duplicate());
        }
    }

    image_ptr_t ViewSettings::get_selected_image() {
        image_ptr_t r;
        int sx, sy, sw, sh;
        if (selected_coords_to_image_coords(&sx, &sy, &sw, &sh)) {
            auto merged = merge_layers_to_image(true);
            r = merged->getCrop(sx, sy, sw, sh);
        }
        return r;
    }

    bool ViewSettings::selected_coords_to_image_coords(int *x, int *y, int *w, int *h) {
        int sx, sy, sw, sh;
        // get the selected area...
        if (get_selected_area(&sx, &sy, &sw, &sh)) {
            // get the final image area
            int iax, iay, iaw, iah;
            get_image_area(&iax, &iay, &iaw, &iah);

            // constraint the selection inside the image
            if (sx < iax) {
                sw -= (iax - sw);
                sx = iax;
            }

            if (sy < iay) {
                sh -= (iay - sh);
                sy = iay;
            }

            if (sx + sw > iaw) {
                sw = sw - (iaw - (sx + sw));
            }

            if (sy + sh > iah) {
                sh = sh - (iah - (sy + sh));
            }

            *x = sx;
            *y = sy;
            *w = sw;
            *h = sh;

            return true;
        }
        return false;
    }

    void ViewSettings::remove_background_selected(background_removal_type_t technique) {
        if (selected_) {
            std::vector<image_ptr_t> mask_list;

            if (technique == remove_using_default) {
                mask_list = ws::diffusion::run_preprocessor("background", {selected_->getImage()->duplicate()});
            } else if (technique == remove_using_sapiens) {
                auto classes = select_sapien_classes();
                if (classes.empty()) {
                    return;
                }
                mask_list = ws::diffusion::run_seg_sapiens(classes, {selected_->getImage()->duplicate()});
            } else if (technique == remove_using_gdino) {
                auto classes = fl_input("Enter the classes to segment (comma separated)", "");
                if (!classes) {
                    return;
                }

                mask_list = ws::diffusion::run_seg_ground_dino(classes, {selected_->getImage()->duplicate()});
            }

            if (!mask_list.empty()) {
                auto mask = mask_list[0];
                // mask = mask->rgba_mask_into_black_white();
                if (technique == remove_using_default) {
                    mask = mask->invert_mask();
                }
                auto fg =  selected_->duplicate();
                auto mask_copy = mask;
                mask = mask->removeBackground(true);

                // clear the foreground at the background image layer
                auto white = newImage(selected_->getImage()->w(), selected_->getImage()->h(), true);
                white->clear(255, 255, 255, 255);
                selected_->getImage()->pasteAt(0, 0, mask.get(), white.get());
                // clear the background at the foreground layer

                auto fg_img = newImage(selected_->getImage()->w(), selected_->getImage()->h(), false);
                fg_img = fg_img->pasteAtNoBackground(0, 0, mask_copy.get(), fg->getImage());
                fg->replace_image(fg_img->resize_down_alpha());
                add_layer(fg);
            }
        }
    }


    void ViewSettings::flip_horizoltal_selected() {
        if (!selected_) {
            return;
        }
        auto img  = selected_->getImage()->flip(false);
        selected_->replace_image(img);
        refresh(true);
    }

    void ViewSettings::flip_vertical_selected() {
        if (!selected_) {
            return;
        }
        auto img  = selected_->getImage()->flip(true);
        selected_->replace_image(img);
        refresh(true);
    }

    void ViewSettings::rotate_selected() {
        if (!selected_) {
            return;
        }
        auto img  = selected_->getImage()->rotate();
        selected_->replace_image(img);
        refresh(true);
    }

    void ViewSettings::remove_layer(size_t position) {
        if (position > layers_.size()) {
            return;
        }
        if (selected_ == layers_.at(position).get()) {
            selected_ = NULL;
        }
        layers_.erase(layers_.begin() + position);
        if (layers_.empty()) {
            cache_.set_scroll(0, 0);
        }
        refresh();
        publish_event(parent_, event_layer_count_changed, NULL);
    }

    void ViewSettings::select(size_t index) {
        if (index < layers_.size()) {
            if (selected_ != layers_.at(index).get()) {
                selected_ = layers_.at(index).get();
                refresh(true);
                publish_event(parent_, event_layer_selected, selected_);
            }
        } else {
            selected_ = NULL;
            refresh(true);
            publish_event(parent_, event_layer_selected, selected_);
        }
    }

    size_t ViewSettings::selected_layer_index() {
        for (size_t i = 0; i < layers_.size(); i++) {
            if (layers_[i].get() == selected_) {
                return i;
            }
        }
        return 0;
    }

    ImageCache *ViewSettings::cache() {
        return &cache_;
    }

    uint16_t ViewSettings::getZoom() {
        return zoom_;
    }

    void ViewSettings::setZoom(uint16_t value) {
        if (value < 10) {
            value = 10;
        } else if (value > 200) {
            value = 200;
        }
        if (value == zoom_) {
            return;
        }
        float old_zoom = zoom_;
        zoom_ = value;
        refresh(true);
    }

    void ImagePanel::anchor_zoom(bool start, int x, int y) {
        if (start) {
            scroll_px_ = view_settings_->cache()->get_scroll_x();
            scroll_py_ = view_settings_->cache()->get_scroll_y();
            anchor_x_ = (x / getZoom()) - scroll_px_;
            anchor_y_ = (y / getZoom()) - scroll_py_;
        } else {
            int new_anchor_x = (x / getZoom()) - scroll_px_;
            int new_anchor_y = (y / getZoom()) - scroll_py_;
            int diff_x = new_anchor_x - anchor_x_;
            int diff_y = new_anchor_y - anchor_y_;
            int new_scroll_x = scroll_px_ + diff_x;
            int new_scroll_y = scroll_py_ + diff_y;
            view_settings_->constraint_scroll(getZoom(), image_->w(), image_->h(), &new_scroll_x, &new_scroll_y);
            view_settings_->cache()->set_scroll(new_scroll_x,  new_scroll_y);
            scroll_px_ = view_settings_->cache()->get_scroll_x();
            scroll_py_ = view_settings_->cache()->get_scroll_y();
            schedule_redraw(true);
        }
    }

    void ViewSettings::clear_layers() {
        name_index_ = 1;
        selected_ = NULL;
        layers_.clear();
        cache_.set_scroll(0, 0);
        cache_.clear_hits();
        cache_.gc();
        refresh(true);
        publish_event(parent_, event_layer_count_changed, NULL);
    }

    Layer* ViewSettings::selected_layer() {
        return selected_;
    }

    void ViewSettings::refresh(bool force) {
        parent_->schedule_redraw(force);
    }

    size_t ViewSettings::layer_at_mouse_coord(float zoom, int x, int y) {
        if (zoom != 0) {
            float mx = x / zoom - cache_.get_scroll_x();
            float my = y / zoom - cache_.get_scroll_y();
            Layer *l;
            for (size_t i = layers_.size(); i > 0; i--) {
                l = layers_[i - 1].get();
                if (l->focusable() && l->x() < mx && mx <  l->x() + l->w() 
                    && l->y() < my && my <  l->y() + l->h() ) {
                    return i - 1;
                }
            }
        }
        return (size_t) -1;
    }

    void ViewSettings::mouse_drag(float zoom, int dx, int dy, int x, int y) {
        if (zoom == 0 || !selected_ || layer_count() < 2) 
            return;
        float mx = x;
        float my = y;
        float mdx = dx;
        float mdy = dy;
        float dragx = (mx - mdx) / zoom;
        float dragy = (my - mdy) / zoom;
        selected_->x(drag_begin_x_ + dragx);
        selected_->y(drag_begin_y_ + dragy);
        compact_image_area(false);
        parent_->schedule_redraw(true);
    }

    void ViewSettings::mouse_drag_end() {
        compact_image_area(true);
        parent_->schedule_redraw(true);
    }

    void ViewSettings::compact_image_area(bool complete) {
        /*
            This function ensure the image does not scroll far than largest image layer size * 2
            This function ensure that at least one layer has the position (0, 0)
        */
        float zoom = getZoom() * 0.01;
        int max_w = 0;
        int max_h = 0;
        for (auto & l : layers_) {
            if (l->w() > max_w) {
                max_w = l->w();
            }
            if (l->h() > max_h) {
                max_h = l->h();
            }
        }

        max_w = int(1.005 * max_w);
        max_h = int(1.005 * max_h);
        int max_w2 = int(2.00 * max_w);
        int max_h2 = int(2.00 * max_h);

        for (auto & l : layers_) {
            if (l->x() < -max_w) {
                l->x(-max_w);
            }
            if (l->y() < -max_h) {
                l->y(-max_h);
            }
            if (l->x() > max_w2) {
                l->x(max_w2);
            }
            if (l->y() > max_h2) {
                l->y(max_h2);
            }
        }
       
        if (!complete) {
            return;
        }

        int x, y, w, h;
        get_image_area(&x, &y, &w, &h); // 10, 10

        int add_x = -x;
        int add_y = -y;

        int scroll_px = cache()->get_scroll_x() + x;
        int scroll_py = cache()->get_scroll_y() + y;
        constraint_scroll(zoom, parent_->view_w(), parent_->view_h(), &scroll_px, &scroll_py);
        cache()->set_scroll(scroll_px, scroll_py); 

        for (auto & l : layers_) {
            l->x(l->x() + add_x);
            l->y(l->y() + add_y);
        }
       
        parent_->schedule_redraw(true);
    }

    void ViewSettings::mouse_drag_begin() {
        if (selected_) {
            drag_begin_x_ = selected_->x();
            drag_begin_y_ = selected_->y();
        }
    }

    void ViewSettings::mouse_scale(bool up) {
        if (selected_) {
            selected_->scale_size(up);
            compact_image_area();
        } 
    }

    void ViewSettings::get_image_area(int *x, int *y, int *w, int *h) {
        if (layers_.size() < 1) {
            *x = 0;
            *y = 0;
            *w = 0;
            *h = 0;
            return;
        }
        int max_x = -32000;
        int max_y = -32000;
        int min_x = 32000;
        int min_y = 32000;

        for (auto & l: layers_) {
            if (l->x() + l->w() > max_x) {
                max_x = l->x() + l->w();
            }
            if (l->y() + l->h() > max_y) {
                max_y = l->y() + l->h();
            }
            if (l->x() < min_x) {
                min_x = l->x();
            }
            if (l->y() < min_y) {
                min_y = l->y();
            }
        }

        *x = min_x;
        *y = min_y;
        *w = max_x - min_x;
        *h = max_y - min_y;
    }

    void ViewSettings::brush_size(int value) {
        brush_size_ = value;
    }

    int ViewSettings::brush_size() {
        return brush_size_;
    }

    void ViewSettings::constraint_scroll(float zoom, int view_w, int view_h, int *sx, int *sy) {
        if (zoom == 0) {
            return;
        }
        int x, y, w, h;
        get_image_area(&x, &y, &w, &h);
        
        int half_view_x = (view_w / 2) / zoom;
        int half_view_y = (view_h / 2) / zoom;

        int scroll_size_x = (w) - half_view_x;
        int scroll_size_y = (h) - half_view_y;

        if (*sx > half_view_x) {
            *sx = half_view_x;
        }
        if (*sy > half_view_y) {
            *sy = half_view_y;
        }
        if (*sx < -scroll_size_x) {
            *sx = -scroll_size_x;
        }
        if (*sy < -scroll_size_y) {
            *sy = -scroll_size_y;
        }
        #ifdef PRINT_SCROLL_COORDINATES
        printf("Scroll coordinate %d x %d max(%d x %d) zoom %03f\n", *sx, *sy, scroll_size_x, scroll_size_y, zoom);
        #endif
    }
    
    bool ViewSettings::get_selected_area(int *x, int *y, int *w, int *h) {
        if (!has_selected_area()) {
            *x = 0;
            *y = 0;
            *w = 0;
            *h = 0;
            return false;
        }
        *x = selected_area_x_;
        *y = selected_area_y_;
        *w = selected_area_w_;
        *h = selected_area_h_;
        return true;
    }

    void ViewSettings::set_selected_area(int x, int y, int w, int h) {
        if (layers_.empty()) {
            return;
        }
        selected_area_ = true;
        selected_area_x_ = x;
        selected_area_y_ = y;
        selected_area_w_ = w;
        selected_area_h_ = h;
        refresh(true);
    }

    bool ViewSettings::has_selected_area() {
        return selected_area_ && !layers_.empty();
    }

    image_ptr_t ViewSettings::merge_layers_to_image(bool enable_alpha) {
        image_ptr_t r;
        if (!layer_count()) {
            return r;
        }
        int x, y, w, h;
        get_image_area(&x, &y, &w, &h);
        r = newImage(w, h, enable_alpha);
        for (auto & l : layers_) {
            if (l->getImage()) {
                r->pasteAtClearFirst(l->x() + x, l->y() + y, l->getImage());
            }
        }
        return r;
    }

    void ViewSettings::clear_selected_area() {
        selected_area_ = false;
    }

    void ViewSettings::set_image(image_ptr_t value) {
        clear_layers();
        add_layer(value);
        refresh(true);
    }

    void ViewSettings::fuse_image(image_ptr_t value) {
        if (layer_count() < 1) {
            add_layer(value);
            refresh(true);
            return;
        }
        int iax, iay, unused;
        int sx, sy, sw, sh;
        int sx0, sy0;
        get_image_area(&iax, &iay, &unused, &unused);
        value = value->addAlpha();
        auto clean_value = value->duplicate();
        if (selected_coords_to_image_coords(&sx, &sy, &sw, &sh)) {
            bool x1_in_layer = false;
            bool y1_in_layer = false;
            bool x2_in_layer = false;
            bool y2_in_layer = false;
            //image_ptr_t negative_mask;
            for (size_t i = layer_count(); i > 0; i--) {
                auto ly = at(i - 1);
                auto limg = ly->getImage();
                if (!limg) {
                    continue;
                }
                
                sx0 = sx - ly->x();
                sy0 = sy - ly->y();

                if (sx0 >= 0) {
                    x1_in_layer = true;
                }
                if (sy0 >= 0) {
                    y1_in_layer = true;
                }
                if (sx0 + value->w() < limg->w()) {
                    x2_in_layer = true;
                }
                if (sy0 + value->h() < limg->h()) {
                    y2_in_layer = true;
                }

                // px = ly->x() - iax;
                // py = ly->y() - iay;
                limg->fuseAt(sx0, sy0, value.get());
                ly->set_modified();
                // mask = mask->negative_mask();
            }
            if (!x1_in_layer || !y1_in_layer || !x2_in_layer || !y2_in_layer) {
                // add a new layer with the image
                auto img = std::make_shared<RawImage>((const unsigned char *)0, clean_value->w(), clean_value->h(), img_rgba, false);
                img->clear(255, 255, 255, 255);
                img->pasteAt(0, 0, clean_value.get());
                auto l = std::make_shared<Layer>(this, img);
                l->x(sx);
                l->y(sy);
                add_layer(l, true);
            }
        }

        refresh(true);
    }
    
    void ViewSettings::set_mask() {
        int layer_count = parent_->enable_colored_mask_editor() ? 3 : 2;
        if (layers_.size() > layer_count || layers_.size() < 1) {
            return;
        }
        auto first_image = layers_[0]->getImage();
        for (int i = 1; i < layer_count; i++) {
            if (layers_.size() < i + 1) {
                add_layer(newImage(first_image->w(), first_image->h(), true));
            } else {
                layers_[i]->replace_image(newImage(first_image->w(), first_image->h(), true));
            }
        }
        refresh(true);
    }

    ImageCache::ImageCache() {

    }

    ImageCache::~ImageCache() {

    }

    void ImageCache::set_scroll(int x, int y) {
        scroll_x_ = x;
        scrool_y_ = y;
    }

    int ImageCache::get_scroll_x() {
        return scroll_x_;
    }

    int ImageCache::get_scroll_y() {
        return scrool_y_;
    }

    void ImageCache::get_bounding_box(float zoom, Layer *layer, int *x, int *y, int *w, int *h) {
        *w = round((float)layer->w() * zoom);
        *h = round((float)layer->h() * zoom);
        *x = round((float)(scroll_x_ + layer->x()) * zoom);
        *y = round((float)(scrool_y_ + layer->y()) * zoom);
    }

    bool ImageCache::is_layer_visible(float zoom, Layer *layer, int scroll_x, int scroll_y, int view_w, int view_h) {
        if (zoom == 0) {
            return false;
        }
        int x, y, w, h;
        get_bounding_box(zoom, layer,  &x, &y, &w, &h);
        return rectRect(x, y, w, h, 0, 0, view_w, view_h);
    }

    RawImage *ImageCache::get_cached(float zoom, Layer *layer, Layer *invert_layer) {
        int _unused;
        int sw, sh;
        get_bounding_box(zoom, layer, &_unused, &_unused, &sw, &sh);
        int iw = (float)layer->getImage()->w() * zoom;
        int ih = (float)layer->getImage()->h() * zoom;
        if (sw * 0.8 < iw ||
            sh * 0.8 < ih || zoom < 0.8) {
            auto it = items_.find(layer);
            RawImage *result = NULL;
            if (it != items_.end()) {
                if (it->second.version == layer->version() && it->second.cache) {
                    result = it->second.cache.get();
                    it->second.hit = true;
                }
            } 
            if (result == NULL) {
                CachedLayer cl;
                cl.version = layer->version();
                cl.hit = true;
                if (invert_layer) {
                    auto tmp = invert_layer->getImage()->resizeImage(sw, sh);
                    cl.cache = layer->getImage()->resizeImage(sw, sh);
                    cl.cache->pasteInvertMask(tmp.get());
                } else {
                    cl.cache = layer->getImage()->resizeImage(sw, sh);
                }
                items_[layer] = cl;
                result  = cl.cache.get();
            }
            return result;
        }

        return layer->getImage();
    }

    bool ImageCache::is_modified(Layer *layer) {
        auto it = items_.find(layer);
        if (it != items_.end()) {
            return it->second.version != layer->version();
        }
        return true;
    }

    void ImageCache::clear_hits() {
        for (auto & i : items_) {
            i.second.hit = false;
        }
    }

    void ImageCache::gc() {
        // clear from cache all images that has no hits
        for (auto it = items_.begin(); it != items_.end(); ) {
            if (it->second.hit) {
                it++;
            } else {
                it = items_.erase(it);
            }
        }
    }

    ImagePanel::ImagePanel(
        uint32_t x, uint32_t y, 
        uint32_t w, uint32_t h, 
        const char *unique_title
    ) : Fl_Gl_Window(x, y, w, h, unique_title) {
        view_settings_.reset(new ViewSettings(this));
        after_constructor();
    }

    ImagePanel::ImagePanel(
        uint32_t x, uint32_t y, 
        uint32_t w, uint32_t h, 
        const char *unique_title,
        std::shared_ptr<ViewSettings> vs
    ) : Fl_Gl_Window(x, y, w, h, unique_title), view_settings_(vs) {
        after_constructor();
    }

    void ImagePanel::after_constructor() {
        register_mouse_hook();
        Fl::remove_timeout(ImagePanel::imageRefresh, this);
        Fl::add_timeout(0.01, ImagePanel::imageRefresh, this);
    }

    ImagePanel::~ImagePanel() {
        Fl::remove_timeout(ImagePanel::imageRefresh, this);
    }
    
    void ImagePanel::imageRefresh(void *cbdata) {
        ((ImagePanel *) cbdata)->imageRefresh();
        Fl::repeat_timeout(0.033, ImagePanel::imageRefresh, cbdata);
    }

    void ImagePanel::imageRefresh() {
        if (should_redraw_) {
            if (!image_ || image_->w() != w() || image_->h() != h()) {
                image_ = newImage(w(), h(), false);
                should_redraw_ = true;
                force_redraw_ = true;
                return;
            }
            redraw();
            should_redraw_ = false;
            publish_event(this, event_layer_after_draw, NULL);
        }
    }

    ViewSettings *ImagePanel::view_settings() {
        return view_settings_.get();
    }

    int ImagePanel::handle(int event)
    {
        switch (event) {
            case FL_MOUSEWHEEL: {
                #ifdef _WIN32
                int16_t z = g_mouse_delta;
                #else
                int16_t z = Fl::event_dy(); 
                #endif
                if (z != 0) {
                    z = z > 0 ? -10 : 10;
                    bool control_pressed = Fl::event_command() != 0;
                    bool shift_pressed = Fl::event_shift() != 0;
                    bool alt_pressed = Fl::event_alt() != 0;
                    if (!control_pressed && !shift_pressed && !alt_pressed) {
                        if (enable_zoom()) {
                            anchor_zoom(true, move_last_x_, move_last_y_);
                            view_settings_->setZoom(view_settings_->getZoom() + z);
                            anchor_zoom(false, move_last_x_, move_last_y_);
                        }
                    } else if (!control_pressed && shift_pressed && !alt_pressed) {
                        if (enable_resize()) {
                            view_settings_->mouse_scale(z > 0);
                        }
                    }
                }
            }
            break;

            case FL_MOVE:
            case FL_DRAG:
            {
                if (!mouse_down_left_ && !mouse_down_right_)
                {
                    mouse_down_x_ = Fl::event_x();
                    mouse_down_y_ = Fl::event_y();
                }
                mouse_move(mouse_down_left_, mouse_down_right_, mouse_down_x_, mouse_down_y_, Fl::event_x(), Fl::event_y());
                move_last_x_ = Fl::event_x();
                move_last_y_ = Fl::event_y();
            }
            break;

            case FL_PUSH:
            {
                bool mouse_down_left = Fl::event_button() == FL_LEFT_MOUSE;
                bool mouse_down_right = !mouse_down_left && Fl::event_button() == FL_RIGHT_MOUSE;
                mouse_down_control_ = false;
                mouse_down_shift_ = false;
                mouse_down_alt_ = false;
                if (mouse_down_left || mouse_down_right)
                {
                    mouse_down_control_ = Fl::event_command() != 0;
                    mouse_down_shift_ = Fl::event_shift() != 0;
                    mouse_down_alt_ = Fl::event_alt() != 0;
                    mouse_down_x_ = Fl::event_x();
                    mouse_down_y_ = Fl::event_y();
                    scroll_px_ = view_settings_->cache()->get_scroll_x();
                    scroll_py_ = view_settings_->cache()->get_scroll_y();
                    move_last_x_ = mouse_down_x_;
                    move_last_y_ = mouse_down_y_;
                    mouse_down_left_ = mouse_down_left;
                    mouse_down_right_ = mouse_down_right;
                    mouse_down(mouse_down_left_, mouse_down_right_, mouse_down_x_, mouse_down_y_);
                    mouse_move(mouse_down_left_, mouse_down_right_, mouse_down_x_, mouse_down_y_, mouse_down_x_, mouse_down_y_);
                }
                else
                {
                    return Fl_Gl_Window::handle(event);
                }
            }
            break;

            case FL_RELEASE:
            {
                if (Fl::event_button() != FL_LEFT_MOUSE && Fl::event_button() != FL_RIGHT_MOUSE)
                {
                    return Fl_Gl_Window::handle(event);
                }
                if (mouse_down_left_ || mouse_down_right_)
                {
                    // mouse_up(mouse_down_left_, mouse_down_right_, mouse_down_x_, mouse_down_y_, Fl::event_x(), Fl::event_y());
                    mouse_down_left_ = false;
                    mouse_down_right_ = false;
                    mouse_down_control_ = false;
                    mouse_down_shift_ = false;
                    mouse_down_alt_ = false;
                }
            }
            break;

            default:
                return Fl_Gl_Window::handle(event);
        }
        return 1;
    }

    void ImagePanel::draw() {
        if (!valid())
        {
            glLoadIdentity();
            glViewport(0, 0, this->w(), this->h());
            glLoadIdentity();
            valid(1);
        }

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (!image_) {
            return;
        }
        
        bool modified = force_redraw_;
        for (size_t i = 0; i < view_settings_->layer_count() && !modified; i++) {
            modified = view_settings_->cache()->is_modified(view_settings_->at(i));
        }

        if (modified) {
            force_redraw_ = false;
            image_->clear(255, 255, 255, 255);
            view_settings_->cache()->clear_hits();
            size_t mask_layer_index = enable_colored_mask_editor() ? 2 : 1;
            for (size_t i = 0; i < view_settings_->layer_count(); i++) {
                if (!view_settings_->at(i)->visible()) {
                    continue;
                }
                draw_layer(view_settings_->at(i), i == mask_layer_index && enable_mask_editor());
            }

            int ix, iy, iw, ih;
            view_settings_->get_image_area(&ix, &iy, &iw, &ih);
            draw_rectangle(ix, iy, iw, ih, gray_color, false);
            if (view_settings_->get_selected_area(&ix, &iy, &iw, &ih)) {
                draw_rectangle(ix, iy, iw, ih, gray_color, true);
                draw_rectangle(ix, iy, iw, ih, pink_color, false);
            }
            view_settings_->cache()->gc();
        }

        glRasterPos2f(-1, 1);
        glPixelZoom(1, -1);

        if (image_->w() % 4 == 0)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        else
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        glDrawPixels(image_->w(), image_->h(), gl_format[image_->format()], GL_UNSIGNED_BYTE, image_->buffer());

        glRasterPos2f(0.0f, 0.0f);
        glPixelZoom(1.0f, 1.0f);

        draw_brush();
        blur_gl_contents(this->w(), this->h(), move_last_x_, move_last_y_);
    }

    int ImagePanel::view_w() {
        return image_->w();
    }

    int ImagePanel::view_h() {
        return image_->h();
    }

    
    void ImagePanel::draw_rectangle(int x, int y, int w, int h, uint8_t color[4], bool fill) {
        x += view_settings_->cache()->get_scroll_x();
        y += view_settings_->cache()->get_scroll_y();
        float zoom = getZoom();
        x *= zoom;
        y *= zoom;
        w *= zoom;
        h *= zoom;
        image_->rectangle(x, y, w, h, color, fill ? 0.1 : 0.0f);
    }

    void ImagePanel::draw_layer(Layer *layer, bool mask) {
        float zoom = getZoom();
        if (!view_settings_->cache()->is_layer_visible(zoom, layer, 0, 0, image_->w(), image_->h())) {
            return;
        }
        Layer *inv_layer = mask ? view_settings_->at(0) : NULL;
        auto img = view_settings_->cache()->get_cached(zoom, layer, inv_layer);
        int x, y, w, h;
        view_settings_->cache()->get_bounding_box(zoom, layer, &x, &y, &w, &h);
        image_->pasteAt(x, y, img);
        if (layer->selected()) {
            image_->rectangle(x, y, w, h, red_color);
        }
    }

    void ImagePanel::draw_brush() {
        if (!enable_mask_editor() || view_settings_->layer_count() < 2 || !view_settings_->at(1)->visible()) {
            return;
        }
        float sx = 2.0 / this->w();
        float sy = 2.0 / this->h();
        uint8_t mask_r, mask_g, mask_b, mask_a;
        get_color_mask_color(&mask_r, &mask_g, &mask_b, &mask_a);
        glBegin(GL_LINE_LOOP);
        float theta;
        float x;
        float y;
        float wx;
        float wy;
        int brush_size = view_settings_->brush_size();
        size_t l_index = view_settings_->layer_at_mouse_coord(getZoom(), move_last_x_, move_last_y_);
        float cx = move_last_x_;
        float cy = move_last_y_;
        if (l_index != (size_t) -1) {
            auto layer = view_settings_->at(l_index);
            brush_size = brush_size / ((layer->scale_x() + layer->scale_y()) / 2.0);
        }
        float fmask_r = mask_r / 255.0;
        float fmask_g = mask_g / 255.0;
        float fmask_b = mask_b / 255.0;
        float fmask_a = mask_a / 255.0;
        uint8_t color_step = enable_colored_mask_editor() ? 3 : 2;
        uint8_t step = 0;
        for (int ii = 0; ii < 36; ++ii)   {
            if (step == 0) {
                glColor3f(0, 0, 0);
            } else if (step == 1) {
                glColor3f(1, 1, 1);
            } else {
                glColor4f(fmask_r, fmask_g, fmask_b, fmask_a);
            }
            step = (step + 1) % color_step;
            theta = (2.0f * 3.1415926f) * float(ii) / float(36);
            x = (brush_size * getZoom()) * cosf(theta);
            y = (brush_size * getZoom()) * sinf(theta);
            wx = ((x + cx) * sx) - 1;
            wy = 1 - ((y + cy) * sy);
            glVertex2f(wx, wy);
        }
        glEnd();
    }

    void ImagePanel::getDrawingCoord(float &x, float &y) {
        x = -1;
        y = 1;
        if (view_settings_->layer_count() < 1) {
            return;
        }

        auto ref = view_settings_->at(0)->getImage(); 
        if (!ref) {
            return;
        }
        
        int refw = ref->w() * getZoom();
        int refh = ref->h() * getZoom();
        x = -1;
        y = 1;
        if (refw < this->w()) {
            x = -(((2.0 / this->w()) * refw) / 2.0);
        }
        if (refh < this->h()) {
            y = (((2.0 / this->h()) * refh) / 2.0);
        }
    }

    void ImagePanel::draw_overlay() {
        //Fl_Gl_Window::draw_overlay();
    }

    void ImagePanel::schedule_redraw(bool force) {
        force_redraw_ = force_redraw_ || force;
        should_redraw_ = true;
        if (force_redraw_) {
            for (size_t i = 0; i < view_settings_->layer_count(); i++) {
                view_settings_->at(i)->set_modified();
            }
        }
    }

    void ImagePanel::resize(int x, int y, int w, int h) {
        Fl_Gl_Window::resize(x, y, w, h);
        schedule_redraw(true);
    };

    float ImagePanel::getZoom() {
        return view_settings_->getZoom() * 0.01;
    }
    
    void ImagePanel::enable_color_mask_editor(bool value) {
        edit_color_mask_ = value;
    }
    
    void ImagePanel::color_mask_color(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
        color_mask_color_[0] = r;
        color_mask_color_[1] = g;
        color_mask_color_[2] = b;
        color_mask_color_[3] = a;
    }

    void ImagePanel::get_color_mask_color(uint8_t *r, uint8_t *g, uint8_t *b, uint8_t *a) {
        *r = color_mask_color_[0];
        *g = color_mask_color_[1];
        *b = color_mask_color_[2];
        *a = color_mask_color_[3];
    }

    void ImagePanel::mouse_drag(int dx, int dy, int x, int y) {
        dx = (x - dx) / getZoom();
        dy = (y - dy) / getZoom();
        auto sx = scroll_px_ + dx;
        auto sy = scroll_py_ + dy;
        view_settings_->constraint_scroll(getZoom(), image_->w(), image_->h(), &sx, &sy);
        view_settings_->cache()->set_scroll(sx,  sy);
        schedule_redraw(true);
    }

    void ImagePanel::mouse_move(bool left_button, bool right_button, int down_x, int down_y, int move_x, int move_y) {
        if (mouse_down_control_ && left_button) {
            // control was pressed during mouse down, so lets change the scroll 
            if (enable_scroll()) {
                mouse_drag(down_x, down_y, move_x, move_y);
            }
            down_x = move_x;
            down_y = move_y;
            return;
        }

        if (enable_mask_editor() && view_settings_->layer_count() > 1 && view_settings_->at(1)->visible() || get_config()->private_mode()) {
            schedule_redraw(false);
        }

        bool ctl_buttons = mouse_down_control_ || mouse_down_alt_ || mouse_down_shift_;

        if (!mouse_down_alt_ && !mouse_down_control_ && mouse_down_shift_ && enable_colored_mask_editor()) {
            if (view_settings_->layer_count() > 0) {
                auto img = view_settings_->at(0)->getImage();
                // pickup the color from the image at x and y
                // move_x = (move_x) / getZoom() - view_settings_->cache()->get_scroll_x();
                // move_y = (move_y) / getZoom() - view_settings_->cache()->get_scroll_y();
               // uint8_t r, g, b, a;
                uint8_t gl_color[4] = {255, 255, 255, 255};
                if (move_x >= 0 && move_y >= 0 && move_x < this->w() && move_y < this->h()) {
                    glReadPixels(move_x, this->h() - move_y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, gl_color);
                    color_mask_color(gl_color[0], gl_color[1], gl_color[2], gl_color[3]);
                    publish_event(this, event_layer_mask_color_picked, NULL);
                }
            }
        }

        if (enable_mask_editor() && !ctl_buttons) {
            int expected_layer_count = enable_colored_mask_editor() ? 3 : 2;
            if (view_settings_->layer_count() < expected_layer_count || view_settings_->brush_size() < 1 || !view_settings_->at(1)->visible()) {
                return;
            }
            bool edit_color = enable_colored_mask_editor();
            int mask_index = edit_color ? 2 : 1;
            if (edit_color_mask_) {
                mask_index = 1;
            }
            auto layer_under_mouse = view_settings_->layer_at_mouse_coord(getZoom(), move_x, move_y);
            if (layer_under_mouse != mask_index) {
                return;
            }
            auto layer = view_settings_->at(mask_index);
            auto img = view_settings_->at(mask_index)->getImage();
            move_x = ((move_x / getZoom()) - layer->x() - view_settings_->cache()->get_scroll_x()) * layer->scale_x();
            move_y = ((move_y / getZoom()) - layer->y() - view_settings_->cache()->get_scroll_y()) * layer->scale_y();
            if (left_button) {
                if (edit_color_mask_ && edit_color) {
                    img->drawCircleColor(move_x, move_y, view_settings_->brush_size(), color_mask_color_, color_mask_color_, false);
                } else {
                    img->drawCircle(move_x, move_y, view_settings_->brush_size(), false);
                }
            } else if (right_button) {
                img->drawCircle(move_x, move_y, view_settings_->brush_size(), true);
            }
            schedule_redraw(true);
            return;
        }

        if (right_button && !mouse_down_control_ && !mouse_down_shift_) {
            if (enable_drag()) {
                view_settings_->mouse_drag(getZoom(), down_x, down_y, move_x, move_y);
                return;
            }
        }

        if (left_button && !right_button && !ctl_buttons) {
            if (enable_selection()) {
                auto zoom = getZoom();
                int sx = view_settings_->cache()->get_scroll_x();
                int sy = view_settings_->cache()->get_scroll_y();
                int sw = down_x - move_x;
                int sh = down_y - move_y;
                if (sw < 0) sw = -sw;
                if (sh < 0) sh = -sh;

                if (move_x < down_x) {
                    down_x = move_x;
                }
                if (move_y < down_y) {
                    down_y = move_y;
                }
                down_x -= sx * zoom;
                down_y -= sy * zoom;
                view_settings_->set_selected_area(down_x / zoom, down_y / zoom, sw / zoom, sh / zoom);
            }
            return;
        }
    }

    void ImagePanel::mouse_down(bool left_button, bool right_button, int down_x, int down_y) {
        view_settings_->mouse_drag_end();

        if (!mouse_down_control_ && (left_button || right_button)) {
            // select the layer at the mouse coordinates
            auto index = view_settings_->layer_at_mouse_coord(getZoom(), down_x, down_y);
            view_settings_->select(index);
        }

        if (right_button && !mouse_down_control_ && !mouse_down_shift_) {
            if (enable_drag()) {
                view_settings_->mouse_drag_begin();
            }
        }
    }

    void ImagePanel::mouse_up(bool left_button, bool right_button, int down_x, int down_y, int up_x, int up_y) {
        if (right_button && !mouse_down_control_ && !mouse_down_shift_) {
            if (enable_drag()) {
                view_settings_->mouse_drag(getZoom(), down_x, down_y, up_x, up_y);
                view_settings_->mouse_drag_end();
            }
        }

    }

    bool ImagePanel::enable_selection() {
        return true;
    }

    bool ImagePanel::enable_scroll() {
        return true;
    }

    bool ImagePanel::enable_zoom() {
        return true;
    }

    bool ImagePanel::enable_drag() {
        return true;
    }

    bool ImagePanel::enable_resize() {
        return true;
    }

    bool ImagePanel::enable_mask_editor() {
        return false;
    }

    bool ImagePanel::enable_colored_mask_editor() {
        return false;
    }

    void ImagePanel::cancel_refresh() {
        should_redraw_ = false;
    }

    void ImagePanel::clear_scroll() {
        view_settings_->mouse_drag_end();
        view_settings_->cache()->set_scroll(0, 0);
        schedule_redraw(true);
    }

} // namespace editorium
