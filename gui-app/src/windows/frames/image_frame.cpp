#include "components/xpm/xpm.h"

#include "misc/dialogs.h"
#include "messagebus/messagebus.h"
#include "windows/frames/image_frame.h"
#include "windows/image_palette_ui.h"

namespace editorium
{
    const char *modes_text[img2img_mode_max] = {
        "Disabled",
        "Img2Img",
        "Inpaint masked",
        "Inpaint not masked"
    };

    const char *brush_captions[brush_size_count] = {
        "Disabled",
        "1 Pixel",
        "2 Pixels",
        "4 Pixels",
        "8 Pixels",
        "16 Pixels",
        "32 Pixels",
        "64 Pixels",
        "128 Pixels"
    };

    const uint8_t brushes_sizes[brush_size_count] = {
        0, 1, 2, 4, 8, 16, 32, 64, 128
    };

    const char *inpaint_modes[inpaint_mode_count] = {
        "Original image",
        "Fill image",
        "Use Img2Image",
        "Whole image (original)",
        "Whole image (fill)",
        // "Latent Noise",
        // "Latent Nothing"
    };


    
ImageFrame::ImageFrame(Fl_Group *parent, ImagePanel *img) {
    parent_ = parent;
    img_ = img;
    auto current = parent->current();

    choice_mode_ = new Fl_Choice(0 , 0, 1, 1, "Image usage mode");
    choice_brush_size_ = new Fl_Choice(0 , 0, 1, 1, "Editor brush size");
    choice_inpaint_mode_ = new Fl_Choice(0 , 0, 1, 1, "Inpainting mode");
    strength_input_ = new Fl_Float_Input(0 , 0, 1, 1, "Strenght");

    btnNewMask_.reset(new Button(xpm::image(xpm::img_24x24_new_document),
        [this] () {
            publish_event(this, event_image_frame_new_mask, NULL);
        }
    ));
    btnOpenMask_.reset(new Button(xpm::image(xpm::img_24x24_open),
        [this] () {
            publish_event(this, event_image_frame_open_mask, NULL);
        }
    ));
    btnColor_.reset(new Button([this] () {
            configure_mask_color();
        }
    ));
    btnUseColor_.reset(new Button(xpm::image(xpm::img_24x24_diagram),
        [this] () {
            configure_mask_color_enabled();
        }
    ));
    btnSegGDino_.reset(new Button(xpm::image(xpm::img_24x24_alien),
        [this] () {
            publish_event(this, event_image_frame_seg_gdino, NULL);
        }
    ));
    btnSegSapiens_.reset(new Button(xpm::image(xpm::img_24x24_female),
        [this] () {
            publish_event(this, event_image_frame_seg_sapiens, NULL);
        }
    ));
    btnPixelate_.reset(new Button(xpm::image(xpm::img_24x24_picture),
        [this] () {
            pixelate_current_image();
        }
    ));
    btnFromPalette_.reset(new Button(xpm::image(xpm::img_24x24_list),
        [this] () {
            pickup_palette_image();
        }
    ));

    for (int i = 0; i < img2img_mode_max; i++) {
        choice_mode_->add(modes_text[i]);
    }

    for (int i = 0; i < brush_size_count; i++) {
        choice_brush_size_->add(brush_captions[i]);
    }

    for (int i = 0; i < inpaint_mode_count; i++) {
        choice_inpaint_mode_->add(inpaint_modes[i]);
    }

    choice_mode_->align(FL_ALIGN_TOP_LEFT);
    choice_brush_size_->align(FL_ALIGN_TOP_LEFT);
    choice_inpaint_mode_->align(FL_ALIGN_TOP_LEFT);
    strength_input_->align(FL_ALIGN_TOP_LEFT);

    choice_mode_->value(0);
    choice_brush_size_->value(5);
    choice_inpaint_mode_->value(0);
    strength_input_->value(25.0);

    choice_mode_->callback(combobox_cb, this);
    choice_brush_size_->callback(combobox_cb, this);
    choice_inpaint_mode_->callback(combobox_cb, this);

    btnNewMask_->tooltip("Create a new mask");
    btnOpenMask_->tooltip("Open a image to use as a mask");
    btnColor_->tooltip("Select a color to use on the image");
    btnUseColor_->tooltip("When the button is down it draws a color over the image");
    btnSegGDino_->tooltip("Create mask using Grounding Dino segmentation");
    btnSegSapiens_->tooltip("Create mask using Sapiens segmentation (from facebook)");
    btnPixelate_->tooltip("Pixelate the current image");
    btnFromPalette_->tooltip("Pick an image from the image palette");

    btnColor_->setColor(255, 255, 255);
    btnUseColor_->enableDownUp();

    alignComponents();
    combobox_selected();
    configure_mask_color_enabled();
}

ImageFrame::~ImageFrame() {
}

bool ImageFrame::inpaint_enabled() {
    return inpaint_enabled_;
}

void ImageFrame::pickup_palette_image() {
    auto img = pickup_image_from_palette();
    if (img) {
        if (img_->view_settings()->layer_count() == 0) {
            img_->view_settings()->add_layer(img);
        } else {
            img_->view_settings()->at(0)->replace_image(img);
        }
    }
}

void ImageFrame::pixelate_current_image() {
    if (img_->view_settings()->layer_count() < 3) {
        return;
    }
    if (!ask("Do you want to pixelate the current image?")) {
        return;
    }

    auto reference_img = img_->view_settings()->at(0)->getImage();
    
    float ratio = 1.0;
    if (reference_img->w() > reference_img->h()) {
        ratio = 512.0 / reference_img->w();
    } else {
        ratio = 512.0 / reference_img->h();
    }
    int min_size_w, min_size_h; // 32 pixels minimum, checking the ratio
    int new_w, new_h;
    if (reference_img->w() > reference_img->h()) {
        new_w = 512;
        new_h = reference_img->h() * ratio;
        min_size_w = 32;
        min_size_h = 32 * ratio;
    } else {
        new_h = 512;
        new_w = reference_img->w() * ratio;
        min_size_h = 32;
        min_size_w = 32 * ratio;
    }

    auto pixelated = reference_img->blur(4)->resizeImage(min_size_w, min_size_h)->resizeImage(new_w, new_h);
    img_->view_settings()->at(1)->replace_image(pixelated);
}

void ImageFrame::inpaint_enabled(bool enabled) {
    if (enabled == inpaint_enabled_) {
        return;
    }
    inpaint_enabled_ = enabled;
    if (enabled) {
        choice_mode_->clear();
        for (int i = 0; i < img2img_mode_max; i++) {
            choice_mode_->add(modes_text[i]);
        }
    } else {
        choice_mode_->clear();
        choice_mode_->add(modes_text[0]);
        choice_mode_->add(modes_text[1]);
    }
    choice_mode_->value(0);
}

img2img_mode_t ImageFrame::get_mode() {
    return mode_;
}

int ImageFrame::get_brush_size() {
    return brush_size_;
}

inpaint_mode_t ImageFrame::get_inpaint_mode() {
    return inpaint_mode_;
}

void ImageFrame::alignComponents() {
    int left = parent_->x();
    int top = parent_->y();
    int w = img_->x() - parent_->x();
    int h = parent_->h();

    choice_mode_->resize(left + 5, top + 25, w - 10, 30);
    choice_inpaint_mode_->resize(left + 5, choice_mode_->h() + choice_mode_->y() + 25, w - 10, 30);
    choice_brush_size_->resize(left + 5, choice_inpaint_mode_->h() + choice_inpaint_mode_->y() + 25, w - 10, 30);
    strength_input_->resize(left + 5, choice_brush_size_->h() + choice_brush_size_->y() + 25, w - 10, 30);

    btnNewMask_->size((w - 15) / 2, 30);
    btnSegGDino_->size((w - 15) / 2, 30);
    btnSegSapiens_->size((w - 15) / 2, 30);
    btnUseColor_->size((w - 15) / 2, 30);
    btnColor_->size((w - 15) / 2, 30);
    btnPixelate_->size((w - 15) / 2, 30);
    btnFromPalette_->size((w - 15) / 2, 30);
    btnOpenMask_->size(btnNewMask_->w(), btnNewMask_->h());
    btnNewMask_->position(left + 5, strength_input_->y() + strength_input_->h() + 5);
    btnOpenMask_->position(btnNewMask_->x() + btnNewMask_->w() + 5, btnNewMask_->y());
    btnUseColor_->position(left + 5, btnOpenMask_->y() + btnOpenMask_->h() + 5);
    btnColor_->position(btnUseColor_->x() + btnUseColor_->w() + 5, btnUseColor_->y());
    btnPixelate_->position(left + 5, btnColor_->y() + btnColor_->h() + 5);
    btnFromPalette_->position(btnPixelate_->x() + btnPixelate_->w() + 5, btnPixelate_->y());
    btnSegGDino_->position(left + 5, btnFromPalette_->y() + btnFromPalette_->h() + 5);
    btnSegSapiens_->position(btnSegGDino_->x() + btnSegGDino_->w() + 5, btnSegGDino_->y());
}

void ImageFrame::combobox_cb(Fl_Widget* widget, void *cbdata) {
    static_cast<ImageFrame *>(cbdata)->combobox_cb(widget);
}

void ImageFrame::combobox_cb(Fl_Widget* widget) {
    if (inside_cb_) {
        return;
    }
    inside_cb_ = true;
    combobox_selected();
    publish_event(this, event_image_frame_mode_selected, NULL);
    inside_cb_ = false;
}

void ImageFrame::combobox_selected() {
    if (enabled()) {
        if (parent_->visible_r()) {
            img_->show();
        }
    } else {
        img_->hide();
    }

    if (choice_mode_->value() > 1) {
        btnNewMask_->show();
        btnOpenMask_->show();
        btnColor_->show();
        btnUseColor_->show();
        btnSegGDino_->show();
        btnSegSapiens_->show();
        choice_brush_size_->show();
        choice_inpaint_mode_->show();
        strength_input_->show();
        btnPixelate_->show();
        btnFromPalette_->show();
    } else {
        btnNewMask_->hide();
        btnColor_->hide();
        btnUseColor_->hide();
        btnOpenMask_->hide();
        btnSegGDino_->hide();
        btnSegSapiens_->hide();
        choice_brush_size_->hide();
        choice_inpaint_mode_->hide();
        btnPixelate_->hide();
        btnFromPalette_->hide();
        if (choice_mode_->value() > 0) {
            strength_input_->show();
        } else {
            strength_input_->hide();
        }
    }

    mode_ = static_cast<img2img_mode_t>(choice_mode_->value());
    brush_size_ = brushes_sizes[choice_brush_size_->value()];
    inpaint_mode_ = static_cast<inpaint_mode_t>(choice_inpaint_mode_->value());
}

void ImageFrame::enable_mode() {
    if (choice_mode_->value() < 1) {
        choice_mode_->value(1);
    }
    combobox_selected();
}

bool ImageFrame::enabled() {
    return choice_mode_->value() > 0;
}

float ImageFrame::get_strength() {
    float value = 80.0;
    sscanf(strength_input_->value(), "%f", &value);
    if (value < 0.0)
        value = 0;
    if (value > 100.0)
        value = 100.0;
    char buffer[100] = { 0, };
    sprintf(buffer, "%0.1f", value);
    strength_input_->value(buffer);
    return value / 100.0; 
}

void ImageFrame::configure_mask_color() {
    uint8_t r = 255, g = 255, b = 255, a = 255;
    img_->get_color_mask_color(&r, &g, &b, &a);
    btnColor_->setColor(r, g, b);
    if (pickup_color("Foreground color", &r, &g, &b)) {
        btnColor_->setColor(r, g, b);
        img_->color_mask_color(r, g, b, a);
        if (!btnUseColor_->down()) {
            btnUseColor_->down(true);
            configure_mask_color_enabled();
        }
    }
}

void ImageFrame::configure_mask_color_enabled() {
    if (img_->view_settings()->layer_count() < 3) {
        return;
    }
    img_->view_settings()->at(0)->focusable(false);
    if (btnUseColor_->down()) {
        img_->enable_color_mask_editor(true);
        img_->view_settings()->at(2)->focusable(false);
        img_->view_settings()->at(2)->visible(false);
    } else {
        img_->enable_color_mask_editor(false);
        img_->view_settings()->at(2)->focusable(true);
        img_->view_settings()->at(2)->visible(true);
    }
}

void ImageFrame::handle_event(int event, void *sender) {
    switch (event) {
        case event_layer_mask_color_picked: {
            uint8_t r = 255, g = 255, b = 255, a = 255;
            img_->get_color_mask_color(&r, &g, &b, &a);
            btnColor_->setColor(r, g, b);
        }
        break;
    }
}

} // namespace editorium
