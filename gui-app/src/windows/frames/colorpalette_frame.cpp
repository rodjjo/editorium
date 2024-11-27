#include <inttypes.h>
#include "components/xpm/xpm.h"
#include "misc/dialogs.h"

#include "colorpalette_frame.h"


namespace editorium {

namespace 
{
    typedef uint8_t color_t[4];
    const color_t palette_colors[sizeof(button_colors_t)/sizeof(button_ptr_t)] = {
        {0, 0, 0, 255},
        {255, 255, 255, 255},
        {255, 0, 0, 255},
        {0, 255, 0, 255},
        {0, 0, 255, 255},
        {255, 255, 0, 255},
        {255, 0, 255, 255},
        {0, 255, 255, 255},
        {128, 128, 128, 255},
        {192, 192, 192, 255},
        {128, 0, 0, 255},
        {0, 128, 0, 255},
        {0, 0, 128, 255},
        {128, 128, 0, 255},
        {128, 0, 128, 255},
        {0, 128, 128, 255}
    };
} // unnamed namespace 


ColorPaletteFrame::ColorPaletteFrame(Fl_Group *parent, ImagePanel *image_panel, const std::string& context) {
    parent_ = parent;
    image_panel_ = image_panel;
    context_ = context;
    
    for (int i = 0; i < sizeof(palette_) / sizeof(palette_[0]); i++) {
        palette_[i] = std::make_shared<Button>([this, palette_index{i}]() {
            this->choose_color_for_palette(palette_index);
        });
        palette_[i]->setColor(palette_colors[i][0], palette_colors[i][1], palette_colors[i][2]);
        palette_[i]->tooltip("click to select a color.\n[SHIFT] + click: change the color.\n[ALT] + click: use current color");
    }

    btn_reset_colors_ = std::make_shared<Button>(xpm::image(xpm::img_24x24_erase), [this]() {
        if (ask("Do you want to reset the colors to default?")) {
            this->reset_colors();
        }
    });
    btn_reset_colors_->tooltip("Reset colors to default");
    btn_current_color_ = std::make_shared<Button>([this]() {
        this->pickup_current_color();
    });
    btn_current_color_ ->tooltip("Shows the current color.\n[SHIFT] + click: change the color");

    aligncomponents();

    update_current_color();
}

ColorPaletteFrame::~ColorPaletteFrame() {

}

void ColorPaletteFrame::aligncomponents() {
    // the button array is 2x8 aligned to fill parent_ with a bottom margin of 1 button height
    int x = parent_->x();
    int y = parent_->y();
    int w = parent_->w();
    int h = parent_->h();
    int button_w = w / 2;
    int button_h = h / 9;
    for (int i = 0; i < sizeof(palette_) / sizeof(palette_[0]); i++) {
        palette_[i]->size(button_w, button_h);
        palette_[i]->position(x + (i % 2) * button_w, y + (i / 2) * button_h);
    }
    btn_reset_colors_->size(button_w, button_h);
    btn_reset_colors_->position(x, y + button_h * 8);
    btn_current_color_->size(button_w, button_h);
    btn_current_color_->position(btn_reset_colors_->x() + button_w, btn_reset_colors_->y());
}

void ColorPaletteFrame::reset_colors() {
    size_t palette_size = sizeof(palette_colors) / sizeof(palette_colors[0]);
    for (size_t i = 0; i <  palette_size; i++) {
        palette_[i]->setColor(palette_colors[i][0], palette_colors[i][1], palette_colors[i][2]);
    }
}

void ColorPaletteFrame::choose_color_for_palette(int palette_index) {
    uint8_t r = 255, g = 255, b = 255, a = 255;
    palette_[palette_index]->getColor(&r, &g, &b);
    if (Fl::event_shift()) {
        if (pickup_color("Foreground color", &r, &g, &b)) {
            palette_[palette_index]->setColor(r, g, b);
        } else {
            return;
        }
    } else if (Fl::event_alt()) {
        image_panel_->get_color_mask_color(&r, &g, &b, &a);
        palette_[palette_index]->setColor(r, g, b);
    }
    image_panel_->color_mask_color(r, g, b, a);
    btn_current_color_->setColor(r, g, b);
}

void ColorPaletteFrame::update_current_color() {
    uint8_t r, g, b, a;
    image_panel_->get_color_mask_color(&r, &g, &b, &a);
    btn_current_color_->setColor(r, g, b);
}

void ColorPaletteFrame::pickup_current_color() {
    uint8_t r, g, b, a;
    btn_current_color_->getColor(&r, &g, &b);
    if (pickup_color("Foreground color", &r, &g, &b)) {
        btn_current_color_->setColor(r, g, b);
        image_panel_->color_mask_color(r, g, b, 255);
    }
}


}  // namespace editorium
