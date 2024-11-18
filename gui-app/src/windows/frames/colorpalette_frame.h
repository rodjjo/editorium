#pragma once

#include <string>
#include <FL/Fl_Group.H>
#include "components/image_panel.h"
#include "components/button.h"

namespace editorium {

typedef std::shared_ptr<Button> button_ptr_t;
typedef button_ptr_t button_colors_t[16];

class ColorPaletteFrame {
public:
    ColorPaletteFrame(Fl_Group *parent, ImagePanel *image_panel, const std::string& context);
    ~ColorPaletteFrame();
    void aligncomponents();
    void update_current_color();
private:
    void reset_colors();
    
    void choose_color_for_palette(int palette_index);

private:
    std::string context_;

private:
    Fl_Group *parent_;
    ImagePanel *image_panel_;
    button_colors_t palette_;
    button_ptr_t btn_current_color_;
    button_ptr_t btn_reset_colors_;
};


} // namespace editorium
