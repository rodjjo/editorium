#pragma once

#include <memory>
#include <FL/Fl_Group.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Float_Input.H>

#include "components/button.h"
#include "components/image_panel.h"

namespace editorium
{

typedef enum {
   img2img_text,
   img2img_img2img,
   img2img_inpaint_masked,
   img2img_inpaint_not_masked,
   
   // keep img2img_mode_max ath the end
   img2img_mode_max
} img2img_mode_t;


typedef enum {
   brush_size_disabled,
   brush_size_1,
   brush_size_2,
   brush_size_4,
   brush_size_8,
   brush_size_16,
   brush_size_32,
   brush_size_64,
   brush_size_128,
   // 
   brush_size_count
} brush_size_t;


typedef enum {
    inpaint_original,
    inpaint_fill,
    inpaint_img2img,
    inpaint_wholepicture,
    inpaint_wholefill,
    // inpaint_none,
    // keep inpaint_mode_count at the end
    inpaint_mode_count
} inpaint_mode_t;



class ImageFrame {
public:
    ImageFrame(Fl_Group *parent, ImagePanel *img);
    ~ImageFrame();

    void alignComponents();
    bool enabled();
    img2img_mode_t get_mode();
    int get_brush_size();
    inpaint_mode_t get_inpaint_mode();
    void enable_mode();
    float get_strength();

    bool inpaint_enabled();
    void inpaint_enabled(bool enabled);
    void handle_event(int event, void *sender);

private:
    void combobox_selected();
    void configure_mask_color();
    void configure_mask_color_enabled();
    void pickup_palette_image();
    void pixelate_current_image();

protected:
    static void combobox_cb(Fl_Widget* widget, void *cbdata);
    void combobox_cb(Fl_Widget* widget);

private:
    bool inside_cb_ = false;
    bool inpaint_enabled_ = true;

private:
    Fl_Group *parent_;
    ImagePanel *img_;
    Fl_Choice *choice_mode_;
    Fl_Choice *choice_brush_size_;
    Fl_Choice *choice_inpaint_mode_;
    Fl_Float_Input *strength_input_;

    std::unique_ptr<Button> btnNewMask_;
    std::unique_ptr<Button> btnOpenMask_;
    std::unique_ptr<Button> btnColor_;
    std::unique_ptr<Button> btnUseColor_;
    std::unique_ptr<Button> btnPixelate_;
    std::unique_ptr<Button> btnFromPalette_;
    std::unique_ptr<Button> btnSegGDino_;
    std::unique_ptr<Button> btnSegSapiens_;

    img2img_mode_t mode_ = img2img_text;
    int brush_size_ = 16;
    inpaint_mode_t inpaint_mode_ = inpaint_original;
};

} // namespace editorium
