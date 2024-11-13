#pragma once

#include <memory>

#include <FL/Fl_RGB_Image.H>
#include <FL/Fl_Pixmap.H>


namespace editorium{
namespace xpm {

typedef enum {
    img_24x24_new = 0,
    img_24x24_new_document,
    img_24x24_open,
    img_24x24_folder,
    img_24x24_wallet,
    img_24x24_flash_drive,
    img_24x24_pinion,
    img_24x24_open_layer,
    img_24x24_remove,
    img_24x24_erase,
    img_24x24_close,
    img_24x24_settings,
    img_24x24_bee,
    img_24x24_copy,
    img_24x24_picture,
    img_24x24_exit,
    img_24x24_ok,
    img_24x24_abort,
    img_24x24_magic_wand,
    img_24x24_back,
    img_24x24_forward,
    img_24x24_green_pin,
    img_24x24_heart,
    img_24x24_medium_rating,
    img_24x24_female,
    img_24x24_alien,
    img_24x24_load,
    img_24x24_left_right,
    img_24x24_up_down,
    img_24x24_redo,
    img_24x24_zoom,
    img_24x24_text_preview,

    // keep no_image at the end
    no_image
} xpm_t;

std::shared_ptr<Fl_RGB_Image> image(xpm_t xpm_id,  Fl_Color bg=FL_GRAY);

} // namespace xpm
} // namespace dexpert

