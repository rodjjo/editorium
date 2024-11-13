/*
 * Copyright (C) 2023 by Rodrigo Antonio de Araujo
 */
#include <map>
#include "components/xpm/xpm.h"

#pragma GCC diagnostic push
// there is no problem, i dont use it as a char *, i do as const char*
// i'd rather not modify .xpm files because it's generated by gimp and i keep it as it is.
#pragma GCC diagnostic ignored "-Wwrite-strings"

#include "components/xpm/data/24x24/new.xpm"
#include "components/xpm/data/24x24/door.xpm"
#include "components/xpm/data/24x24/folder.xpm"
#include "components/xpm/data/24x24/wallet.xpm"
#include "components/xpm/data/24x24/pinion.xpm"
#include "components/xpm/data/24x24/exit.xpm"
#include "components/xpm/data/24x24/component.xpm"
#include "components/xpm/data/24x24/close.xpm"
#include "components/xpm/data/24x24/computer.xpm"
#include "components/xpm/data/24x24/remove.xpm"
#include "components/xpm/data/24x24/erase.xpm"
#include "components/xpm/data/24x24/copy.xpm"
#include "components/xpm/data/24x24/picture.xpm"
#include "components/xpm/data/24x24/ok.xpm"
#include "components/xpm/data/24x24/abort.xpm"
#include "components/xpm/data/24x24/bee.xpm"
#include "components/xpm/data/24x24/magic_wand.xpm"
#include "components/xpm/data/24x24/back.xpm"
#include "components/xpm/data/24x24/forward.xpm"
#include "components/xpm/data/24x24/heart.xpm"
#include "components/xpm/data/24x24/medium_rating.xpm"
#include "components/xpm/data/24x24/new_document.xpm"
#include "components/xpm/data/24x24/flash_drive.xpm"
#include "components/xpm/data/24x24/green_pin.xpm"
#include "components/xpm/data/24x24/female.xpm"
#include "components/xpm/data/24x24/alien.xpm"
#include "components/xpm/data/24x24/load.xpm"
#include "components/xpm/data/24x24/left-right.xpm"
#include "components/xpm/data/24x24/up-down.xpm"
#include "components/xpm/data/24x24/redo.xpm"
#include "components/xpm/data/24x24/zoom.xpm"
#include "components/xpm/data/24x24/text_preview.xpm"

#pragma GCC diagnostic pop

namespace editorium{
namespace xpm {

std::map<xpm_t, const char * const*> xpm_db = {
    { img_24x24_exit, xpm_exit },
    { img_24x24_new, xpm_new },
    { img_24x24_new_document, xpm_new_document },
    { img_24x24_open_layer, xpm_component},
    { img_24x24_close, xpm_close},
    { img_24x24_open, xpm_door },
    { img_24x24_folder, xpm_folder },
    { img_24x24_wallet, xpm_wallet },
    { img_24x24_flash_drive, xpm_flash_drive },
    { img_24x24_pinion, xpm_pinion },
    { img_24x24_green_pin, xpm_green_pin },
    { img_24x24_settings, xpm_computer },
    { img_24x24_bee, xpm_bee },
    { img_24x24_remove, xpm_remove },
    { img_24x24_erase, xpm_erase},
    { img_24x24_copy, xpm_copy },
    { img_24x24_picture, xpm_picture },
    { img_24x24_ok, xpm_ok },
    { img_24x24_abort, xpm_abort },
    { img_24x24_magic_wand, xpm_magic_wand},
    { img_24x24_back, xpm_back },
    { img_24x24_forward, xpm_forward },
    { img_24x24_heart, xpm_heart },
    { img_24x24_medium_rating, xpm_medium_rating },
    { img_24x24_female, xpm_female },
    { img_24x24_alien, xpm_alien },
    { img_24x24_load, xpm_load },
    { img_24x24_left_right, xpm_left_right },
    { img_24x24_up_down, xpm_up_down },
    { img_24x24_redo,  xpm_redo },
    { img_24x24_zoom, xpm_zoom },
    { img_24x24_text_preview, xpm_text_preview },

};

std::shared_ptr<Fl_RGB_Image> image(xpm_t xpm_id, Fl_Color bg) {
    if (xpm_id == no_image) {
        static Fl_Color saved_bg  = 0;
        static unsigned char buffer[24 * 24 * 3] = {0};
        if (saved_bg != bg) {
            saved_bg = bg;
            unsigned char r;
            unsigned char g;
            unsigned char b;
            Fl::get_color(bg, r, g, b);
            size_t pixels_count = 24 * 24;
            for (int i = 0; i < pixels_count; i++) {
                buffer[i * 3] = r;
                buffer[i * 3 + 1] = g;
                buffer[i * 3 + 2] = b;
            }
        }
        return std::shared_ptr<Fl_RGB_Image>(new Fl_RGB_Image(buffer, 24, 24, 3, 0));
    }
    Fl_Pixmap image(xpm_db[xpm_id]);
    return std::shared_ptr<Fl_RGB_Image>(new Fl_RGB_Image(&image, bg));
}

} // namespace xpm
}  // namespace vcutter
