
#include "components/xpm/xpm.h"

#include "misc/dialogs.h"
#include "websocket/tasks.h"
#include "windows/frames/controlnet_frame.h"

namespace editorium
{
    namespace {
       std::vector<std::pair<std::string, std::string> > controlnet_modes;
       std::vector<std::pair<std::string, std::string> > ip_adapter_modes;
    }

ControlnetFrame::ControlnetFrame(Fl_Group *parent, ImagePanel *img, ImagePanel *reference, bool ip_adapter) {
    parent_ = parent;
    parent_->begin();

    img_ = img;
    reference_ = reference;
    ip_adapter_ = ip_adapter;
    mode_ = new Fl_Choice(0, 0, 1, 1, ip_adapter ? "Adapter Model" : "Controlnet Model");
    strength_input_ = new Fl_Float_Input(0, 0, 1, 1, "Strength");
    btnOpenMask_.reset(new Button(xpm::image(xpm::img_24x24_folder), [this] () {
        open_mask();
    }));
    btnSaveMask_.reset(new Button(xpm::image(xpm::img_24x24_wallet), [this] () {
        save_mask();
    }));
    btnPreprocess_.reset(new Button(xpm::image(xpm::img_24x24_pinion), [this] () {
        pre_process();
    }));

    parent_->end();

    btnOpenMask_->tooltip("Open a pre-processed image");
    btnPreprocess_->tooltip("Pre-process the input image");
    btnSaveMask_->tooltip("Save the pre-processed image");

    mode_->align(FL_ALIGN_TOP_LEFT);
    mode_->add("Disabled");
    mode_->value(0);

    strength_input_->align(FL_ALIGN_TOP_LEFT);
    strength_input_->value("1.0");

    alignComponents();
    load_modes();
    mode_->callback(combobox_cb, this);
    combobox_cb(mode_);

}

void ControlnetFrame::alignComponents() {
    int left = parent_->x();
    int top = parent_->y();
    int w = img_->x() - parent_->x();
    int h = parent_->h();

    mode_->resize(left + 5, top + 25, w - 10, 30);
    strength_input_->resize(left + 5, mode_->y() + mode_->h() + 25, w - 10, 30);
    btnPreprocess_->position(left +5 , strength_input_->y() + strength_input_->h() + 5);
    btnPreprocess_->size(w - 10, 30);
    btnOpenMask_->position(left +5 , btnPreprocess_->y() + btnPreprocess_->h() + 5);
    btnOpenMask_->size(w - 10, 30);
    btnSaveMask_->position(left +5 , btnOpenMask_->y() + btnOpenMask_->h() + 5);
    btnSaveMask_->size(w - 10, 30);
    combobox_cb(mode_);
}

ControlnetFrame::~ControlnetFrame() {
    
}

void ControlnetFrame::combobox_cb(Fl_Widget* widget, void *cbdata) {
    static_cast<ControlnetFrame *>(cbdata)->combobox_cb(widget);
}

void ControlnetFrame::combobox_cb(Fl_Widget* widget) {
    if (inside_cb_)  {
        return;
    }
    inside_cb_ = true;
    if (enabled()) {
        btnOpenMask_->show();
        btnSaveMask_->show();
        strength_input_->show();
        if (!ip_adapter_) {
            btnPreprocess_->show();
        } else {
            btnPreprocess_->hide();
        }
        if (parent_->visible_r()) {
            img_->show();
        }
    } else {
        strength_input_->hide();
        btnOpenMask_->hide();
        btnSaveMask_->hide();
        btnPreprocess_->hide();
        img_->hide();
    }
    inside_cb_ = false;
}

bool ControlnetFrame::enabled() {
    return mode_->value() > 0;
}

float ControlnetFrame::getStrength() {
    float result = 1.0;
    char buffer[25] = "";
    bool changed = false;
    sscanf(strength_input_->value(), "%f", &result);

    if (result < 0.01) {
        result = 0.01;
        changed = true;
    } else if (result > 2.0) {
        result = 2.0;
        changed = true;
    }

    if (ip_adapter_ && result > 1.0) {
        result = 1.0;
        changed = true;
    }

    if (changed) {
        sprintf(buffer, "%.2f", result);
        strength_input_->value(buffer);
    }

    return result;
}


void ControlnetFrame::load_modes() {
    if (controlnet_modes.empty()) {
        controlnet_modes = {
            {"disabled", "Disabled"},
            {"canny", "Canny Edge Detection"},
            {"depth", "Depth Estimation"},
            {"pose", "Pose Estimation"},
            {"scribble", "Scribble"},
            {"segmentation", "Segmentation"},
            {"lineart", "Line Art"},
            {"mangaline", "Manga Line"},
            {"inpaint", "Inpainting"}
        };
    }
    if (ip_adapter_modes.empty()) {
        ip_adapter_modes = {
            {"disabled", "Disabled"},
            {"plus-face", "Plus Face"},
            {"full-face", "Full Face"},
            {"plus", "Plus"},
            {"common", "Common"},
            {"light", "Light"},
            {"vit", "Vit"},
        };
    }

    mode_->clear();
    if (ip_adapter_) {
        for (const auto & c : ip_adapter_modes) {
            if (supported_modes_.find(c.first) == supported_modes_.end() && c.first != "disabled") {
                continue;
            }
            mode_->add(c.second.c_str());
        }
    } else {
        for (const auto & c : controlnet_modes) {
            if (supported_modes_.find(c.first) == supported_modes_.end() && c.first != "disabled") {
                continue;
            }
            mode_->add(c.second.c_str());
        }
    }
    mode_->value(0);
}

void ControlnetFrame::supported_modes(const std::set<std::string>& modes) {
    if (modes == supported_modes_) {
        return;
    }
    supported_modes_ = modes;
    load_modes();
}

void ControlnetFrame::pre_process() {
    if (reference_->view_settings()->layer_count() < 1 || mode_->value() < 1) {
        return;
    }
    auto mode = getModeStr();
    if (mode == "segmentation" || mode == "inpaint") {
        show_error("Segmentation and inpaint do not have a pre-processor");
        return;
    }

    auto result = ws::diffusion::run_preprocessor(mode, { reference_->view_settings()->at(0)->getImage()->duplicate() });

    if (!result.empty()) {
        img_->view_settings()->clear_layers();
        img_->view_settings()->add_layer(result[0]);
    }
}

std::string ControlnetFrame::getModeStr() {
    if (mode_->value() > 0) {
        if (ip_adapter_) {
            for (const auto & c : ip_adapter_modes) {
                if (c.second == mode_->text()) {
                    return c.first;
                }
            }
        } else {
            for (const auto & c : controlnet_modes) {
                if (c.second == mode_->text()) {
                    return c.first;
                }
            }
        }
    }
    return std::string();
}

image_ptr_t ControlnetFrame::getImage() {
    image_ptr_t r;
    if (mode_->value() > 0 && img_->view_settings()->layer_count() > 0) {
        if (getModeStr() == "depth" || getModeStr() == "segmentation" || getModeStr() == "pose") {
            r = img_->view_settings()->at(0)->getImage()->duplicate();
        } else {
            r = img_->view_settings()->at(0)->getImage()->removeAlpha();
        }
    }
    
    return r;
}

void ControlnetFrame::open_mask() {
    std::string p = choose_image_to_open_fl("controlnet_mask_open");
    if (!p.empty()) {
        auto img = ws::filesystem::load_image(p.c_str());
        if (img) {
            img_->view_settings()->clear_layers();
            img_->view_settings()->add_layer(img);
        }
    }
}

void ControlnetFrame::save_mask() {
    if (img_->view_settings()->layer_count() < 1) {
        return;
    }
    auto img = img_->view_settings()->at(0)->getImage()->duplicate();
    std::string p = choose_image_to_save_fl("controlnet_mask_save");    
    if (!p.empty()) {
        ws::filesystem::save_image(p.c_str(), img, p.find(".png") != std::string::npos);
    }
}


} // namespace editorium
