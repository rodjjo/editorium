
#include "components/xpm/xpm.h"

#include "misc/dialogs.h"
#include "websocket/tasks.h"
#include "windows/frames/controlnet_frame.h"

namespace editorium
{

    namespace {
       std::vector<std::pair<std::string, std::string> > controlnet_modes;
    }

ControlnetFrame::ControlnetFrame(Fl_Group *parent, ImagePanel *img, ImagePanel *reference) {
    parent_ = parent;
    img_ = img;
    reference_ = reference;
    mode_ = new Fl_Choice(0, 0, 1, 1, "Controlnet mode");
    btnOpenMask_.reset(new Button(xpm::image(xpm::img_24x24_folder), [this] () {
        open_mask();
    }));
    btnSaveMask_.reset(new Button(xpm::image(xpm::img_24x24_wallet), [this] () {
        save_mask();
    }));
    btnPreprocess_.reset(new Button(xpm::image(xpm::img_24x24_pinion), [this] () {
        pre_process();
    }));
    btnOpenMask_->tooltip("Open a pre-processed image");
    btnPreprocess_->tooltip("Pre-process the input image");
    btnSaveMask_->tooltip("Save the pre-processed image");
    
    mode_->align(FL_ALIGN_TOP_LEFT);
    mode_->add("Disabled");
    mode_->value(0);

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
    btnPreprocess_->position(left +5 , mode_->y() + mode_->h() + 5);
    btnPreprocess_->size(w - 10, 30);
    btnOpenMask_->position(left +5 , btnPreprocess_->y() + btnPreprocess_->h() + 5);
    btnOpenMask_->size(w - 10, 30);
    btnSaveMask_->position(left +5 , btnOpenMask_->y() + btnOpenMask_->h() + 5);
    btnSaveMask_->size(w - 10, 30);
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
        btnPreprocess_->show();
        if (parent_->visible_r()) {
            img_->show();
        }
    } else {
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
    mode_->clear();
    for (const auto & c : controlnet_modes) {
        if (supported_modes_.find(c.first) == supported_modes_.end() && c.first != "disabled") {
            continue;
        }
        mode_->add(c.second.c_str());
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
        for (const auto & c : controlnet_modes) {
            if (c.second == mode_->text()) {
                return c.first;
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
