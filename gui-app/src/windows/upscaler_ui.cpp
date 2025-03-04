#include "components/xpm/xpm.h"
#include "misc/dialogs.h"
#include "windows/upscaler_ui.h"


namespace editorium
{

UpscalerWindow::UpscalerWindow(const char *title) : Fl_Window(0, 0, 300, 150, title) {
    this->position(Fl::w()/ 2 - this->w() / 2,  Fl::h() / 2 - this->h() / 2);
    this->set_modal();

    scale_ = new Fl_Float_Input(5, 25, 120, 30, "Weight:");
    face_weight_ = new Fl_Float_Input(scale_->x() + scale_->w() + 35, 25, 120, 30, "Face scale:");
    btn_restore_bg_ = new Fl_Check_Button(5, 60, 120, 30, "Restore BG");

    btn_ok_.reset(new Button(xpm::image(xpm::img_24x24_ok), [this] {
        confirmOk();
    }));
    btn_cancel_.reset(new Button(xpm::image(xpm::img_24x24_abort), [this] {
        this->hide();
    }));
    scale_->align(FL_ALIGN_TOP_LEFT);
    face_weight_->align(FL_ALIGN_TOP_LEFT);
    btn_ok_->tooltip("Confirm and close.");
    btn_cancel_->tooltip("Cancel and close.");
    btn_cancel_->size(60, 30);
    btn_ok_->size(60, 30);
    btn_cancel_->position(w() - 5 - btn_cancel_->w(), h() - 5 - btn_cancel_->h());
    btn_ok_->position(btn_cancel_->x() - 5 - btn_ok_->w(), btn_cancel_->y());

    scale_->value("2.0");
    face_weight_->value("1.0");
    btn_restore_bg_->value(1);
}

UpscalerWindow::~UpscalerWindow() {
}

float UpscalerWindow::get_scale() {
    float result = 2.0;
    sscanf(scale_->value(), "%f", &result);
    return result;
}

float UpscalerWindow::get_face_weight() {
    float result = 1.0;
    sscanf(face_weight_->value(), "%f", &result);
    return result;
}

bool UpscalerWindow::get_restore_bg() {
    return btn_restore_bg_->value() == 1;
}
 
void UpscalerWindow::confirmOk() {
    if (get_scale() < 1.0) {
        show_error("Invalid scale. The value should be greater or equal to 1.0");
        return;
    }
    if (get_scale() >= 4.0) {
        show_error("Invalid scale. The value should be less than 4.0");
        return;
    }
    if (get_face_weight() <= 0.0) {
        show_error("Invalid face weight. The value should be greater than 0.0");
        return;
    }
    if (get_face_weight() > 1.0) {
        show_error("Invalid face weight. The value should be less than 1.0");
        return;
    }
    ok_confirmed_ = true;
    hide();
}

bool UpscalerWindow::run() {
    this->show();
    while (this->shown()) {
        Fl::wait(0.01);
    }
    return ok_confirmed_;
}

bool get_gfpgan_upscaler_params(float &scale, float &face_weight, bool &restore_bg) {
    UpscalerWindow *wnd = new UpscalerWindow("Parameters for GFPGAN Upscaler");
    bool result = wnd->run();
    if (result) {
        scale = wnd->get_scale();
        face_weight = wnd->get_face_weight();
        restore_bg = wnd->get_restore_bg();
    }
    Fl::delete_widget(wnd);
    Fl::do_widget_deletion();
    return result;
}
    
} // namespace dexpert
