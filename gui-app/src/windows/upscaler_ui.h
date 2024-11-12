#pragma once

#include <memory>

#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Menu_Window.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Tabs.H>
#include <FL/Fl_Int_Input.H>
#include <FL/Fl_Float_Input.H>

#include "components/button.h"


namespace editorium
{

class UpscalerWindow : public Fl_Window  {
 public:
    UpscalerWindow(const char *title);
    virtual ~UpscalerWindow();
    bool run();
    float get_scale();
    float get_face_weight();
    bool  get_restore_bg();
 private:
    void confirmOk();
 private:
    bool ok_confirmed_ = false;
    Fl_Float_Input *face_weight_;
    Fl_Float_Input *scale_;
    Fl_Check_Button *btn_restore_bg_;
    std::unique_ptr<Button> btn_ok_;
    std::unique_ptr<Button> btn_cancel_;
};

bool get_gfpgan_upscaler_params(float &scale, float &face_weight, bool &restore_bg);

} // namespace editorium

