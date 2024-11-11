#pragma once

#include <string>
#include <FL/Fl_Window.H>
#include <FL/Fl_Check_Button.H>

#include "components/button.h"

namespace editorium
{
    
class SapiensClassesWindow: public Fl_Window {
public:
    SapiensClassesWindow();
    ~SapiensClassesWindow();
    std::string get_selected_classes();
    
private:
    void alignComponents();

private:
    bool confirmed_ = false;

private:
    Fl_Check_Button *check_classes_[28];

    std::unique_ptr<Button> btnOk_;
    std::unique_ptr<Button> btnCancel_;
};

std::string  select_sapien_classes();

} // namespace editorium
