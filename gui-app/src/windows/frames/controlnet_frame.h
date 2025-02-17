#pragma once

#include <string>
#include <set>
#include <memory>
#include <vector>

#include <FL/Fl_Group.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Float_Input.H>

#include "components/image_panel.h"
#include "components/button.h"

namespace editorium
{

class ControlnetFrame {
public:
    ControlnetFrame(Fl_Group *parent, ImagePanel *img, ImagePanel *reference, bool ip_adapter=false);
    ~ControlnetFrame();

    bool enabled();
    void alignComponents();
    std::string getModeStr();
    image_ptr_t getImage();
    float getStrength();

    void supported_modes(const std::set<std::string>& modes);

private:
    void pre_process();
    void open_mask();
    void save_mask();
    void load_modes();

protected:
    static void combobox_cb(Fl_Widget* widget, void *cbdata);
    void combobox_cb(Fl_Widget* widget);

private:
    bool ip_adapter_ = false;
    bool inside_cb_ = false;
    std::set<std::string>  supported_modes_;

private:
    Fl_Group *parent_;
    ImagePanel *img_;
    ImagePanel *reference_;
    Fl_Choice *mode_;
    Fl_Float_Input *strength_input_;
    std::unique_ptr<Button> btnPreprocess_;
    std::unique_ptr<Button> btnOpenMask_;
    std::unique_ptr<Button> btnSaveMask_;
};

} // namespace editorium
