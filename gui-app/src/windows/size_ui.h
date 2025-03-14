#ifndef SRC_DIALOG_SIZE_DIALOG_H
#define SRC_DIALOG_SIZE_DIALOG_H

#include <string>
#include <memory>

#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Menu_Window.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Select_Browser.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Tabs.H>
#include <FL/Fl_Int_Input.H>
#include <FL/Fl_Float_Input.H>

#include "components/button.h"


namespace editorium
{

class SizeWindow : public Fl_Window  {
 public:
    SizeWindow(const char *title, bool single_value, bool is_float = false, bool match_proportion = true);
    virtual ~SizeWindow();
    void setInitialSize(int x, int y);
    void setInitialSizeFloat(float x, float y);
    void retriveSize(int *x, int *y);
    void retriveSizeFloat(float *x, float *y);
    bool run();
 private:
    void confirmOk();
    void proportionalChanged();
    static void valueChangedCb(Fl_Widget *wd, void *cbdata);
    void valueChangedCb(Fl_Widget *wd);
 private:
    std::string last_width_;
    std::string last_height_;
    float proportion_ = 1.0;
    bool changing_proportion_ = false;
    bool proportion_to_x_ = true;
    bool ok_confirmed_ = false;
    bool single_value_ = false;
    bool is_float_  = false;
    Fl_Input *width_;
    Fl_Input *height_;
    std::unique_ptr<Button> btn_proportion_;
    std::unique_ptr<Button> btn_ok_;
    std::unique_ptr<Button> btn_cancel_;
};

bool getSizeFromDialog(const char *title, int *x);
bool getSizeFromDialog(const char *title, int *x, int *y, bool match_proportion=true);
bool getSizeFromDialogFloat(const char *title, float *v);

    
} // namespace dexpert


#endif // SRC_DIALOG_SIZE_DIALOG_H