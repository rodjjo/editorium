#pragma once

#include <FL/Fl_Window.H>
#include <FL/Fl_Check_Button.H>

#include "components/button.h"
#include "components/image_panel.h"
#include "images/image.h"


namespace editorium {

class ImagePalleteWindow: public Fl_Window  {
 public:
    ImagePalleteWindow();
    virtual ~ImagePalleteWindow();
    image_ptr_t get_picked_image();    

 private:
    void go_next_image();
    void go_prior_image();
    void align_components();
    static void widget_cb(Fl_Widget* widget, void *cbdata);
    void widget_cb(Fl_Widget* widget);
    void update_title();
    void show_current_image();
    void save_current_image();

 private:
    bool confirmed_ = false;
    bool ignore_pinned_cb_ = false;
 
 private:
    size_t                  selected_index_  = 0;
    ImagePanel              *img_;
    Fl_Check_Button         *pinned_;
    std::unique_ptr<Button> btnSave_;
    std::unique_ptr<Button> btnPrior_;
    std::unique_ptr<Button> btnNext_;
    std::unique_ptr<Button> btnOk_;
    std::unique_ptr<Button> btnCancel_;
};


image_ptr_t pickup_image_from_palette();

    
} // namespace editorium
