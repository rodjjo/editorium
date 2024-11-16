#pragma once

#include <FL/Fl_Window.H>
#include <FL/Fl_Tabs.H>
#include <FL/Fl_Group.H>
#include "components/button.h"
#include "components/image_panel.h"
#include "images/image.h"

namespace editorium {


class CopyRegionWindow: public Fl_Window  {
 public:
    CopyRegionWindow(image_ptr_t original_img, image_ptr_t new_img);
    virtual ~CopyRegionWindow();
    image_ptr_t get_merged_image();    

 private:
    void align_components();

 private:
    bool confirmed_ = false;
    bool ignore_pinned_cb_ = false;
 
 private:
    Fl_Tabs *tabs_;
    Fl_Group *page_original_;
    Fl_Group *page_new_img_;
    image_ptr_t original_img_;
    image_ptr_t new_img_;
    ImagePanel *original_panel_ = nullptr;
    ImagePanel *new_panel_ = nullptr;
    std::unique_ptr<Button> btnOk_;
    std::unique_ptr<Button> btnCancel_;
};


image_ptr_t copy_image_region(image_ptr_t original, image_ptr_t new_image);
    
} // namespace editorium
