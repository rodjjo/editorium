#pragma once

#include <memory>

#include <FL/Fl_Group.H>
#include <FL/Fl_Box.H>
#include "components/image_panel.h"
#include "components/button.h"


namespace editorium
{

class ResultFrame {
  public:
    ResultFrame(Fl_Group *parent, ImagePanel *img);
    ~ResultFrame();
    void alignComponents();
    void set_page_text(const char *value);

  private:
    Fl_Group *parent_;
    ImagePanel *img_;
    Fl_Box *page_;
    std::unique_ptr<Button> btn_previous_;
    std::unique_ptr<Button> btn_next_;
    std::unique_ptr<Button> btn_accept_;
    std::unique_ptr<Button> btn_accept_partially_;
    std::unique_ptr<Button> btn_to_palette_;
    std::unique_ptr<Button> btn_save_;
};

    
} // namespace editorium
