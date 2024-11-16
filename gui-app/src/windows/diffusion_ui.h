#pragma once

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <FL/Fl_Window.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Select_Browser.H>

#include "images/image.h"

#include "messagebus/messagebus.h"
#include "windows/frames/controlnet_frame.h"
#include "windows/frames/prompt_frame.h"
#include "windows/frames/image_frame.h"
#include "windows/frames/results_frame.h"
#include "components/image_panel.h"
#include "components/button.h"

namespace editorium
{

typedef enum {
    page_type_prompt = 0,
    page_type_image,
    page_type_controlnet1,
    page_type_controlnet2,
    page_type_controlnet3,
    page_type_controlnet4,
    page_type_ip_adapter1,
    page_type_ip_adapter2,
    page_type_results,
    // keep page_type_count at the end
    page_type_count 
} page_type_t;
    
class DiffusionWindow: public Fl_Double_Window, public SubscriberThis {
public:
    DiffusionWindow();
    DiffusionWindow(ViewSettings *view_settings);
    ~DiffusionWindow();
    image_ptr_t get_current_image();
    bool was_confirmed();

protected:
    void resize(int x, int y, int w, int h) override;
    void dfe_handle_event(void *sender, event_id_t event, void *data) override;
    int handle(int event) override;

private:
    void after_constructor();
    void alignComponents();
    static void page_cb(Fl_Widget* widget, void *cbdata);
    void page_cb(Fl_Widget* widget);
    void show_current_page();
    void generate();
    void improve_prompt(bool second_pass);
    void interrogate_image();
    image_ptr_t choose_and_open_image(const char * scope);
    void choose_and_save_image(const char * scope, image_ptr_t image);
    const char *get_mode();
    void show_current_result();
    void accept_current_image();
    void accept_current_image_partial();
    void check_accept_current_image();
    void set_architecture_view();
    bool page_visible(page_type_t page);
    
private:
    bool image_generated_ = false;
    bool confirm_ = false;
    bool selecting_page_ = false;
    ViewSettings *view_settings_;
    Fl_Group *bottom_panel_;
    Fl_Group *right_panel_;
    Fl_Select_Browser *selector_;
    size_t result_index_ = 0;
    std::vector<page_type_t> visible_pages_;
    std::vector<image_ptr_t> results_;
    std::unique_ptr<ImageFrame> image_frame_;
    std::unique_ptr<PromptFrame> prompt_frame_;
    std::unique_ptr<ResultFrame> result_frame_;
    std::map<page_type_t, std::unique_ptr<ControlnetFrame> > control_frames_;
    std::map<page_type_t, Fl_Group *> pages_;
    std::map<page_type_t, std::string> titles_;
    std::map<page_type_t, ImagePanel *> images_;
    std::unique_ptr<Button> btnGenerate_;
    std::unique_ptr<Button> btnOk_;
    std::unique_ptr<Button> btnCancel_;
    int last_size_w_ = 0;
    int last_size_h_ = 0;
};

image_ptr_t generate_image(bool modal);
image_ptr_t generate_image(bool modal, ViewSettings* view_settings);

} // namespace editorium
