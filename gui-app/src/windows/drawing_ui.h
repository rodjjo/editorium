#pragma once

#include <string>
#include <map>

#include <FL/Fl_Window.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Int_Input.H>
#include <FL/Fl_Multiline_Input.H>
#include <FL/Fl_Choice.H>

#include "components/button.h"
#include "components/image_panel.h"
#include "windows/frames/colorpalette_frame.h"
#include "windows/frames/embeddings_frame.h"
#include "messagebus/messagebus.h"
#include "images/image.h"

namespace editorium {


class DrawingWindow: public Fl_Window, public SubscriberThis {
 public:
    DrawingWindow(image_ptr_t reference_img);
    virtual ~DrawingWindow();
    image_ptr_t get_image();    

 protected:
    void dfe_handle_event(void *sender, event_id_t event, void *data) override;
    int handle(int event) override;

 private:
   void align_components();
   void toggle_settings();
   void generate_image(bool second_pass);
   void load_arch_models();
   void update_model_list();
   std::string get_arch();
   std::string get_model();
   void random_seed();
   int get_seed();
   void brush_size_selected();
   void insert_current_lora();
   void from_palette();
   void to_palette();
   void use_current_image();
   void reset_image();

 private:
    bool confirmed_ = false;
    bool ignore_pinned_cb_ = false;
    static void cb_widget(Fl_Widget *widget, void *data);
    void cb_widget(Fl_Widget *widget);
    

 private:
   bool validate();
   std::string positive_prompt();
   std::vector<std::string> get_loras();

 private:
   std::vector<std::pair<std::string, std::string> > arch_models_;

 private:
    size_t mask_version_ = 999;
    image_ptr_t image_;
    std::shared_ptr<ColorPaletteFrame> color_palette_;
    ImagePanel *image_panel_ = nullptr;
    Fl_Group *settings_panel_ = nullptr;
    Fl_Group *right_panel_ = nullptr;
    Fl_Group *color_pal_group_ = nullptr;
    Fl_Group  *lora_gp_ = nullptr;
    Fl_Int_Input *seed_input_;
    Fl_Multiline_Input *prompt_input_;
    Fl_Choice *brush_size_;
    Fl_Choice *arch_input_;
    Fl_Choice *model_input_;

    std::unique_ptr<EmbeddingFrame> loras_;
    std::unique_ptr<Button> btnPinSeed_;
    std::unique_ptr<Button> btnRandomSeed_;
    std::unique_ptr<Button> btnFirstPass_;
    std::unique_ptr<Button> btnSecondPass_;
    std::unique_ptr<Button> btnBtnResetImage_;
    std::unique_ptr<Button> btnSettings_;
    std::unique_ptr<Button> btnFromPalette_;
    std::unique_ptr<Button> btnToPalette_;
    std::unique_ptr<Button> btnUseCurrent_;
    std::unique_ptr<Button> btnOk_;
    std::unique_ptr<Button> btnCancel_;
};


image_ptr_t draw_image(image_ptr_t reference_image);
    
} // namespace editorium
