#pragma once

#include <memory>
#include <FL/Fl_Window.H>
#include <FL/Fl_Tabs.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Int_Input.H>
#include <FL/Fl_Float_Input.H>
#include <FL/Fl_Check_Button.H>

#include "components/button.h"

namespace editorium
{
    
class SettingsWindow: public Fl_Window {
public:
    SettingsWindow();
    ~SettingsWindow();
    int handle(int event);
    void save_settings();
    void load_settings();

private:
    void alignComponents();

private:
    Fl_Tabs *tabs_;
    Fl_Group *page_params_;
    Fl_Group *page_directories_;
    Fl_Group *page_base_models_;
    Fl_Group *page_chat_bot_;
    Fl_Group *page_chat_vision_;
    Fl_Input *profiles_dir_;
    Fl_Input *sdxl_base_model_;
    Fl_Input *flux_base_model_;
    Fl_Input *server_url_;
    Fl_Input *sd35_base_model_;
    Fl_Check_Button *use_float16_;
    Fl_Check_Button *private_mode_;
    Fl_Check_Button *keep_models_;

    //chat bot
    Fl_Input *chat_bot_repo_id_;
    Fl_Input *chat_bot_model_name_;
    Fl_Input *chat_bot_template_;
    Fl_Int_Input *chat_bot_max_new_tokens_;
    Fl_Float_Input *chat_bot_temperature_;
    Fl_Float_Input *chat_bot_top_p_;
    Fl_Int_Input *chat_bot_top_k_;
    Fl_Float_Input *chat_bot_repetition_penalty_;
    Fl_Input *chat_bot_response_after_;
    // chat vision
    Fl_Input *chat_vision_repo_id_;
    Fl_Float_Input *chat_vision_temperature_;

    std::unique_ptr<Button> btnOk_;
    std::unique_ptr<Button> btnCancel_;
};

void edit_settings();

} // namespace editorium
