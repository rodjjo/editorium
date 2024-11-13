#pragma once

#include <string>

#include <FL/Fl_Group.H>
#include <FL/Fl_Multiline_Input.H>
#include "components/button.h"
#include <FL/Fl_Window.H>

namespace editorium
{

class ChatbotWindow: public Fl_Window {
public:
    ChatbotWindow(const char *title, const std::string& context, const std::string& default_prompt);
    ~ChatbotWindow();
    bool confirmed();
    std::string get_system_prompt();
    void set_system_prompt(const std::string& prompt);

private:
    void align_component();
    void use_default_prompt();

private:
    std::string default_prompt_;

private:
    bool confirmed_ = false;
    Fl_Multiline_Input *system_prompt_;
    std::unique_ptr<Button> btnDefault_;
    std::unique_ptr<Button> btnOk_;
    std::unique_ptr<Button> btnCancel_;
};

std::string get_prompts_for_vision(const std::string& title, const std::string& context = "");
std::string get_prompts_for_chat(const std::string& title, const std::string& context = "");


    
} // namespace editoruim
