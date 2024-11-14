#pragma once

#include <string>

#include <FL/Fl_Group.H>
#include <FL/Fl_Text_Editor.H>
#include "components/button.h"
#include <FL/Fl_Window.H>
#include <FL/Fl_Text_Buffer.H>

namespace editorium
{

class ChatbotWindow: public Fl_Window {
public:
    ChatbotWindow(
        const char *title, 
        const std::string& context, 
        const std::string& default_prompt,
        const std::string& default_user_prompt,
        bool enable_user_prompt);
    ~ChatbotWindow();
    bool confirmed();
    std::string get_system_prompt();
    std::string get_user_prompt();
    void set_system_prompt(const std::string& prompt);
    void set_user_prompt(const std::string& prompt);
    void set_to_display_result();

private:
    void align_component();
    void use_default_prompt();

private:
    std::string default_prompt_;
    std::string default_user_prompt_;

private:
    bool confirmed_ = false;
    bool display_result_ = false;
    Fl_Text_Editor *system_prompt_ = nullptr;
    Fl_Text_Editor *user_prompt_ = nullptr;
    Fl_Text_Buffer *sys_prompt_buffer_ = nullptr;
    Fl_Text_Buffer *usr_prompt_buffer_ = nullptr;
    std::unique_ptr<Button> btnDefault_;
    std::unique_ptr<Button> btnOk_;
    std::unique_ptr<Button> btnCancel_;
};

std::pair<std::string,std::string> get_prompts_for_vision(const std::string& title, const std::string& context = "");
std::string get_prompts_for_chat(const std::string& title, const std::string& context = "");
void chatbot_display_result(const std::string& title, const std::string& result);

} // namespace editorium
