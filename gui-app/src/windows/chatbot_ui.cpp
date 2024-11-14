#include "components/xpm/xpm.h"
#include "misc/dialogs.h"

#include "misc/profiles.h"
#include "windows/chatbot_ui.h"



namespace editorium {
    

ChatbotWindow::ChatbotWindow(
        const char *title, 
        const std::string& context, 
        const std::string& default_prompt,
        const std::string& default_user_prompt,
        bool enable_user_prompt
    ):
        Fl_Window(Fl::w() / 2 - 860 / 2, Fl::h() / 2 - 640 / 2, 860, enable_user_prompt ? 520 : 340, title) {
    default_prompt_ = default_prompt;
    default_user_prompt_ = default_user_prompt;
    system_prompt_ = new Fl_Text_Editor(0, 0, 1, 1, "System prompt");
    sys_prompt_buffer_ = new Fl_Text_Buffer();
    system_prompt_->buffer(sys_prompt_buffer_);
    system_prompt_->wrap_mode(Fl_Text_Display::WRAP_AT_BOUNDS, 5);
    if (enable_user_prompt) {
        user_prompt_ = new Fl_Text_Editor(0, 0, 1, 1, "User prompt");
        usr_prompt_buffer_ = new Fl_Text_Buffer();
        user_prompt_->buffer(usr_prompt_buffer_);
        user_prompt_->wrap_mode(Fl_Text_Display::WRAP_AT_BOUNDS, 5);
    }
    btnDefault_.reset(new Button(xpm::image(xpm::img_24x24_text), [this] {
        use_default_prompt();
    }));
    btnOk_.reset(new Button(xpm::image(xpm::img_24x24_ok), [this] {
        std::string system_prompt = system_prompt_->buffer()->text();
        std::string user_prompt = user_prompt_ != nullptr ? user_prompt_->buffer()->text() : "place_holder";
        if (!system_prompt.empty() && !user_prompt.empty()) {
            confirmed_ = true;
        } else {
            show_error("Please fill in the prompt!");
            return;
        }
        this->hide();
    }));
    btnCancel_.reset(new Button(xpm::image(xpm::img_24x24_abort), [this] {
        this->hide();        
    }));

    system_prompt_->align(FL_ALIGN_TOP_LEFT);
    if (enable_user_prompt) {
        user_prompt_->align(FL_ALIGN_TOP_LEFT);
    }
    btnDefault_->tooltip("Use a default prompt");
    
    chatbot_load_profile();
    system_prompt_->insert(chatbot_profile_get_string({context, "system_prompt"}, default_prompt).c_str());
    if (enable_user_prompt) {
        user_prompt_->insert(chatbot_profile_get_string({context, "user_prompt"}, default_user_prompt).c_str());
    }
    set_modal();
    align_component();
}

ChatbotWindow::~ChatbotWindow() {
    if(sys_prompt_buffer_) {
        delete sys_prompt_buffer_;
    }
    if (usr_prompt_buffer_) {
        delete usr_prompt_buffer_;
    }
}

bool ChatbotWindow::confirmed() {
    return confirmed_ && !display_result_;    
}

void ChatbotWindow::use_default_prompt() {
    system_prompt_->buffer()->text(default_prompt_.c_str());
    if (user_prompt_) {
        user_prompt_->buffer()->text(default_user_prompt_.c_str());
    }
}

void ChatbotWindow::align_component() {
    system_prompt_->resize(10, 40, this->w() - 20, 200);
    if (user_prompt_) {
        user_prompt_->resize(10, system_prompt_->y() + system_prompt_->h() + 20, this->w() - 20, 200);
    }

    btnDefault_->position(10, this->h() - 40);
    btnDefault_->size(100, 30);
    btnOk_->position(this->w() - 215, this->h() - 40);
    btnOk_->size(100, 30);
    btnCancel_->position(btnOk_->x() + btnOk_->w() + 2, btnOk_->y());
    btnCancel_->size(100, 30);
}

std::string ChatbotWindow::get_system_prompt() {
    return system_prompt_->buffer()->text();
}

std::string ChatbotWindow::get_user_prompt() {
    if (!user_prompt_) {
        return "";
    }
    return user_prompt_->buffer()->text();
}


void ChatbotWindow::set_system_prompt(const std::string& prompt) {
    system_prompt_->buffer()->text(prompt.c_str());
}

void ChatbotWindow::set_user_prompt(const std::string& prompt) {
    if (user_prompt_) {
        user_prompt_->buffer()->text(prompt.c_str());
    }
}

void ChatbotWindow::set_to_display_result() {
    system_prompt_->label("Result:");
    btnDefault_->hide();
    btnOk_->hide();
    display_result_ = true;
}


std::pair<std::string, std::string> get_prompts(
        const std::string& title, 
        const std::string& context, 
        const char *default_system_prompt,
        const char *default_user_prompt,
        bool enable_user_prompt) {
    std::pair<std::string, std::string> result;
    auto window = new ChatbotWindow(title.c_str(), context, default_system_prompt, default_user_prompt, enable_user_prompt);
    window->show();
    while (true) {
        if (!window->visible_r()) {
            break;
        }
        Fl::wait();
    }
    if (window->confirmed()) {
        result = std::make_pair(window->get_system_prompt(), window->get_user_prompt());
        chatbot_profile_set_string({context, "system_prompt"}, result.first);
        if (!result.second.empty()) {
            chatbot_profile_set_string({context, "user_prompt"}, result.second);
        }
        chatbot_save_profile();
    }
    Fl::delete_widget(window);
    Fl::do_widget_deletion();
    return result;
}

std::pair<std::string, std::string> get_prompts_for_vision(const std::string& title, const std::string& context) {
    const char *default_system_prompt = "You look at the image and follow the instructions to describe it.";
    const char *default_user_prompt = "Describe the image in detail.";
    return get_prompts(title, context, default_system_prompt, default_user_prompt, true);
}

std::string get_prompts_for_chat(const std::string& title, const std::string& context) {
    const char *default_system_prompt = "You are an helpful assistant that convert comma separated tags into a detailed description.";
    auto r = get_prompts(title, context, default_system_prompt, "", false);
    return r.first;
}

void chatbot_display_result(const std::string& title, const std::string& result) {
    auto window = new ChatbotWindow(title.c_str(), "", "", "", false);
    window->set_system_prompt(result);
    window->set_to_display_result();
    window->show();
    while (true) {
        if (!window->visible_r()) {
            break;
        }
        Fl::wait();
    }
    Fl::delete_widget(window);
    Fl::do_widget_deletion();
}

} // namespace editorium
