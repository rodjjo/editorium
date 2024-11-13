#include "components/xpm/xpm.h"
#include "misc/dialogs.h"

#include "misc/profiles.h"
#include "windows/chatbot_ui.h"



namespace editorium {
    

ChatbotWindow::ChatbotWindow(const char *title, const std::string& context, const std::string& default_prompt):
        Fl_Window(Fl::w() / 2 - 860 / 2, Fl::h() / 2 - 640 / 2, 860, 340, title) {
    default_prompt_ = default_prompt;
    system_prompt_ = new Fl_Multiline_Input(0, 0, 1, 1, "System prompt");
    btnDefault_.reset(new Button(xpm::image(xpm::img_24x24_text), [this] {
        use_default_prompt();
    }));
    btnOk_.reset(new Button(xpm::image(xpm::img_24x24_ok), [this] {
        std::string system_prompt = system_prompt_->value();
        if (!system_prompt.empty()) {
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
    btnDefault_->tooltip("Use a default prompt");
    
    chatbot_load_profile();
    system_prompt_->value(chatbot_profile_get_string({context, "system_prompt"}, default_prompt).c_str());

    set_modal();
    align_component();
}

ChatbotWindow::~ChatbotWindow() {
}

bool ChatbotWindow::confirmed() {
    return confirmed_;    
}

void ChatbotWindow::use_default_prompt() {
    system_prompt_->value(default_prompt_.c_str());
}

void ChatbotWindow::align_component() {
    system_prompt_->resize(10, 40, this->w() - 20, 250);

    btnDefault_->position(10, this->h() - 40);
    btnDefault_->size(100, 30);
    btnOk_->position(this->w() - 215, this->h() - 40);
    btnOk_->size(100, 30);
    btnCancel_->position(btnOk_->x() + btnOk_->w() + 2, btnOk_->y());
    btnCancel_->size(100, 30);
}

std::string ChatbotWindow::get_system_prompt() {
    return system_prompt_->value();
}


void ChatbotWindow::set_system_prompt(const std::string& prompt) {
    system_prompt_->value(prompt.c_str());
}


std::string get_prompts(const std::string& title,const std::string& context, const char *default_system_prompt) {
    std::string result;
    auto window = new ChatbotWindow(title.c_str(), context, default_system_prompt);
    window->show();
    while (true) {
        if (!window->visible_r()) {
            break;
        }
        Fl::wait();
    }
    if (window->confirmed()) {
        result = window->get_system_prompt();
        chatbot_profile_set_string({context, "system_prompt"}, result);
        chatbot_save_profile();
    }
    Fl::delete_widget(window);
    Fl::do_widget_deletion();
    return result;
}

std::string get_prompts_for_vision(const std::string& title, const std::string& context) {
    const char *default_system_prompt = "You look at the image and follow the instructions to describe it.";
    return get_prompts(title, context, default_system_prompt);
}

std::string get_prompts_for_chat(const std::string& title, const std::string& context) {
    const char *default_system_prompt = "You are an helpful assistant that convert comma separated tags into a detailed description.";
    return get_prompts(title, context, default_system_prompt);
}

} // namespace editorium
