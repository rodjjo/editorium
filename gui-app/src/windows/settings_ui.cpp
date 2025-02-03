#include "components/xpm/xpm.h"
#include "windows/settings_ui.h"
#include "misc/dialogs.h"

#include "misc/config.h"

namespace editorium
{

SettingsWindow::SettingsWindow() : Fl_Window(Fl::w() / 2 - 640 / 2, Fl::h() / 2 - 480 / 2, 640, 480, "Configurations") {
    auto wnd_ = this;
    wnd_->begin();

    tabs_ = new Fl_Tabs(0, 0, 1, 1);
    tabs_->begin();
    page_params_ = new Fl_Group(0, 0, 1, 1, "Parameters");
    use_float16_ = new Fl_Check_Button(0, 0, 1, 1, "Use float16");
    private_mode_ = new Fl_Check_Button(0, 0, 1, 1, "Private mode");
    keep_models_ = new Fl_Check_Button(0, 0, 1, 1, "Keep models in the memory");
    page_params_->end();

    tabs_->begin();
    
    page_directories_ = new Fl_Group(0, 0, 1, 1, "Paths/Urls");
    profiles_dir_ = new Fl_Input(1, 1, 1, 1, "Directory to save the profiles");
    server_url_ = new Fl_Input(1, 1, 1, 1, "Server URL (ws://<host>:<port>)");
    page_directories_->end();
    
    tabs_->begin();
    page_base_models_ = new Fl_Group(0, 0, 1, 1, "Base models");
    sdxl_base_model_ = new Fl_Input(1, 1, 1, 1, "Base model for SDXL");
    flux_base_model_ = new Fl_Input(1, 1, 1, 1, "Base model for Flux");
    lumina_base_model_ = new Fl_Input(1, 1, 1, 1, "Base model for Lumina 2.0");
    sd35_base_model_ = new Fl_Input(1, 1, 1, 1, "Base model for SD35");
    arch_speed_models_ = new Fl_Input(1, 1, 1, 1, "Arch/Model for speed drawing");
    page_base_models_->end();

    tabs_->begin();
    page_chat_bot_ = new Fl_Group(0, 0, 1, 1, "Chatbot");
    chat_bot_repo_id_ = new Fl_Input(1, 1, 1, 1, "Repository ID (GPTQ)");
    chat_bot_model_name_ = new Fl_Input(1, 1, 1, 1, "Model name");
    chat_bot_template_ = new Fl_Multiline_Input(1, 1, 1, 1, "Template");
    chat_bot_max_new_tokens_ = new Fl_Int_Input(1, 1, 1, 1, "Max new tokens");
    chat_bot_temperature_ = new Fl_Float_Input(1, 1, 1, 1, "Temperature");
    chat_bot_top_p_ = new Fl_Float_Input(1, 1, 1, 1, "Top P");
    chat_bot_top_k_ = new Fl_Int_Input(1, 1, 1, 1, "Top K");
    chat_bot_repetition_penalty_ = new Fl_Float_Input(1, 1, 1, 1, "Repetition penalty");
    chat_bot_response_after_ = new Fl_Input(1, 1, 1, 1, "Response after");
    page_chat_bot_->end();

    tabs_->begin();
    page_chat_vision_ = new Fl_Group(0, 0, 1, 1, "Vision chatbot");
    chat_vision_repo_id_ = new Fl_Input(1, 1, 1, 1, "Repository ID");
    chat_vision_temperature_ = new Fl_Float_Input(1, 1, 1, 1, "Temperature");
    page_chat_vision_->end();

    wnd_->begin();
    btnOk_.reset(new Button(xpm::image(xpm::img_24x24_ok), [this] {
        this->save_settings();
        this->hide();
    }));
    btnCancel_.reset(new Button(xpm::image(xpm::img_24x24_abort), [this] {
        this->hide();        
    }));

    wnd_->end();
    wnd_->set_modal();

    btnOk_->tooltip("Save the configuration");
    btnCancel_->tooltip("Discart the changes");
    keep_models_->tooltip("Move the model to RAM when it exchanges between inpaint and normal model");

    profiles_dir_->align(FL_ALIGN_TOP_LEFT);
    server_url_->align(FL_ALIGN_TOP_LEFT);
    sd35_base_model_->align(FL_ALIGN_TOP_LEFT);
    flux_base_model_->align(FL_ALIGN_TOP_LEFT);
    lumina_base_model_->align(FL_ALIGN_TOP_LEFT);
    sdxl_base_model_->align(FL_ALIGN_TOP_LEFT);
    arch_speed_models_->align(FL_ALIGN_TOP_LEFT);

    chat_bot_repo_id_->align(FL_ALIGN_TOP_LEFT);
    chat_bot_model_name_->align(FL_ALIGN_TOP_LEFT);
    chat_bot_template_->align(FL_ALIGN_TOP_LEFT);
    chat_bot_max_new_tokens_->align(FL_ALIGN_TOP_LEFT);
    chat_bot_temperature_->align(FL_ALIGN_TOP_LEFT);
    chat_bot_top_p_->align(FL_ALIGN_TOP_LEFT);
    chat_bot_top_k_->align(FL_ALIGN_TOP_LEFT);
    chat_bot_repetition_penalty_->align(FL_ALIGN_TOP_LEFT);
    chat_bot_response_after_->align(FL_ALIGN_TOP_LEFT);

    chat_vision_repo_id_->align(FL_ALIGN_TOP_LEFT);
    chat_vision_temperature_->align(FL_ALIGN_TOP_LEFT);

    arch_speed_models_->tooltip("Comma separated format: arch:model,arch2:model2  (one model per architecture)");

    alignComponents();
}

SettingsWindow::~SettingsWindow() {
}

void SettingsWindow::save_settings() {
    auto cfg = get_config();
    cfg->profiles_dir(profiles_dir_->value());
    cfg->server_url(server_url_->value());
    cfg->sdxl_base_model(sdxl_base_model_->value());
    cfg->flux_base_model(flux_base_model_->value());
    cfg->lumina_base_model(lumina_base_model_->value());
    cfg->sd35_base_model(sd35_base_model_->value());
    cfg->arch_speed_model(arch_speed_models_->value());
    cfg->use_float16(use_float16_->value() == 1);
    cfg->private_mode(private_mode_->value() == 1);
    cfg->keep_in_memory(keep_models_->value() == 1);
    cfg->chat_bot_repo_id(chat_bot_repo_id_->value());
    cfg->chat_bot_model_name(chat_bot_model_name_->value());
    cfg->chat_bot_template(chat_bot_template_->value());
    int max_new_tokens = 512;
    sscanf(chat_bot_max_new_tokens_->value(), "%d", &max_new_tokens);
    cfg->chat_bot_max_new_tokens(max_new_tokens);
    cfg->chat_bot_temperature(atof(chat_bot_temperature_->value()));
    cfg->chat_bot_top_p(atof(chat_bot_top_p_->value()));
    float top_k = 0;
    sscanf(chat_bot_top_k_->value(), "%f", &top_k);
    cfg->chat_bot_top_k(top_k);
    float repetition_penalty = 1;
    sscanf(chat_bot_repetition_penalty_->value(), "%f", &repetition_penalty);
    cfg->chat_bot_repetition_penalty(repetition_penalty);
    cfg->chat_bot_response_after(chat_bot_response_after_->value());
    cfg->chat_vision_repo_id(chat_vision_repo_id_->value());
    cfg->chat_vision_temperature(atof(chat_vision_temperature_->value()));


    cfg->save();
}

void SettingsWindow::load_settings() {
    auto cfg = get_config();
    cfg->load();
    profiles_dir_->value(cfg->profiles_dir().c_str());
    server_url_->value(cfg->server_url().c_str());
    sdxl_base_model_->value(cfg->sdxl_base_model().c_str());
    flux_base_model_->value(cfg->flux_base_model().c_str());
    lumina_base_model_->value(cfg->lumina_base_model().c_str());
    sd35_base_model_->value(cfg->sd35_base_model().c_str());
    arch_speed_models_->value(cfg->arch_speed_model().c_str());
    use_float16_->value((int) cfg->use_float16());
    private_mode_->value((int) cfg->private_mode());
    keep_models_->value((int) cfg->keep_in_memory());
    char buffer[25] = "";
    sprintf(buffer, "%d", cfg->chat_bot_max_new_tokens());
    chat_bot_max_new_tokens_->value(buffer);
    sprintf(buffer, "%0.3f", cfg->chat_bot_temperature());
    chat_bot_temperature_->value(buffer);
    sprintf(buffer, "%0.3f", cfg->chat_bot_top_p());
    chat_bot_top_p_->value(buffer);
    sprintf(buffer, "%d", cfg->chat_bot_top_k());
    chat_bot_top_k_->value(buffer);
    sprintf(buffer, "%0.3f", cfg->chat_bot_repetition_penalty());
    chat_bot_repetition_penalty_->value(buffer);
    chat_bot_repo_id_->value(cfg->chat_bot_repo_id().c_str());
    chat_bot_model_name_->value(cfg->chat_bot_model_name().c_str());
    chat_bot_template_->value(cfg->chat_bot_template().c_str());
    chat_bot_response_after_->value(cfg->chat_bot_response_after().c_str());
    chat_vision_repo_id_->value(cfg->chat_vision_repo_id().c_str());
    sprintf(buffer, "%0.3f", cfg->chat_vision_temperature());
    chat_vision_temperature_->value(buffer);

}

int SettingsWindow::handle(int event) {
    switch (event) {
        case FL_KEYDOWN:
        case FL_KEYUP: {
            if (Fl::event_key() == FL_Escape) {
                return  1;
            }
            if (Fl::event_key() == (FL_F + 4) && (Fl::event_state() & FL_ALT) != 0) {
                return  1; // Do not allow ALT + F4
            }
        }
        break;
    }
    return Fl_Window::handle(event);
}

void SettingsWindow::alignComponents() {
    ((Fl_Widget *) tabs_)->resize(0, 0, this->w(), this->h() - 50);
    
    page_params_->resize(tabs_->x(), tabs_->y() + 30, tabs_->w(), tabs_->h() - 22);
    page_directories_->resize(tabs_->x(), tabs_->y() + 30, tabs_->w(), tabs_->h() - 22);
    page_base_models_->resize(tabs_->x(), tabs_->y() + 30, tabs_->w(), tabs_->h() - 22);
    page_chat_bot_->resize(tabs_->x(), tabs_->y() + 30, tabs_->w(), tabs_->h() - 22);
    page_chat_vision_->resize(tabs_->x(), tabs_->y() + 30, tabs_->w(), tabs_->h() - 22);

    int left = tabs_->x() + 10;
    int top = tabs_->y() + 55;
    int height = 30;

    btnOk_->position(this->w() - 215, this->h() - 40);
    btnOk_->size(100, 30);
    btnCancel_->position(btnOk_->x() + btnOk_->w() + 2, btnOk_->y());
    btnCancel_->size(100, 30);

    use_float16_->resize(left, top, 200, height);
    private_mode_->resize(use_float16_->x() + use_float16_->w() + 5, top, 200, height);
    keep_models_->resize(left, private_mode_->y() + private_mode_->h() + 5, 200, height);

    profiles_dir_->resize(left, top, page_directories_->w() - 20, height);
    server_url_->resize(left, profiles_dir_->y() + profiles_dir_->h() + 25, page_directories_->w() - 20, height);

    sdxl_base_model_->resize(left, top, page_base_models_->w() - 20, height);
    flux_base_model_->resize(left, sdxl_base_model_->y() + sdxl_base_model_->h() + 25, page_base_models_->w() - 20, height);
    lumina_base_model_->resize(left, flux_base_model_->y() + flux_base_model_->h() + 25, page_base_models_->w() - 20, height);
    sd35_base_model_->resize(left, lumina_base_model_->y() + lumina_base_model_->h() + 25, page_base_models_->w() - 20, height);
    arch_speed_models_->resize(left, sd35_base_model_->y() + sd35_base_model_->h() + 25, page_base_models_->w() - 20, height);
    
    chat_bot_repo_id_->resize(left, top, page_chat_bot_->w() - 20, height);
    chat_bot_model_name_->resize(left, chat_bot_repo_id_->y() + chat_bot_repo_id_->h() + 25, page_chat_bot_->w() - 20, height);
    chat_bot_template_->resize(left, chat_bot_model_name_->y() + chat_bot_model_name_->h() + 25, page_chat_bot_->w() - 20, height * 3);
    chat_bot_max_new_tokens_->resize(left, chat_bot_template_->y() + chat_bot_template_->h() + 25, (page_chat_bot_->w() - 20) / 3, height);
    chat_bot_temperature_->resize(chat_bot_max_new_tokens_->x() + chat_bot_max_new_tokens_->w() + 5, chat_bot_max_new_tokens_->y(), chat_bot_max_new_tokens_->w(), height);
    chat_bot_top_p_->resize(chat_bot_temperature_->x() + chat_bot_temperature_->w() + 5, chat_bot_temperature_->y(), chat_bot_temperature_->w() - 10, height);
    chat_bot_top_k_->resize(left, chat_bot_max_new_tokens_->y() + chat_bot_max_new_tokens_->h() + 25, (page_chat_bot_->w() - 20) / 3, height);
    chat_bot_repetition_penalty_->resize(chat_bot_top_k_->x() + chat_bot_top_k_->w() + 5, chat_bot_top_k_->y(), chat_bot_top_k_->w(), height);
    chat_bot_response_after_->resize(chat_bot_top_k_->x(), chat_bot_repetition_penalty_->y() + chat_bot_repetition_penalty_->h() + 25, page_chat_bot_->w() - 20, height);

    chat_vision_repo_id_->resize(left, top, (page_chat_vision_->w() - 20) / 2, height);
    chat_vision_temperature_->resize(chat_vision_repo_id_->x(), chat_vision_repo_id_->y() + chat_vision_repo_id_->h() + 25, chat_vision_repo_id_->w(), height);
}

void edit_settings() {
    auto window = new SettingsWindow();
    window->load_settings();
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
