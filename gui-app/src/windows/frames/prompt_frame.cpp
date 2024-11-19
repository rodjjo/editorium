#include "websocket/tasks.h"

#include "components/xpm/xpm.h"
#include "misc/profiles.h"

#include "misc/dialogs.h"
#include "windows/frames/prompt_frame.h"

namespace editorium
{

namespace {
    const char *resize_mode_texts[resize_mode_count] = {
        "Do not scale down the image",
        "Scale down to fit in 512x512",
        "Scale down to fit in 768x768",
        "Scale down to fit in 1024x1024",
        "Scale down to fit in 1280x1280"
    };
    std::string last_prompt;
}

PromptFrame::PromptFrame(Fl_Group *parent) : SubscriberThis({
    event_prompt_lora_selected,
    event_prompt_textual_selected
})  {
    parent_ = parent;

    positive_input_ = new Fl_Multiline_Input(0, 0, 1, 1, "Positive prompt");
    negative_input_ = new Fl_Multiline_Input(0, 0, 1, 1, "Negative Prompt");

    btn_improve_.reset(new Button(xpm::image(xpm::img_24x24_list), [this] {
        publish_event(this, event_prompt_improve_requested, nullptr);
    }));

    btn_improve2_.reset(new Button(xpm::image(xpm::img_24x24_list), [this] {
        publish_event(this, event_prompt_improve_requested2, nullptr);
    }));

    btn_interrogate_.reset(new Button(xpm::image(xpm::img_24x24_question), [this] {
        publish_event(this, event_prompt_interrogate_requested, nullptr);
    }));

    seed_input_ = new Fl_Int_Input(0, 0, 1, 1, "Seed");
    batch_input_ = new Fl_Int_Input(0, 0, 1, 1, "Batch size");
    steps_input_ = new Fl_Int_Input(0, 0, 1, 1, "Steps");
    guidance_input_ = new Fl_Float_Input(0, 0, 1, 1, "CFG");
    width_input_ = new Fl_Int_Input(0, 0, 1, 1, "Width");
    height_input_ = new Fl_Int_Input(0, 0, 1, 1, "Height");
    arch_input_ = new Fl_Choice(0, 0, 1, 1, "Architecture");
    models_input_ = new Fl_Choice(0, 0, 1, 1, "Model");
    schedulers_ =  new Fl_Choice(0, 0, 1, 1, "Scheduler");
    resizeModes_ =  new Fl_Choice(0, 0, 1, 1, "Resize mode");
    use_lcm_lora_ = new Fl_Check_Button(0, 0, 1, 1, "Use LCM lora");
    use_tiny_vae_ = new Fl_Check_Button(0, 0, 1, 1, "Use Tiny AutoEncoder");
    correct_colors_ = new Fl_Check_Button(0, 0, 1, 1, "Correct colors");
    ensure_min_512_ = new Fl_Check_Button(0, 0, 1, 1, "Ensure min 512x512 size");
    lora_gp_ = new Fl_Group(0, 0, 1, 1);
    loras_.reset(new EmbeddingFrame(true, lora_gp_));
    lora_gp_->end();
    emb_gp_ = new Fl_Group(0, 0, 1, 1);
    embeddings_.reset(new EmbeddingFrame(false, emb_gp_));
    emb_gp_->end();
    
    lora_gp_->box(FL_DOWN_BOX);
    emb_gp_->box(FL_DOWN_BOX);

    positive_input_->align(FL_ALIGN_TOP_LEFT);
    negative_input_->align(FL_ALIGN_TOP_LEFT);
    seed_input_->align(FL_ALIGN_TOP_LEFT);
    batch_input_->align(FL_ALIGN_TOP_LEFT);
    steps_input_->align(FL_ALIGN_TOP_LEFT);
    guidance_input_->align(FL_ALIGN_TOP_LEFT);
    width_input_->align(FL_ALIGN_TOP_LEFT);
    height_input_->align(FL_ALIGN_TOP_LEFT);
    models_input_->align(FL_ALIGN_TOP_LEFT);
    arch_input_->align(FL_ALIGN_TOP_LEFT);
    schedulers_->align(FL_ALIGN_TOP_LEFT);
    resizeModes_->align(FL_ALIGN_TOP_LEFT);
    
    seed_input_->value("-1");
    batch_input_->value("1");
    steps_input_->value("25");
    guidance_input_->value("7.5");
    width_input_->value("512");
    height_input_->value("512");
    correct_colors_->value(0);

    for (int i = 0; i < resize_mode_count; i++) {
        resizeModes_->add(resize_mode_texts[i]);
    }
    resizeModes_->value(resize_fit_1024x1024);
    resizeModes_->tooltip("Scale down the image before processing it");
    
    architectures_ = ws::diffusion::list_architectures();
    for (auto &arch: architectures_) {
        arch_input_->add(arch.second.c_str());
    }

    arch_input_->value(0);
    arch_input_->tooltip("Select the architecture to use");
    
    use_lcm_lora_->callback(widget_cb, this);
    arch_input_->callback(widget_cb, this);
    models_input_->callback(widget_cb, this);

    correct_colors_->tooltip("On inpaiting operation, correct colors in the output image");
    ensure_min_512_->tooltip("Ensure resizing images to at least 512x512 before doing img2img or inpainting operations");
    btn_improve_->tooltip("Uses a LLM to improve the positive prompt.");
    btn_improve2_->tooltip("Uses a LLM to improve the positive prompt. (SECOND PASS)");
    btn_interrogate_->tooltip("Use a multimodal model to look at the current image and describe it.");

    positive_input_->value(last_prompt.c_str());

    alignComponents();

    prompt_load_profile();
}

PromptFrame::~PromptFrame() {

}

void PromptFrame::widget_cb(Fl_Widget* widget, void *cbdata) {
    static_cast<PromptFrame *>(cbdata)->widget_cb(widget);
}

void PromptFrame::widget_cb(Fl_Widget* widget) {
    if (widget == use_lcm_lora_) {
        if (use_lcm_lora()) {
            if (get_steps() > 8) {
                steps_input_->value("4");
            }
            if (get_cfg() > 2.0) {
                guidance_input_->value("2.0");
            }
            if (get_scheduler() != "LCMScheduler") {
                auto idx = schedulers_->find_index("LCMScheduler");
                if (idx >= 0) {
                    schedulers_->value(idx);
                }
            }
        } else {
            if (get_steps() < 20) {
                steps_input_->value("25");
            }
            if (get_cfg() < 6.0) {
                guidance_input_->value("7.5");
            }
            if (get_scheduler() == "LCMScheduler") {
                auto idx = schedulers_->find_index("EulerAncestralDiscreteScheduler");
                if (idx >= 0) {
                    schedulers_->value(idx);
                }
            }

        }
    } else if (widget == arch_input_) {
        refresh_models();
        publish_event(this, event_prompt_architecture_selected, nullptr);
    } else if (widget == models_input_) {
        // publish_event(this, event_prompt_model_selected, nullptr);
        from_profile();
    }
}

void PromptFrame::alignComponents() {
    int sx = parent_->x(), sy = parent_->y();
    int pw = parent_->w(), ph = parent_->h();
    if (pw > 860) {
        pw = 860;
        sx = sx + (parent_->w() - 860);
    }
    positive_input_->resize(sx + 5, sy + 35, pw - 70, 50);
    btn_improve_->position(positive_input_->x() + positive_input_->w() + 5, positive_input_->y());
    btn_improve_->size(28, 28);
    btn_improve2_->position(btn_improve_->x() + btn_improve_->w() + 2, btn_improve_->y());
    btn_improve2_->size(28, 28);
    btn_interrogate_->position(btn_improve_->x(), btn_improve_->y() + btn_improve_->h() + 2);
    btn_interrogate_->size(28, 28);
    negative_input_->resize(sx + 5, positive_input_->y() + 75, positive_input_->w(), positive_input_->h());
    seed_input_->resize(sx + 5, negative_input_->y() + 75, (pw - 20) / 3, 20);
    batch_input_->resize(seed_input_->x() + seed_input_->w() + 5, seed_input_->y(), seed_input_->w(), seed_input_->h());
    steps_input_->resize(batch_input_->x() + batch_input_->w() + 5, batch_input_->y(), batch_input_->w(), batch_input_->h());
    guidance_input_->resize(sx + 5, steps_input_->y() + 45, steps_input_->w(), steps_input_->h());
    width_input_->resize(guidance_input_->x() + guidance_input_->w() + 5, guidance_input_->y(), guidance_input_->w(), guidance_input_->h());
    height_input_->resize(width_input_->x() + width_input_->w() + 5, width_input_->y(), width_input_->w(), width_input_->h());
    arch_input_->resize(sx + 5, height_input_->y() + 45, (pw - 15) / 2, height_input_->h());
    models_input_->resize(arch_input_->x() + arch_input_->w() + 5, arch_input_->y(), (pw - 15) / 2, height_input_->h());
    use_lcm_lora_->resize(sx + 5, models_input_->y() + models_input_->h() + 5, 160, 20);
    use_tiny_vae_->resize(use_lcm_lora_->x() + use_lcm_lora_->w() + 5, use_lcm_lora_->y(), use_lcm_lora_->w(), use_lcm_lora_->h());
    correct_colors_->resize(use_tiny_vae_->x() + use_tiny_vae_->w() + 10, use_tiny_vae_->y(), use_tiny_vae_->w(), use_tiny_vae_->h());
    ensure_min_512_->resize(correct_colors_->x() + correct_colors_->w() + 10, correct_colors_->y(), correct_colors_->w(), correct_colors_->h());
    schedulers_->resize(sx + 5, use_tiny_vae_->y() + use_tiny_vae_->h() + 20, models_input_->w(), models_input_->h());
    resizeModes_->resize(schedulers_->x() +  schedulers_->w() + 5, schedulers_->y(), models_input_->w(), models_input_->h());

    int embedding_pos_y = resizeModes_->y() + resizeModes_->h() + 15;
    lora_gp_->resize(sx + 5, embedding_pos_y, (pw - 15) / 2, ph - embedding_pos_y - 5);
    emb_gp_->resize(lora_gp_->x() + lora_gp_->w() + 5, embedding_pos_y, lora_gp_->w(), lora_gp_->h());

    loras_->alignComponents();
    embeddings_->alignComponents();
}

void PromptFrame::positive_prompt(const std::string& value, bool keep_loras) {
    auto prompt = value;
    if (keep_loras) {
        auto lora = get_loras();
        for (auto & l : lora) {
            printf("keeping lora: %s\n", l.c_str());
            prompt += " <lora:" + l + ">";
        }
    }
    positive_input_->value(value.c_str());
}

std::string PromptFrame::positive_prompt() {
    std::string result = positive_input_->value();
    last_prompt = result;
    size_t lpos = result.find("<lora:");
    while (lpos != result.npos) {
        size_t rpos = result.find(">", lpos);
        if (rpos == result.npos) {
            rpos = result.size()-1;
        }
        result = result.substr(0, lpos) + result.substr(rpos + 1);
        lpos = result.find("<lora:");
    }
    return result;
}

std::vector<std::string> PromptFrame::get_loras() {
    std::vector<std::string> result;
    std::string text = positive_input_->value();
    size_t lpos = text.find("<lora:");
    while (lpos != text.npos) {
        size_t rpos = text.find(">", lpos);
        if (rpos == text.npos) {
            rpos = text.size()-1;
        }
        result.push_back(text.substr(lpos + 6, rpos - lpos - 6));
        lpos = text.find("<lora:", rpos);
    }
    return result;
}

std::string PromptFrame::negative_prompt() {
    return negative_input_->value();
}

std::string PromptFrame::get_model() {
    if (models_input_->value() == 0) {
        ws::diffusion::architecture_features_t features = ws::diffusion::get_architecture_features(get_arch());
        if (features.support_base_model) {
            return "";
        }
    }
    if (models_input_->value() >= 0 ) {
        return models_input_->text(models_input_->value());
    }
    return std::string();
}

std::string PromptFrame::get_arch() {
    if (arch_input_->value() >= 0) {
        return architectures_[arch_input_->value()].first;
    }
    return "sd15";
}

std::string PromptFrame::get_scheduler() {
    if (schedulers_->value() >= 0) {
        return schedulers_->text(schedulers_->value());
    }
    return std::string();
}

int PromptFrame::get_seed() {
    int result = -1;
    sscanf(seed_input_->value(), "%d", &result);
    return result;
}

int PromptFrame::get_batch_size() {
    int result = 1;
    sscanf(batch_input_->value(), "%d", &result);
    
    if (result < 1) {
        result = 1;
    } else if (result > 8) {
        result = 8;
    } else {
        return result;
    }
    char buffer[25] = "";
    sprintf(buffer, "%d", result);
    batch_input_->value(buffer);
    return result;
}

int PromptFrame::get_steps() {
    int result = 30;
    sscanf(steps_input_->value(), "%d", &result);
    return result;
}

float PromptFrame::get_cfg() {
    float result = 7.5;
    sscanf(guidance_input_->value(), "%f", &result);
    return result;
}

int PromptFrame::get_width() {
    int result = 512;
    sscanf(width_input_->value(), "%d", &result);
    return result;
}

int PromptFrame::get_height() {
    int result = 512;
    sscanf(height_input_->value(), "%d", &result);
    return result;
}

bool PromptFrame::validate() {
    auto loras = get_loras();
    for (size_t i = 0; i < loras.size(); i++) {
        size_t pos = loras[i].find(":");
        if (pos == std::string::npos) {
            show_error(("Invalid Lora: " + loras[i]).c_str());
            return false;
        }
        std::string name = loras[i].substr(0, pos);
        if (!loras_->contains(name)) {
            show_error(("Lora not supported by this architecture: " + name).c_str());
            return false;
        }
    }
    return true;
}

bool PromptFrame::use_lcm_lora() {
    return use_lcm_lora_->value() != 0;
}

bool PromptFrame::use_tiny_vae() {
    return use_tiny_vae_->value() != 0;
}

bool PromptFrame::get_correct_colors() {
    return correct_colors_->value() != 0;
}

void PromptFrame::refresh_models() {
    auto model_list = ws::models::list_models(get_arch(), false);
    ws::diffusion::architecture_features_t features = ws::diffusion::get_architecture_features(get_arch());
    models_input_->clear();
    if (features.support_base_model) {
        models_input_->add("Base model (see settings)");
    }
    for (auto & m : model_list) {
        models_input_->add(m.c_str());
    }
    if (models_input_->size() > 0) {
        models_input_->value(0);
    } else {
        models_input_->value(-1);
        models_input_->redraw();
    }

    std::vector<std::string> scheduler_list = { "EulerAncestralDiscreteScheduler"};
    schedulers_->clear();
    for (auto & s : scheduler_list) {
        schedulers_->add(s.c_str());
    }
    schedulers_->value(0);

    embeddings_->refresh_models(get_arch());
    loras_->refresh_models(get_arch());

    auto model_name = prompt_profile_get_string({get_arch(), "model_name"}, get_model());
    // sets the model selected to the model name
    if (!model_name.empty()) {
        auto idx = models_input_->find_index(model_name.c_str());
        if (idx >= 0) {
            models_input_->value(idx);
        }
    }
    
    from_profile();
}

void PromptFrame::dfe_handle_event(void *sender, event_id_t event, void *data) {
    switch (event)
    {
    case event_prompt_lora_selected:
        if (sender == loras_.get()) {
            insert_current_lora();
        }
        break;
    
    case event_prompt_textual_selected:
        if (sender == embeddings_.get()) {
            insert_current_textual();
        }
        break;
    }
}


void PromptFrame::insert_current_textual() {
    std::string text = positive_input_->value();
    auto current_concept = embeddings_->getSelected();
    if (current_concept.name.empty()) {
        return;
    }
    if (text.find(current_concept.name) == text.npos) {
        text += " ";
        text += current_concept.name;
    }
    positive_input_->value(text.c_str());
}

void PromptFrame::insert_current_lora() {
    std::string text = positive_input_->value();
    auto current_concept = loras_->getSelected();
    if (current_concept.name.empty()) {
        return;
    }
    current_concept.name = std::string("<lora:") + current_concept.name;
    if (text.find(current_concept.name) == text.npos) {
        text += " ";
        text += current_concept.name + ":1.0>";
    }
    positive_input_->value(text.c_str());
}

void PromptFrame::from_profile() {
    guidance_input_->value();
    float cfg = prompt_profile_get_float({get_arch(), get_model(), "cfg"}, get_cfg());
    char buffer[25] = "";
    sprintf(buffer, "%.2f", cfg);
    guidance_input_->value(buffer);
    int steps = prompt_profile_get_int({get_arch(), get_model(), "steps"}, get_steps());
    sprintf(buffer, "%d", steps);
    steps_input_->value(buffer);

    int width = prompt_profile_get_int({get_arch(), get_model(), "width"}, get_width());
    sprintf(buffer, "%d", width);
    width_input_->value(buffer);

    int height = prompt_profile_get_int({get_arch(), get_model(), "height"}, get_height());
    sprintf(buffer, "%d", height);
    height_input_->value(buffer);

    negative_input_->value(prompt_profile_get_string({get_arch(), get_model(), "negative_prompt"}, negative_prompt()).c_str());
}

void PromptFrame::to_profile() {
    prompt_profile_set_string({get_arch(), "model_name"}, get_model());
    prompt_profile_set_float({get_arch(), get_model(), "cfg"}, get_cfg());
    prompt_profile_set_int({get_arch(), get_model(), "steps"}, get_steps());
    prompt_profile_set_int({get_arch(), get_model(), "width"}, get_width());
    prompt_profile_set_int({get_arch(), get_model(), "height"}, get_height());
    prompt_profile_set_string({get_arch(), get_model(), "negative_prompt"}, negative_prompt());
}

void PromptFrame::save_profile() {
    to_profile();
    prompt_save_profile();
}

} // namespace editorium
