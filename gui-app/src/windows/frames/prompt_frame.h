#pragma once

#include <memory>
#include <vector>
#include <string>

#include <FL/Fl_Group.H>
#include <FL/Fl_Multiline_Input.H>
#include <FL/Fl_Int_Input.H>
#include <FL/Fl_Float_Input.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Check_Button.H>

#include "components/button.h"
#include "windows/frames/embeddings_frame.h"

namespace editorium
{

typedef enum {
    resize_not_resize,
    resize_fit_512x512,
    resize_fit_768x768,
    resize_fit_1024x1024,
    resize_fit_1280x1280,
    //  keep resize_mode_count at the end
    resize_mode_count
} resize_modes_t;

class PromptFrame: public SubscriberThis {
public:
    PromptFrame(Fl_Group *parent);
    ~PromptFrame();

    void alignComponents();
    std::string positive_prompt();
    void positive_prompt(const std::string& value, bool keep_loras);
    std::string negative_prompt();
    std::string get_model();
    std::string get_scheduler();
    std::string get_arch();
    std::vector<std::string> get_loras();
    int get_seed();
    int get_batch_size();
    int get_steps();
    float get_cfg();
    int get_width();
    int get_height();
    bool use_lcm_lora();
    bool use_tiny_vae();
    bool get_correct_colors();
    void save_profile();
    bool get_ensure_min_512();
    int get_scale_down_size();

    bool validate();
    void refresh_models();
private:
    void insert_current_textual();
    void insert_current_lora();
    void from_profile();
    void to_profile();
    
protected:
    static void widget_cb(Fl_Widget* widget, void *cbdata);
    void widget_cb(Fl_Widget* widget);
    void dfe_handle_event(void *sender, event_id_t event, void *data) override;

private:
    std::vector<std::pair<std::string, std::string> > architectures_;
    std::unique_ptr<EmbeddingFrame> loras_;
    std::unique_ptr<EmbeddingFrame> embeddings_;
    Fl_Group             *parent_;
    Fl_Group             *lora_gp_;
    Fl_Group             *emb_gp_;
    Fl_Multiline_Input   *positive_input_;
    Fl_Multiline_Input   *negative_input_;
    std::unique_ptr<Button> btn_improve_;
    std::unique_ptr<Button> btn_improve2_;
    std::unique_ptr<Button> btn_interrogate_;
    Fl_Int_Input         *seed_input_;
    Fl_Int_Input         *batch_input_;
    Fl_Int_Input         *steps_input_;
    Fl_Float_Input       *guidance_input_;
    Fl_Int_Input         *width_input_;
    Fl_Int_Input         *height_input_;
    Fl_Choice            *arch_input_;
    Fl_Choice            *models_input_;
    Fl_Choice            *schedulers_;
    Fl_Choice            *resizeModes_;
    Fl_Check_Button      *use_lcm_lora_;
    Fl_Check_Button      *use_tiny_vae_;
    Fl_Check_Button      *correct_colors_;
    Fl_Check_Button      *ensure_min_512_;
};

} // namespace editorium
