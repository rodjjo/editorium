#pragma once

#include <string>

#include <FL/Fl_Group.H>
#include <FL/Fl_Text_Editor.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Int_Input.H>
#include <FL/Fl_Float_Input.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Text_Buffer.H>

#include "images/image.h"
#include "components/button.h"

namespace editorium
{

class LtxVideoWindow: public Fl_Window {
public:
    LtxVideoWindow(image_ptr_t first_frame);
    ~LtxVideoWindow();
    bool confirmed();
    std::string get_positive_prompt();
    std::string get_negative_prompt();
    std::string get_file_name();
    int get_seed();
    int get_width();
    int get_height();
    int get_num_frames();
    int get_frame_rate();
    int get_num_inference_steps();
    float get_guidance_scale();
    float get_strength();
    std::string get_stg_skip_layers();
    std::string get_stg_mode();
    float get_stg_scale();
    float get_stg_rescale();
    float get_image_cond_noise_scale();
    float get_decode_timestep();
    float get_decode_noise_scale();
    int get_num_generated_videos();
    image_ptr_t get_first_frame();
    image_ptr_t get_last_frame();

private:
    void align_component();
    void interrogate_image();

private:
    bool confirmed_ = false;
    Fl_Text_Editor *positive_prompt_ = nullptr;
    Fl_Text_Editor *negative_prompt_ = nullptr;
    Fl_Text_Buffer *sys_prompt_buffer_ = nullptr;
    Fl_Text_Buffer *usr_prompt_buffer_ = nullptr;
    
    Fl_Input *file_name_ = nullptr;
    Fl_Int_Input *seed_ = nullptr;
    Fl_Int_Input *width_ = nullptr;
    Fl_Int_Input *height_ = nullptr;
    Fl_Int_Input *num_frames_ = nullptr;
    Fl_Int_Input *frame_rate_ = nullptr;
    Fl_Int_Input *num_inference_steps_ = nullptr;
    Fl_Float_Input *guidance_scale_ = nullptr;
    Fl_Float_Input *strength_ = nullptr;
    Fl_Input *stg_skip_layers_ = nullptr;
    Fl_Choice *stg_mode_ = nullptr;
    Fl_Float_Input *stg_scale_ = nullptr;
    Fl_Float_Input *stg_rescale_ = nullptr;
    Fl_Float_Input *image_cond_noise_scale_ = nullptr;
    Fl_Float_Input *decode_timestep_ = nullptr;
    Fl_Float_Input *decode_noise_scale_ = nullptr;
    Fl_Int_Input *num_generated_videos_ = nullptr;
    image_ptr_t first_frame_;
    image_ptr_t last_frame_; 
    std::unique_ptr<Button> btn_interrogate_;
    std::unique_ptr<Button> btnOk_;
    std::unique_ptr<Button> btnCancel_;
};

void generate_video_ltx_model(image_ptr_t img);

} // namespace editorium
