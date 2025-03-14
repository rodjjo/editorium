#pragma once

#include <list>
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
#include "components/image_panel.h"

namespace editorium
{

class LtxVideoWindow: public Fl_Window {
public:
    LtxVideoWindow(std::list<image_ptr_t> frames);
    ~LtxVideoWindow();
    std::string get_positive_prompt();
    std::string get_negative_prompt();
    std::string get_file_name();
    int get_seed();
    int get_width();
    int get_height();
    int get_num_frames();
    int get_frame_rate();
    int get_num_inference_steps();
    int get_intermediate_start();
    int get_limit_frames_video();
    int get_skip_frames_video();
    float get_guidance_scale();
    float get_strength();
    float get_intermediate_strength();
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
    bool should_ignore_first_frame();
    bool should_ignore_last_frame();
    std::list<image_ptr_t> get_intermediate_frames();

private:
    static void clear_scroll(void *this_ptr);
    void clear_scroll();
    void suggest_size();

private:
    void align_component();
    void interrogate_image();
    void first_frame_open();
    void first_frame_palette();
    void first_frame_clipbrd();
    void first_frame_generate();
    void first_frame_to_palette();
    void first_frame_select_all();
    void second_frame_open();
    void second_frame_palette();
    void second_frame_clipbrd();
    void second_frame_generate();
    void second_frame_select_all();
    void second_frame_to_palette();
    void set_frame(image_ptr_t img, ImagePanel *panel);
    void generate_clicked();

private:
    std::list<image_ptr_t> intermediate_frames_;
    Fl_Text_Editor *positive_prompt_ = nullptr;
    Fl_Text_Editor *negative_prompt_ = nullptr;
    Fl_Text_Buffer *sys_prompt_buffer_ = nullptr;
    Fl_Text_Buffer *usr_prompt_buffer_ = nullptr;
    ImagePanel *first_img_ = nullptr;
    ImagePanel *last_img_ = nullptr;
    Fl_Input *file_name_ = nullptr;
    Fl_Int_Input *seed_ = nullptr;
    Fl_Int_Input *width_ = nullptr;
    Fl_Int_Input *height_ = nullptr;
    Fl_Int_Input *num_frames_ = nullptr;
    Fl_Int_Input *frame_rate_ = nullptr;
    Fl_Int_Input *num_inference_steps_ = nullptr;
    Fl_Float_Input *guidance_scale_ = nullptr;
    Fl_Float_Input *strength_ = nullptr;
    Fl_Float_Input *intermediate_strength_ = nullptr;
    Fl_Int_Input *intermediate_start_ = nullptr;
    Fl_Int_Input *skip_frames_video_ = nullptr;
    Fl_Int_Input *limit_frames_video_ = nullptr;
    Fl_Input *stg_skip_layers_ = nullptr;
    Fl_Choice *stg_mode_ = nullptr;
    Fl_Float_Input *stg_scale_ = nullptr;
    Fl_Float_Input *stg_rescale_ = nullptr;
    Fl_Float_Input *image_cond_noise_scale_ = nullptr;
    Fl_Float_Input *decode_timestep_ = nullptr;
    Fl_Float_Input *decode_noise_scale_ = nullptr;
    Fl_Int_Input *num_generated_videos_ = nullptr;
    Fl_Check_Button *btn_ignore_first_frame_ = nullptr;
    Fl_Check_Button *btn_ignore_last_frame_ = nullptr;
    image_ptr_t first_frame_;
    image_ptr_t last_frame_; 
    std::unique_ptr<Button> btn_first_frame_open_;
    std::unique_ptr<Button> btn_first_frame_palette_;
    std::unique_ptr<Button> btn_first_frame_clipbrd_;
    std::unique_ptr<Button> btn_first_frame_generate_;
    std::unique_ptr<Button> btn_first_frame_all_;
    std::unique_ptr<Button> btn_first_frame_to_pal_;
    std::unique_ptr<Button> btn_second_frame_open_;
    std::unique_ptr<Button> btn_second_frame_palette_;
    std::unique_ptr<Button> btn_second_frame_clipbrd_;
    std::unique_ptr<Button> btn_second_frame_generate_;
    std::unique_ptr<Button> btn_second_frame_all_;
    std::unique_ptr<Button> btn_second_frame_to_pal_;

    std::unique_ptr<Button> btn_interrogate_;
    std::unique_ptr<Button> btnOk_;
    std::unique_ptr<Button> btnCancel_;
};

void generate_video_ltx_model(image_ptr_t img);
void generate_video_ltx_model(const std::string& video_path);

} // namespace editorium
