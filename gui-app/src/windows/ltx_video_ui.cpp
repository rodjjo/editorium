#include "components/xpm/xpm.h"
#include "misc/dialogs.h"

#include "misc/profiles.h"
#include "windows/ltx_video_ui.h"
#include "windows/chatbot_ui.h"
#include "websocket/tasks.h"


namespace editorium {

namespace {
    std::string last_prompt_used;
    std::string last_negative_prompt_used;
}

LtxVideoWindow::LtxVideoWindow(image_ptr_t first_frame): Fl_Window(Fl::w() / 2 - 860 / 2, Fl::h() / 2 - 640 / 2, 860, 520, "Generate Video (LTX)") {
    first_frame_ = first_frame;
    positive_prompt_ = new Fl_Text_Editor(0, 0, 1, 1, "Positive prompt");
    sys_prompt_buffer_ = new Fl_Text_Buffer();
    positive_prompt_->buffer(sys_prompt_buffer_);
    positive_prompt_->wrap_mode(Fl_Text_Display::WRAP_AT_BOUNDS, 5);
    negative_prompt_ = new Fl_Text_Editor(0, 0, 1, 1, "Negative prompt");
    usr_prompt_buffer_ = new Fl_Text_Buffer();
    negative_prompt_->buffer(usr_prompt_buffer_);
    negative_prompt_->wrap_mode(Fl_Text_Display::WRAP_AT_BOUNDS, 5);

    file_name_ = new Fl_Input(0, 0, 0, 0, "Filename");
    seed_ = new Fl_Int_Input(0, 0, 0, 0, "Seed");
    num_inference_steps_ = new Fl_Int_Input(0, 0, 0, 0, "Steps");
    guidance_scale_ = new Fl_Float_Input(0, 0, 0, 0, "CFG");
    strength_ = new Fl_Float_Input(0, 0, 0, 0, "Strength");
    stg_skip_layers_ = new Fl_Input(0, 0, 0, 0, "stg skip layers");
    stg_mode_ = new Fl_Choice(0, 0, 0, 0, "stg mode");
    stg_scale_ = new Fl_Float_Input(0, 0, 0, 0, "Stg scale");
    stg_rescale_ = new Fl_Float_Input(0, 0, 0, 0, "Stg rescale");
    image_cond_noise_scale_ = new Fl_Float_Input(0, 0, 0, 0, "Image cond noise scale");
    decode_timestep_ = new Fl_Float_Input(0, 0, 0, 0, "Decode timestep");
    decode_noise_scale_ = new Fl_Float_Input(0, 0, 0, 0, "Decode noise scale");
    num_generated_videos_ = new Fl_Int_Input(0, 0, 0, 0, "Num generated videos");
    width_ = new Fl_Int_Input(0, 0, 0, 0, "Width");
    height_ = new Fl_Int_Input(0, 0, 0, 0, "Height");
    num_frames_ = new Fl_Int_Input(0, 0, 0, 0, "Num Frames");
    frame_rate_ = new Fl_Int_Input(0, 0, 0, 0, "Frame Rate");

    btn_interrogate_.reset(new Button(xpm::image(xpm::img_24x24_question), [this] {
        interrogate_image();
    }));

    btnOk_.reset(new Button(xpm::image(xpm::img_24x24_ok), [this] {
        std::string positive_prompt = positive_prompt_->buffer()->text();
        std::string negative_prompt = negative_prompt_ != nullptr ? negative_prompt_->buffer()->text() : "place_holder";
        if (!positive_prompt.empty() && !negative_prompt.empty()) {
            confirmed_ = true;
            last_prompt_used = positive_prompt;
            last_negative_prompt_used = negative_prompt;
        } else {
            show_error("Please fill in the prompt!");
            return;
        }
        this->hide();
    }));
    btnCancel_.reset(new Button(xpm::image(xpm::img_24x24_abort), [this] {
        this->hide();        
    }));

    
    stg_mode_->add("attention_values");
    stg_mode_->add("attention_skip");
    stg_mode_->add("residual");
    stg_mode_->add("transformer_block");
    stg_mode_->value(0);

    seed_->value("-1");
    num_inference_steps_->value("50");
    guidance_scale_->value("5.0");
    strength_->value("0.8");
    stg_skip_layers_->value("19");
    stg_scale_->value("1.0");
    stg_rescale_->value("0.7");
    image_cond_noise_scale_->value("0.15");
    decode_timestep_->value("0.025");
    decode_noise_scale_->value("0.0125");
    file_name_->value("saved-videos/output.mp4");
    num_generated_videos_->value("1");
    width_->value("704");
    height_->value("480");
    num_frames_->value("121");
    frame_rate_->value("25");
    positive_prompt_->align(FL_ALIGN_TOP_LEFT);
    negative_prompt_->align(FL_ALIGN_TOP_LEFT);
    file_name_->align(FL_ALIGN_TOP_LEFT);
    seed_->align(FL_ALIGN_TOP_LEFT);
    num_inference_steps_->align(FL_ALIGN_TOP_LEFT);
    guidance_scale_->align(FL_ALIGN_TOP_LEFT);
    strength_->align(FL_ALIGN_TOP_LEFT);
    stg_skip_layers_->align(FL_ALIGN_TOP_LEFT);
    stg_mode_->align(FL_ALIGN_TOP_LEFT);
    stg_scale_->align(FL_ALIGN_TOP_LEFT);
    stg_rescale_->align(FL_ALIGN_TOP_LEFT);
    image_cond_noise_scale_->align(FL_ALIGN_TOP_LEFT);
    decode_timestep_->align(FL_ALIGN_TOP_LEFT);
    decode_noise_scale_->align(FL_ALIGN_TOP_LEFT);
    num_generated_videos_->align(FL_ALIGN_TOP_LEFT);
    width_->align(FL_ALIGN_TOP_LEFT);
    height_->align(FL_ALIGN_TOP_LEFT);
    num_frames_->align(FL_ALIGN_TOP_LEFT);
    frame_rate_->align(FL_ALIGN_TOP_LEFT);

    if (first_frame_) {
        float width_by_704 = 704.0 / first_frame_->w();
        float height_by_480 = 480.0 / first_frame_->h();
        float scale = std::min(width_by_704, height_by_480);
        int new_width = first_frame_->w() * scale;
        int new_height = first_frame_->h() * scale;
        std::string width_str = std::to_string(new_width);
        std::string height_str = std::to_string(new_height);
        width_->value(width_str.c_str());
        height_->value(height_str.c_str());
    }

    btn_interrogate_->tooltip("Use a multimodal model to look at the current image and describe it.");
   
    chatbot_load_profile();
    if (!last_prompt_used.empty()) {
        positive_prompt_->insert(last_prompt_used.c_str());
    }
    if (!last_negative_prompt_used.empty()) {
        negative_prompt_->insert(last_negative_prompt_used.c_str());
    } else {
        negative_prompt_->insert(
            "Bright tones, overexposed, blurred details, subtitles, overall gray, worst quality,"
            "low quality, JPEG compression residue, ugly," 
            "deformed, disfigured, misshapen limbs, fused fingers"
        );
    }
    set_modal();
    align_component();
}

LtxVideoWindow::~LtxVideoWindow() {
    if(sys_prompt_buffer_) {
        delete sys_prompt_buffer_;
    }
    if (usr_prompt_buffer_) {
        delete usr_prompt_buffer_;
    }
}

bool LtxVideoWindow::confirmed() {
    return confirmed_;    
}

void LtxVideoWindow::align_component() {
    positive_prompt_->resize(10, 40, this->w() - 58, 100);
    btn_interrogate_->position(positive_prompt_->x() + positive_prompt_->w() + 5, positive_prompt_->y());
    btn_interrogate_->size(28, 28);

    negative_prompt_->resize(10, positive_prompt_->y() + positive_prompt_->h() + 20, this->w() - 20, 100);
    file_name_->resize(10, negative_prompt_->y() + negative_prompt_->h() + 20, 200, 30);
    seed_->resize(file_name_->x() + file_name_->w() + 10, file_name_->y(), 80, 30);
    num_inference_steps_->resize(seed_->x() + seed_->w() + 10, seed_->y(), 80, 30);
    guidance_scale_->resize(num_inference_steps_->x() + num_inference_steps_->w() + 10, num_inference_steps_->y(), 80, 30);
    strength_->resize(guidance_scale_->x() + guidance_scale_->w() + 10, guidance_scale_->y(), 80, 30);
    stg_skip_layers_->resize(10, file_name_->y() + file_name_->h() + 20, 80, 30);
    stg_mode_->resize(stg_skip_layers_->x() + stg_skip_layers_->w() + 10, stg_skip_layers_->y(), 150, 30);
    stg_scale_->resize(stg_mode_->x() + stg_mode_->w() + 10, stg_mode_->y(), 80, 30);
    stg_rescale_->resize(stg_scale_->x() + stg_scale_->w() + 10, stg_scale_->y(), 80, 30);
    image_cond_noise_scale_->resize(10, stg_skip_layers_->y() + stg_skip_layers_->h() + 20, 150, 30);
    decode_timestep_->resize(image_cond_noise_scale_->x() + image_cond_noise_scale_->w() + 10, image_cond_noise_scale_->y(), 150, 30);
    decode_noise_scale_->resize(decode_timestep_->x() + decode_timestep_->w() + 10, decode_timestep_->y(), 150, 30);
    num_generated_videos_->resize(decode_noise_scale_->x() + decode_noise_scale_->w() + 10, decode_noise_scale_->y(), 80, 30);
    width_->resize(10, image_cond_noise_scale_->y() + image_cond_noise_scale_->h() + 20, 80, 30);
    height_->resize(width_->x() + width_->w() + 10, width_->y(), 80, 30);
    num_frames_->resize(height_->x() + height_->w() + 10, height_->y(), 80, 30);
    frame_rate_->resize(num_frames_->x() + num_frames_->w() + 10, num_frames_->y(), 80, 30);

    btnOk_->position(this->w() - 215, this->h() - 40);
    btnOk_->size(100, 30);
    btnCancel_->position(btnOk_->x() + btnOk_->w() + 2, btnOk_->y());
    btnCancel_->size(100, 30);
}


std::string LtxVideoWindow::get_positive_prompt() {
    return positive_prompt_->buffer()->text();
}

std::string LtxVideoWindow::get_negative_prompt() {
    return negative_prompt_->buffer()->text();
}

std::string LtxVideoWindow::get_file_name() {
    return file_name_->value();
}

int LtxVideoWindow::get_seed() {
    return std::stoi(seed_->value());
}

int LtxVideoWindow::get_num_inference_steps() {
    return std::stoi(num_inference_steps_->value());
}

float LtxVideoWindow::get_guidance_scale() {
    return std::stof(guidance_scale_->value());
}

float LtxVideoWindow::get_strength() {
    return std::stof(strength_->value());
}

std::string LtxVideoWindow::get_stg_skip_layers() {
    return stg_skip_layers_->value();
}

std::string LtxVideoWindow::get_stg_mode() {
    return stg_mode_->text();
}

float LtxVideoWindow::get_stg_scale() {
    return std::stof(stg_scale_->value());
}

float LtxVideoWindow::get_stg_rescale() {
    return std::stof(stg_rescale_->value());
}

float LtxVideoWindow::get_image_cond_noise_scale() {
    return std::stof(image_cond_noise_scale_->value());
}

float LtxVideoWindow::get_decode_timestep() {
    return std::stof(decode_timestep_->value());
}

float LtxVideoWindow::get_decode_noise_scale() {
    return std::stof(decode_noise_scale_->value());
}
int LtxVideoWindow::get_width() {
    return std::stoi(width_->value());
}

int LtxVideoWindow::get_height() {
    return std::stoi(height_->value());
}

int LtxVideoWindow::get_num_frames() {
    return std::stoi(num_frames_->value());
}

int LtxVideoWindow::get_frame_rate() {
    return std::stoi(frame_rate_->value());
}

image_ptr_t LtxVideoWindow::get_first_frame() {
    return first_frame_;
}

image_ptr_t LtxVideoWindow::get_last_frame() {
    return last_frame_;
}

int LtxVideoWindow::get_num_generated_videos() {
    return std::stoi(num_generated_videos_->value());
}

void LtxVideoWindow::interrogate_image() {
    auto img = first_frame_ ? first_frame_ : last_frame_;
    if (!img) {
        show_error("No image to interrogate!");
        return;
    }
    auto prompts = get_prompts_for_vision("Configuration - for image interrogation - arch ltx-video ", "ltx-video::prompt-from-image");
    if (!prompts.first.empty() && !prompts.second.empty()) {
        ws::chatbots::vision_chat_request_t req;
        req.system_prompt = prompts.first;
        req.prompt = prompts.second;
        req.image = img;
        auto result = ws::chatbots::chat_bot_vision(req);
        if (!result.empty()) {
            positive_prompt_->buffer()->text(result.c_str());
        }
    }
}

void generate_video_ltx_model(image_ptr_t first_frame) {
    auto window = new LtxVideoWindow(first_frame);
    window->show();
    while (true) {
        if (!window->visible_r()) {
            break;
        }
        Fl::wait();
    }
    if (window->confirmed()) {
        ws::video_gen::ltx_video_gen_request_t request;
        request.prompt = window->get_positive_prompt();
        request.negative_prompt = window->get_negative_prompt();
        request.lora_path = "lora";
        request.lora_rank = 128;
        request.num_inference_steps = window->get_num_inference_steps();
        request.guidance_scale = window->get_guidance_scale();
        request.num_videos_per_prompt = 1;
        request.seed = window->get_seed();
        request.strength = window->get_strength();
        request.stg_skip_layers = window->get_stg_skip_layers();
        request.stg_mode = window->get_stg_mode();
        request.stg_scale = window->get_stg_scale();
        request.stg_rescale = window->get_stg_rescale();
        request.image_cond_noise_scale = window->get_image_cond_noise_scale();
        request.decode_timestep = window->get_decode_timestep();
        request.decode_noise_scale = window->get_decode_noise_scale();
        request.save_path = window->get_file_name();
        request.first_frame = window->get_first_frame();
        request.last_frame = window->get_last_frame();
        request.width = window->get_width();
        request.height = window->get_height();
        request.num_frames = window->get_num_frames();
        request.frame_rate = window->get_frame_rate();
        int current_video = 0;
        while (current_video < window->get_num_generated_videos()) {
            if (!ws::video_gen::run_ltx_video_gen(request)) {
                break;
            }
            request.seed = -1;
            current_video += 1;
        }
    }
    Fl::delete_widget(window);
    Fl::do_widget_deletion();
}

} // namespace editorium
