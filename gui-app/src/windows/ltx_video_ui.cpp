#include "components/xpm/xpm.h"
#include "misc/dialogs.h"
#include "misc/profiles.h"
#include "windows/ltx_video_ui.h"
#include "windows/chatbot_ui.h"
#include "windows/image_palette_ui.h"
#include "images/image_palette.h"
#include "websocket/tasks.h"
#include "diffusion_ui.h"


namespace editorium {

namespace {
    std::string last_prompt_used;
    std::string last_negative_prompt_used;
}

LtxVideoWindow::LtxVideoWindow(std::list<image_ptr_t> frames): Fl_Window(Fl::w() / 2 - 860 / 2, Fl::h() / 2 - 800 / 2, 860, 800, "Generate Video (LTX)") {
    first_frame_ = *(frames.begin());
    if (frames.size() > 1) {
        last_frame_ = *(frames.rbegin());
    }
    if (frames.size() > 2) {
        intermediate_frames_ = std::list<image_ptr_t>(std::next(frames.begin(), 1), std::prev(frames.end()));
    }

    first_img_ = new AllowSelectionImagePanel(0, 0, 1, 1, "First Frame");
    last_img_ = new AllowSelectionImagePanel(0, 0, 1, 1, "Last Frame");

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
    strength_ = new Fl_Float_Input(0, 0, 0, 0, "Strength first/last frame");
    intermediate_strength_ = new Fl_Float_Input(0, 0, 0, 0, "Strength int. frames");
    intermediate_start_ = new Fl_Int_Input(0, 0, 0, 0, "Intermediate start");
    skip_frames_video_ = new Fl_Int_Input(0, 0, 0, 0, "V2V Skip frames");
    limit_frames_video_ = new Fl_Int_Input(0, 0, 0, 0, "V2V Limit frames");
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
    btn_ignore_first_frame_ = new Fl_Check_Button(0, 0, 0, 0, "Ignore First Frame");
    btn_ignore_last_frame_ = new Fl_Check_Button(0, 0, 0, 0, "Ignore Last Frame");


    btn_first_frame_open_.reset(new Button(xpm::image(xpm::img_24x24_open), [this] {
        first_frame_open();
    }));
    btn_first_frame_palette_.reset(new Button(xpm::image(xpm::img_24x24_pinion), [this] {
        first_frame_palette();
    }));
    btn_first_frame_clipbrd_.reset(new Button(xpm::image(xpm::img_24x24_copy), [this] {
        first_frame_clipbrd();
    }));
    btn_first_frame_all_.reset(new Button(xpm::image(xpm::img_24x24_female), [this] {
        first_frame_select_all();
    }));
    btn_first_frame_to_pal_.reset(new Button(xpm::image(xpm::img_24x24_forward), [this] {
        first_frame_to_palette();
    }));
    btn_first_frame_generate_.reset(new Button(xpm::image(xpm::img_24x24_bee), [this] {
        first_frame_generate();
    }));
    btn_second_frame_open_.reset(new Button(xpm::image(xpm::img_24x24_open), [this] {
        second_frame_open();
    }));
    btn_second_frame_palette_.reset(new Button(xpm::image(xpm::img_24x24_pinion), [this] {
        second_frame_palette();
    }));
    btn_second_frame_clipbrd_.reset(new Button(xpm::image(xpm::img_24x24_copy), [this] {
        second_frame_clipbrd();
    }));
    btn_second_frame_all_.reset(new Button(xpm::image(xpm::img_24x24_female), [this] {
        second_frame_select_all();
    }));
    btn_second_frame_to_pal_.reset(new Button(xpm::image(xpm::img_24x24_forward), [this] {
        second_frame_to_palette();
    }));
    btn_second_frame_generate_.reset(new Button(xpm::image(xpm::img_24x24_bee), [this] {
        second_frame_generate();
    }));
    

    btn_interrogate_.reset(new Button(xpm::image(xpm::img_24x24_question), [this] {
        interrogate_image();
    }));

    btnOk_.reset(new Button(xpm::image(xpm::img_24x24_ok), [this] {
        std::string positive_prompt = positive_prompt_->buffer()->text();
        std::string negative_prompt = negative_prompt_ != nullptr ? negative_prompt_->buffer()->text() : "place_holder";
        if (!positive_prompt.empty() && !negative_prompt.empty()) {
            last_prompt_used = positive_prompt;
            last_negative_prompt_used = negative_prompt;
            first_img_->hide();
            last_img_->hide();
            generate_clicked();
            first_img_->show();
            last_img_->show();
        } else {
            show_error("Please fill in the prompt!");
        }
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
    strength_->value("0.95");
    intermediate_strength_->value("0.10");
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
    intermediate_start_->value("8");
    skip_frames_video_->value("0");
    limit_frames_video_->value("-1");
    positive_prompt_->align(FL_ALIGN_TOP_LEFT);
    negative_prompt_->align(FL_ALIGN_TOP_LEFT);
    file_name_->align(FL_ALIGN_TOP_LEFT);
    seed_->align(FL_ALIGN_TOP_LEFT);
    num_inference_steps_->align(FL_ALIGN_TOP_LEFT);
    guidance_scale_->align(FL_ALIGN_TOP_LEFT);
    strength_->align(FL_ALIGN_TOP_LEFT);
    intermediate_strength_->align(FL_ALIGN_TOP_LEFT);
    intermediate_start_->align(FL_ALIGN_TOP_LEFT);
    skip_frames_video_->align(FL_ALIGN_TOP_LEFT);
    limit_frames_video_->align(FL_ALIGN_TOP_LEFT);
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
    btn_ignore_first_frame_->align(FL_ALIGN_TOP_LEFT);
    btn_ignore_last_frame_->align(FL_ALIGN_TOP_LEFT);

    suggest_size();

    btn_interrogate_->tooltip("Use a multimodal model to look at the current image and describe it.");
    first_img_->tooltip("First frame");
    last_img_->tooltip("Last frame");
    btn_first_frame_open_->tooltip("Open first frame");
    btn_first_frame_palette_->tooltip("Open first frame in palette");
    btn_first_frame_clipbrd_->tooltip("Past first frame from the clipboard");
    btn_first_frame_all_->tooltip("Select the entire image for the first frame");
    btn_first_frame_to_pal_->tooltip("Add the first frame to the palette");
    btn_first_frame_generate_->tooltip("Generate an image for the first frame");
    btn_second_frame_open_->tooltip("Open last frame");
    btn_second_frame_palette_->tooltip("Open last frame in palette");
    btn_second_frame_clipbrd_->tooltip("Past last frame from the clipboard");
    btn_second_frame_all_->tooltip("Select the entire image for the last frame");
    btn_second_frame_to_pal_->tooltip("Add the last frame to the palette");
    btn_second_frame_generate_->tooltip("Generate an image for the last frame");
    intermediate_start_->tooltip("For video extending the starting frame (multiple of 8)");
    skip_frames_video_->tooltip("For video to video, skip x first frames");
    limit_frames_video_->tooltip("For video to video, limit the number of frames (-1 for all)");

    if (first_frame_) {
        first_img_->view_settings()->add_layer(first_frame_);
    }

    if (last_frame_) {
        last_img_->view_settings()->add_layer(last_frame_);
    }
   
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
    Fl::add_timeout(1.0, LtxVideoWindow::clear_scroll, this);
}

LtxVideoWindow::~LtxVideoWindow() {
    Fl::remove_timeout(LtxVideoWindow::clear_scroll, this);
    if(sys_prompt_buffer_) {
        delete sys_prompt_buffer_;
    }
    if (usr_prompt_buffer_) {
        delete usr_prompt_buffer_;
    }
}

void LtxVideoWindow::clear_scroll(void *this_ptr) {
    static_cast<LtxVideoWindow *>(this_ptr)->clear_scroll();
}

void LtxVideoWindow::clear_scroll() {
    if (first_img_->view_settings()->layer_count() > 0) {
        first_img_->view_settings()->setZoom(100);
        first_img_->clear_scroll();
    }
    if (last_img_->view_settings()->layer_count() > 0) {
        last_img_->view_settings()->setZoom(100);
        last_img_->clear_scroll();
    }
}

void LtxVideoWindow::suggest_size() {
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
}

void LtxVideoWindow::align_component() {
    first_img_->resize(10, 10, this->w() / 2 - 100, 200);
    btn_first_frame_open_->position(first_img_->x() + first_img_->w() + 5, first_img_->y());
    btn_first_frame_open_->size(28, 28);
    btn_first_frame_palette_->position(first_img_->x() + first_img_->w() + 5, first_img_->y() + 30);
    btn_first_frame_palette_->size(28, 28);
    btn_first_frame_clipbrd_->position(first_img_->x() + first_img_->w() + 5, first_img_->y() + 60);
    btn_first_frame_clipbrd_->size(28, 28);
    btn_first_frame_all_->position(first_img_->x() + first_img_->w() + 5, first_img_->y() + 90);
    btn_first_frame_all_->size(28, 28);
    btn_first_frame_to_pal_->position(first_img_->x() + first_img_->w() + 5, first_img_->y() + 120);
    btn_first_frame_to_pal_->size(28, 28);
    btn_first_frame_generate_->position(first_img_->x() + first_img_->w() + 5, first_img_->y() + 150);
    btn_first_frame_generate_->size(28, 28);

    last_img_->resize(first_img_->x() + first_img_->w() + 10 + 48, first_img_->y(), first_img_->w(), 200);
    btn_second_frame_open_->position(last_img_->x() + last_img_->w() + 5, last_img_->y());
    btn_second_frame_open_->size(28, 28);
    btn_second_frame_palette_->position(last_img_->x() + last_img_->w() + 5, last_img_->y() + 30);
    btn_second_frame_palette_->size(28, 28);
    btn_second_frame_clipbrd_->position(last_img_->x() + last_img_->w() + 5, last_img_->y() + 60);
    btn_second_frame_clipbrd_->size(28, 28);
    btn_second_frame_all_->position(last_img_->x() + last_img_->w() + 5, last_img_->y() + 90);
    btn_second_frame_all_->size(28, 28);
    btn_second_frame_to_pal_->position(last_img_->x() + last_img_->w() + 5, last_img_->y() + 120);
    btn_second_frame_to_pal_->size(28, 28);
    btn_second_frame_generate_->position(last_img_->x() + last_img_->w() + 5, last_img_->y() + 150);
    btn_second_frame_generate_->size(28, 28);

    positive_prompt_->resize(10, this->w() - 20, this->w() - 58, 100);
    positive_prompt_->position(10, first_img_->y() + first_img_->h() + 40);
    btn_interrogate_->position(positive_prompt_->x() + positive_prompt_->w() + 5, positive_prompt_->y());
    btn_interrogate_->size(28, 28);

    negative_prompt_->resize(10, positive_prompt_->y() + positive_prompt_->h() + 20, this->w() - 20, 100);
    file_name_->resize(10, negative_prompt_->y() + negative_prompt_->h() + 20, 200, 30);
    seed_->resize(file_name_->x() + file_name_->w() + 10, file_name_->y(), 80, 30);
    num_inference_steps_->resize(seed_->x() + seed_->w() + 10, seed_->y(), 80, 30);
    guidance_scale_->resize(num_inference_steps_->x() + num_inference_steps_->w() + 10, num_inference_steps_->y(), 80, 30);
    strength_->resize(guidance_scale_->x() + guidance_scale_->w() + 10, guidance_scale_->y(), 100, 30);
    intermediate_strength_->resize(10, file_name_->y() + file_name_->h() + 20, 160, 30);
    intermediate_start_->resize(intermediate_strength_->x() + intermediate_strength_->w() + 10, intermediate_strength_->y(), 160, 30);
    skip_frames_video_->resize(intermediate_start_->x() + intermediate_start_->w() + 10, intermediate_start_->y(), 160, 30);
    limit_frames_video_->resize(skip_frames_video_->x() + skip_frames_video_->w() + 10, skip_frames_video_->y(), 160, 30);
    stg_skip_layers_->resize(10, intermediate_strength_->y() + intermediate_strength_->h() + 20, 150, 30);
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
    btn_ignore_first_frame_->resize(10, width_->y() + width_->h() + 20, 150, 30);
    btn_ignore_last_frame_->resize(btn_ignore_first_frame_->x() + btn_ignore_first_frame_->w() + 10, btn_ignore_first_frame_->y(), 150, 30);

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

float LtxVideoWindow::get_intermediate_strength() {
    return std::stof(intermediate_strength_->value());
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

int LtxVideoWindow::get_intermediate_start() {
    return std::stoi(intermediate_start_->value());
}

int LtxVideoWindow::get_limit_frames_video() {
    return std::stoi(limit_frames_video_->value());
}

int LtxVideoWindow::get_skip_frames_video() {
    return std::stoi(skip_frames_video_->value());
}

void LtxVideoWindow::first_frame_to_palette() {
    if (first_frame_) {
        add_image_palette(first_frame_);
    }
}

void LtxVideoWindow::first_frame_select_all() {
    if (first_frame_) {
        int x = 0, y = 0;
        int w = 0, h = 0;
        first_img_->view_settings()->get_image_area(&x, &y, &w, &h);
        first_img_->view_settings()->set_selected_area(0, 0, w, h);
    }
}

void LtxVideoWindow::second_frame_select_all() {
    if (last_frame_) {
        int x = 0, y = 0;
        int w = 0, h = 0;
        last_img_->view_settings()->get_image_area(&x, &y, &w, &h);
        last_img_->view_settings()->set_selected_area(0, 0, w, h);
    }
}

void LtxVideoWindow::second_frame_to_palette() {
    if(last_frame_) {
        add_image_palette(last_frame_);
    }
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

void LtxVideoWindow::set_frame(image_ptr_t img, ImagePanel *panel) {
    if (!img) {
        panel->view_settings()->clear_layers();
        return;
    }
    panel->view_settings()->add_layer(img);
    panel->view_settings()->merge_layers_to_image(false);
    panel->view_settings()->setZoom(100);
    panel->clear_scroll();
}

void LtxVideoWindow::first_frame_open() {
    auto path = choose_image_to_open_fl("LtxVideoWindow::FirstFrameOpen");
    if (path.empty()) {
        return;
    }
    auto img = ws::filesystem::load_image(path);
    if (img) {
        first_frame_ = img;
        set_frame(img, first_img_);
        suggest_size();
    }
}

void LtxVideoWindow::first_frame_palette() {
    auto img = pickup_image_from_palette();
    if (img) {
        first_frame_ = img;
        set_frame(img, first_img_);
        suggest_size();
    }
}

void LtxVideoWindow::first_frame_clipbrd() {
    auto img = ws::diffusion::run_paste_image();
    if (img) {
        first_frame_ = img;
        set_frame(img, first_img_);
        suggest_size();
    } else {
        show_error("No image in the clipboard");
    }
}

void LtxVideoWindow::second_frame_open() {
    auto path = choose_image_to_open_fl("LtxVideoWindow::FirstFrameOpen");
    if (path.empty()) {
        return;
    }
    auto img = ws::filesystem::load_image(path);
    if (img) {
        last_frame_ = img;
        set_frame(img, last_img_);
    }
}

void LtxVideoWindow::second_frame_palette() {
    auto img = pickup_image_from_palette();
    if (img) {
        last_frame_ = img;
        set_frame(img, last_img_);
    }
}

void LtxVideoWindow::second_frame_clipbrd() {
    auto img = ws::diffusion::run_paste_image();
    if (img) {
        last_frame_ = img;
        set_frame(img, last_img_);
    } else {
        show_error("No image in the clipboard");
    }
}

void LtxVideoWindow::first_frame_generate() {
    first_img_->view_settings()->shrink_selected_area();
    auto img = first_img_->view_settings()->layer_count() > 0 ? generate_image(true, first_img_->view_settings()) : generate_image(true);
    if (img) {
        first_frame_ = img;
        set_frame(img, first_img_);
        suggest_size();
    }
}

void LtxVideoWindow::second_frame_generate() {
    first_img_->view_settings()->shrink_selected_area();
    auto img = last_img_->view_settings()->layer_count() > 0 ? generate_image(true, last_img_->view_settings()) : generate_image(true);
    if (img) {
        last_frame_ = img;
        set_frame(img, last_img_);
    }
}

bool LtxVideoWindow::should_ignore_first_frame() {
    return btn_ignore_first_frame_->value() != 0;
}

bool LtxVideoWindow::should_ignore_last_frame() {
    return btn_ignore_last_frame_->value() != 0;
}

std::list<image_ptr_t> LtxVideoWindow::get_intermediate_frames() {
    std::list<image_ptr_t> result;
    size_t used_frames = 0;
    if (first_frame_ && !should_ignore_first_frame()) {
        used_frames += 1;
    }
    if (last_frame_ && !should_ignore_last_frame()) {
        used_frames += 1;
    }
    size_t middle_frame = 0;
    size_t skipped_frames = get_skip_frames_video();
    std::list<image_ptr_t>::iterator first = intermediate_frames_.begin();
    while (used_frames < get_num_frames() && first != intermediate_frames_.end()) {
        if (first_frame_) {
            if (skipped_frames > 0) {
                skipped_frames--;
                continue;
            }
        }
        result.push_back(*first);
        first++;
        used_frames += 1;
        if (used_frames >= get_limit_frames_video() && get_limit_frames_video() != -1) {
            break;
        }
    }
    return result;
}

void LtxVideoWindow::generate_clicked() {
    if (get_intermediate_start() % 8 != 0) {
        show_error("Intermediate start must be a multiple of 8");
        return;
    }
    ws::video_gen::ltx_video_gen_request_t request;
    request.prompt = get_positive_prompt();
    request.negative_prompt = get_negative_prompt();
    request.lora_path = "lora";
    request.lora_rank = 128;
    request.num_inference_steps = get_num_inference_steps();
    request.guidance_scale = get_guidance_scale();
    request.num_videos_per_prompt = 1;
    request.seed = get_seed();
    request.strength = get_strength();
    request.stg_skip_layers = get_stg_skip_layers();
    request.stg_mode = get_stg_mode();
    request.stg_scale = get_stg_scale();
    request.stg_rescale = get_stg_rescale();
    request.image_cond_noise_scale = get_image_cond_noise_scale();
    request.decode_timestep = get_decode_timestep();
    request.decode_noise_scale = get_decode_noise_scale();
    request.intermediate_start = get_intermediate_start();
    request.intermediate_strength = get_intermediate_strength();
    request.save_path = get_file_name();
    if (!should_ignore_first_frame()) {
        request.first_frame = get_first_frame();
    }
    if (!should_ignore_last_frame()) {
        request.last_frame = get_last_frame();
    }
    request.width = get_width();
    request.height = get_height();
    request.num_frames = get_num_frames();
    request.frame_rate = get_frame_rate();
    request.intermediate_frames = get_intermediate_frames();
    int current_video = 0;
    while (current_video < get_num_generated_videos()) {
        if (!ws::video_gen::run_ltx_video_gen(request)) {
            break;
        }
        request.seed = -1;
        current_video += 1;
    }
}

void generate_video_ltx_model_from_list(std::list<image_ptr_t> frames) {
    auto window = new LtxVideoWindow(frames);
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


void generate_video_ltx_model(image_ptr_t first_frame) {
    std::list<image_ptr_t> frames;
    if (first_frame) {
        frames.push_back(first_frame);
    }
    generate_video_ltx_model_from_list(frames);
}


void generate_video_ltx_model(const std::string& video_path) {
    auto frame_list = ws::filesystem::grab_frames(video_path, 161 + 8);
    if (frame_list.empty()) {
        show_error("No frames found in the video");
        return;
    }
    generate_video_ltx_model_from_list(frame_list);
}

} // namespace editorium
