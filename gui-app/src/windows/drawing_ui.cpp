#include <cstdlib>
#include "misc/dialogs.h"

#include "websocket/tasks.h"
#include "components/xpm/xpm.h"
#include "misc/config.h"
#include "images/image_palette.h"
#include "windows/image_palette_ui.h"
#include "drawing_ui.h"

namespace editorium {

namespace {
    const uint8_t brushes_sizes[] = {
        4, 8, 16, 32, 64, 128
    };

    uint8_t brush_size_count() {
        return sizeof(brushes_sizes) / sizeof(brushes_sizes[0]);
    }
    const std::list<event_id_t> drawing_ui_events = {
            event_layer_mask_color_picked,
            event_prompt_lora_selected,
    };

    std::string last_prompt;


    image_ptr_t pixelate_image(image_ptr_t reference_img) {
        float ratio = 1.0;
        if (reference_img->w() > reference_img->h()) {
            ratio = 512.0 / reference_img->w();
        } else {
            ratio = 512.0 / reference_img->h();
        }
        int min_size_w, min_size_h; // 32 pixels minimum, checking the ratio
        int new_w, new_h;
        if (reference_img->w() > reference_img->h()) {
            new_w = 512;
            new_h = reference_img->h() * ratio;
            min_size_w = 32;
            min_size_h = 32 * ratio;
        } else {
            new_h = 512;
            new_w = reference_img->w() * ratio;
            min_size_h = 32;
            min_size_w = 32 * ratio;
        }

        return reference_img->blur(4)->resizeImage(min_size_w, min_size_h)->resizeImage(new_w, new_h);
    }
}

DrawingWindow::DrawingWindow(image_ptr_t reference_img) : Fl_Window(Fl::w() / 2 - 1024 / 2, Fl::h() / 2 - 680 / 2, 1024, 680, "Image palette - Select an image"),
        SubscriberThis(drawing_ui_events) {
    this->set_modal();

    image_ = pixelate_image(reference_img);
    
    this->begin();
    image_panel_ = new LayerDrawingImagePanel(0, 0, 1, 1, "Image");
    image_panel_->view_settings()->add_layer(newImage(image_->w(), image_->h(), true));
    image_panel_->enable_color_mask_editor(true);
    image_panel_->view_settings()->set_mask();
    image_panel_->view_settings()->at(1)->replace_image(image_->duplicate());
    image_panel_->view_settings()->at(0)->pinned(true);
    image_panel_->view_settings()->at(0)->focusable(false);
    image_panel_->view_settings()->at(2)->pinned(true);
    image_panel_->view_settings()->at(2)->focusable(false);
    
    image_panel_->view_settings()->at(1)->w(256);
    image_panel_->view_settings()->at(1)->h(256);
    image_panel_->view_settings()->at(1)->y(256);
    image_panel_->view_settings()->at(1)->x(0);

    right_panel_ = new Fl_Group(0, 0, 1, 1);
    right_panel_->begin();
    color_pal_group_ = new Fl_Group(0, 0, 1, 1);
    right_panel_->begin();
    settings_panel_ = new Fl_Group(0, 0, 1, 1);
    prompt_input_ = new Fl_Multiline_Input(0, 0, 1, 1, "Prompt");
    seed_input_ = new Fl_Int_Input(0, 0, 1, 1, "Seed");
    btnRandomSeed_.reset(new Button(xpm::image(xpm::img_24x24_magic_wand), [this] {
        random_seed();
    }));
    arch_input_ = new Fl_Choice(0, 0, 1, 1, "Architecture");
    model_input_ = new Fl_Choice(0, 0, 1, 1, "Model");

    settings_panel_->begin();
    lora_gp_ = new Fl_Group(0, 0, 1, 1);
    lora_gp_->begin();
    loras_.reset(new EmbeddingFrame(true, lora_gp_));
    lora_gp_->end();

    color_pal_group_->box(FL_DOWN_BOX);
    {
        color_pal_group_->begin();
        color_palette_.reset(new ColorPaletteFrame(color_pal_group_, image_panel_, "drawing-session"));
        color_pal_group_->end();
    }

    right_panel_->begin();
    btnFirstPass_.reset(new Button("1st pass", [this] {
        generate_image(false);
    }));
    btnSecondPass_.reset(new Button("2nd pass", [this] {
        generate_image(true);
    }));
    btnBtnResetImage_.reset(new Button("Reset", [this] {
        if (ask("Are you sure you want to reset the image?")) {
            reset_image();
        }
    }));
    btnPinSeed_.reset(new Button(xpm::image(xpm::img_24x24_green_pin), [this] {
        // does nothing
    }));
    btnPinSeed_->enableDownUp();
    brush_size_ = new Fl_Choice(0, 0, 1, 1, "Brush size");
    btnFromPalette_.reset(new Button(xpm::image(xpm::img_24x24_list), [this] {
        from_palette();
    }));
    btnToPalette_.reset(new Button(xpm::image(xpm::img_24x24_folder), [this] {
        to_palette();
    }));
    btnUseCurrent_.reset(new Button(xpm::image(xpm::img_24x24_picture), [this] {
        use_current_image();
    }));

    this->begin();
    btnSettings_.reset(new Button(xpm::image(xpm::img_24x24_settings), [this] {
        toggle_settings();
    }));
    btnSettings_->enableDownUp();

    btnOk_.reset(new Button("Ok", [this] {
        confirmed_ = true;
        hide();
    }));
    btnCancel_.reset(new Button("Cancel", [this] {
        hide();
    }));

    arch_input_->align(FL_ALIGN_TOP_LEFT);
    model_input_->align(FL_ALIGN_TOP_LEFT);
    prompt_input_->align(FL_ALIGN_TOP_LEFT);
    seed_input_->align(FL_ALIGN_TOP_LEFT);
    brush_size_->align(FL_ALIGN_TOP_LEFT);
    char buffer[64] = "";
    for (uint8_t i = 0; i < brush_size_count(); i++) {
        sprintf(buffer, "%d Pixels", brushes_sizes[i]);
        brush_size_->add(buffer);
    }
    brush_size_->value(0);
    int random_value = (rand() % 10000) + 1;
    sprintf(buffer, "%d", random_value);
    seed_input_->value(buffer);
    btnSettings_->tooltip("Prompt and other settings...");
    btnRandomSeed_->tooltip("Randomizes the seed");
    btnPinSeed_->tooltip("When pinned, it does not change the current after generating images");
    btnFromPalette_->tooltip("Pick an image from the image palette");
    btnToPalette_->tooltip("Send the generated image to the image palette");
    btnUseCurrent_->tooltip("Use the current image as the reference");
    settings_panel_->hide();
    align_components();
    load_arch_models();
    arch_input_->callback(cb_widget, this);
    brush_size_->callback(cb_widget, this);
    brush_size_selected();

    prompt_input_->value(last_prompt.c_str());
}

DrawingWindow::~DrawingWindow() {

}

void DrawingWindow::reset_image() {
    int w = image_panel_->view_settings()->at(1)->w();
    int h = image_panel_->view_settings()->at(1)->h();
    image_panel_->view_settings()->at(1)->replace_image(image_->duplicate());
    image_panel_->view_settings()->at(1)->w(w);
    image_panel_->view_settings()->at(1)->h(h);
}

image_ptr_t DrawingWindow::get_image() {
    image_ptr_t r;
    if (confirmed_) {
        r = image_panel_->view_settings()->at(0)->getImage()->duplicate();
    }
    return r;
}

void DrawingWindow::load_arch_models() {
    // lets load it from config
    std::string model_archs = get_config()->arch_speed_model();
    if (model_archs.empty()) {
        return;
    }
    if (*model_archs.rbegin() != ',') {
        model_archs += ",";
    }
    // it's on comma separated format like: arch:model,arch2:model2 lets first fill the map arch_models_ parsing it
    std::string::size_type pos = 0;
    arch_input_->clear();
    while ((pos = model_archs.find(",")) != std::string::npos) {
        std::string arch_model = model_archs.substr(0, pos);
        model_archs = model_archs.substr(pos + 1);
        std::string::size_type pos2 = arch_model.find(":");
        if (pos2 != std::string::npos) {
            std::string arch = arch_model.substr(0, pos2);
            std::string model = arch_model.substr(pos2 + 1);
            arch_models_.push_back(std::make_pair(arch, model));
        }
    }

    for (size_t i = 0; i < arch_models_.size(); i++) {
        arch_input_->add(arch_models_[i].first.c_str());
    }
    
    if (arch_models_.size() > 0) {
        arch_input_->value(0);
        loras_->refresh_models(get_arch());
    }
    update_model_list();
}


std::string DrawingWindow::positive_prompt() {
    std::string result = prompt_input_->value();
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

std::vector<std::string> DrawingWindow::get_loras() {
    std::vector<std::string> result;
    std::string text = prompt_input_->value();
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

bool DrawingWindow::validate() {
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

void DrawingWindow::update_model_list() {
    model_input_->clear();
    for (auto &arch_model : arch_models_) {
        if (arch_model.first == arch_input_->text()) {
            model_input_->add(arch_model.second.c_str());
        }
    }
    if (model_input_->size() > 0) {
        model_input_->value(0);
    }
}

void DrawingWindow::dfe_handle_event(void *sender, event_id_t event, void *data) {
    if (event == event_layer_mask_color_picked && sender == image_panel_) {
        color_palette_->update_current_color();
    } else if (event == event_prompt_lora_selected && sender == loras_.get()) {
        insert_current_lora();
    }
}

void DrawingWindow::align_components() {
    // the image_panel_ is positioned at top left with a margin of 5 pixels
    image_panel_->size(this->w() - 100, this->h() - 50);
    image_panel_->position(5, 5);
    
    btnSettings_->size(30, 30);
    btnSettings_->position(this->w() - 45, 5);
    right_panel_->size(80, this->h() - 90);
    right_panel_->position(this->w() - 90, btnSettings_->y() + btnSettings_->h() + 5);

    color_pal_group_->size(right_panel_->w() - 10, 288);
    color_pal_group_->position(right_panel_->x() + 5, right_panel_->y() + 5);

    if (settings_panel_->visible_r()) {
        right_panel_->size(310, right_panel_->h());
        right_panel_->position(this->w() - 310, right_panel_->y());
        image_panel_->size(this->w() - right_panel_->w() - 10, image_panel_->h());
    }

    settings_panel_->position(right_panel_->x() + 5, right_panel_->y() + 5);
    settings_panel_->size(300, right_panel_->h() - 10);
    
    prompt_input_->resize(settings_panel_->x() + 5, settings_panel_->y() + 10, settings_panel_->w() - 10, 150);
    seed_input_->resize(prompt_input_->x(), prompt_input_->y() + prompt_input_->h() + 25, (settings_panel_->w() - 15) / 2, 30);
    btnRandomSeed_->position(seed_input_->x() + seed_input_->w() + 5, seed_input_->y());
    btnRandomSeed_->size(30, 30);
    arch_input_->position(seed_input_->x(), seed_input_->y() + seed_input_->h() + 25);
    arch_input_->size(prompt_input_->w(), 30);
    model_input_->position(arch_input_->x(), arch_input_->y() + arch_input_->h() + 25);
    model_input_->size(arch_input_->w(), 30);
    lora_gp_->position(model_input_->x(), model_input_->y() + model_input_->h() + 25);
    lora_gp_->size(model_input_->w(), model_input_->w());

    brush_size_->position(color_pal_group_->x(), color_pal_group_->y() + color_pal_group_->h() + 25);
    brush_size_->size(right_panel_->w() - 10, 30);
    btnFromPalette_->position(brush_size_->x(), brush_size_->y() + brush_size_->h() + 5);
    btnFromPalette_->size(brush_size_->w() / 2, 30);
    btnUseCurrent_->position(btnFromPalette_->x() + btnFromPalette_->w() + 5, btnFromPalette_->y());
    btnUseCurrent_->size(btnFromPalette_->w(), 30);

    btnFirstPass_->position(brush_size_->x(), btnUseCurrent_->y() + btnUseCurrent_->h() + 5);
    btnFirstPass_->size(brush_size_->w(), 30);
    btnSecondPass_->position(btnFirstPass_->x(), btnFirstPass_->y() + btnFirstPass_->h() + 5);
    btnSecondPass_->size(btnFirstPass_->w(), 30);
    btnBtnResetImage_->position(btnSecondPass_->x(), btnSecondPass_->y() + btnSecondPass_->h() + 5);
    btnBtnResetImage_->size(btnSecondPass_->w(), 30);
    btnPinSeed_->position(btnBtnResetImage_->x(), btnBtnResetImage_->y() + btnBtnResetImage_->h() + 5);
    btnPinSeed_->size(btnBtnResetImage_->w(), 30);
    btnToPalette_->position(btnPinSeed_->x(), btnPinSeed_->y() + btnPinSeed_->h() + 5);
    btnToPalette_->size(btnPinSeed_->w() / 2, 30);

    // the buttons at the bottom right corner
    btnOk_->position(this->w() - 215, this->h() - 40);
    btnOk_->size(100, 30);
    btnCancel_->position(btnOk_->x() + btnOk_->w() + 2, btnOk_->y());
    btnCancel_->size(100, 30);

    loras_->alignComponents();
    color_palette_->aligncomponents();
}

void DrawingWindow::cb_widget(Fl_Widget *widget, void *data) {
    DrawingWindow *self = static_cast<DrawingWindow*>(data);
    self->cb_widget(widget);
}

void DrawingWindow::cb_widget(Fl_Widget *widget) {
    if (widget == arch_input_) {
        update_model_list();
        loras_->refresh_models(get_arch());
    } else if (widget == brush_size_) {
        brush_size_selected();
    }
}

void DrawingWindow::brush_size_selected() {
    image_panel_->view_settings()->brush_size(brushes_sizes[brush_size_->value()]);
}

std::string DrawingWindow::get_arch() {
    // get the selected architecture, if none return empty string
    if (arch_input_->size() == 0 || arch_input_->value() < 0) {
        return "";
    }
    return arch_input_->text();
}

std::string DrawingWindow::get_model() {
    // get the selected model, if none return empty string
    if (model_input_->size() == 0 || model_input_->value() < 0) {
        return "";
    }
    return model_input_->text();
}

int DrawingWindow::get_seed() {
    int result = -1;
    sscanf(seed_input_->value(), "%d", &result);
    if (result < 0) {
        result = rand() % 10000 + 1;
        char buffer[64] = "";
        sprintf(buffer, "%d", result);
        seed_input_->value(buffer);
    }
    return result;
}

void DrawingWindow::generate_image(bool second_pass) {
    std::string prompt = positive_prompt();
    if (prompt.empty()) {
        show_error("You need to provide a prompt!");
        return;
    }
    ws::diffusion::diffusion_request_t params;
    params.model_type = get_arch();
    params.model_name = get_model();
    if (params.model_name.empty()) {
        show_error("You need to select a model!");
        return;
    }
    if (!btnPinSeed_->down() && !second_pass) {
        random_seed();
    }
    params.prompt = prompt;
    params.negative_prompt = "";
    params.seed = second_pass ? -1 : get_seed();
    params.steps = second_pass ? 4 : 8;
    params.correct_colors = false;
    params.batch_size = 1;
    params.loras = get_loras();
    
    params.cfg = 0.0;
    params.scheduler =  "";
    params.width = 512;
    params.height = 512;
    params.use_lcm = true;
    params.use_tiny_vae = true;
    params.use_float16 = true;
    params.image_strength = second_pass ? 0.50 : 0.65;
    params.inpaint_mode = "original"; 

    auto img2 = image_panel_->view_settings()->at(second_pass ? 0 : 1)->getImage();
    params.images = {img2->duplicate()};
     
    auto result = run_diffusion(params);
    if (!result.empty()) {
        image_panel_->view_settings()->at(0)->replace_image(result[0]);
        image_panel_->redraw();
    }
}

void DrawingWindow::random_seed() {
    int seed = (rand() % 10000) + 1;
    char buffer[64] = "";
    sprintf(buffer, "%d", seed);
    seed_input_->value(buffer);
}

void DrawingWindow::toggle_settings() {
    if (btnSettings_->down()) {
        color_pal_group_->hide();
        settings_panel_->show();
        brush_size_->hide();
        btnFirstPass_->hide();
        btnSecondPass_->hide();
    } else {
        color_pal_group_->show();
        settings_panel_->hide();
        brush_size_->show();
        btnFirstPass_->show();
        btnSecondPass_->show();
    }
    align_components();
}

int DrawingWindow::handle(int event)
{
    switch (event)
    {
    case FL_KEYUP:
    {
        if (Fl::event_key() == FL_Escape)
        {
            return 1;
        }
        if (Fl::event_key() == FL_F + 1) {
            generate_image(false);
            return 1;
        }
        if (Fl::event_key() == FL_F + 2) {
            generate_image(true);
            return 1;
        }
    }
    break;
    case FL_KEYDOWN:
    {
        if (Fl::event_key() == FL_Escape)
        {
            return 1;
        }
    }
    break;
    }

    return Fl_Window::handle(event);
}

void DrawingWindow::from_palette() {
    auto img = pickup_image_from_palette();
    if (img) {
        image_ = pixelate_image(img);
        reset_image();
    }
}

void DrawingWindow::to_palette() {
    auto img = image_panel_->view_settings()->at(0)->getImage()->duplicate();
    add_image_palette(img);
}

void DrawingWindow::use_current_image() {
    auto img = pixelate_image(image_panel_->view_settings()->at(0)->getImage()->duplicate());
    image_ = img;
    reset_image();
}

void DrawingWindow::insert_current_lora() {
    std::string text = prompt_input_->value();
    auto current_concept = loras_->getSelected();
    if (current_concept.name.empty()) {
        return;
    }
    current_concept.name = std::string("<lora:") + current_concept.name;
    if (text.find(current_concept.name) == text.npos) {
        text += " ";
        text += current_concept.name + ":1.0>";
    }
    prompt_input_->value(text.c_str());
}

image_ptr_t draw_image(image_ptr_t reference_image) {
    image_ptr_t r;

    if (!reference_image) {
        show_error("You need two images to merge!");
        return r;
    }

    if (reference_image->w() < 32 || reference_image->h() < 32) {
        show_error("It's required an image with at least 32x32 pixels!");
        return r;
    }

    auto window = new DrawingWindow(reference_image);
    window->show();
    while (true) {
        if (!window->visible_r()) {
            break;
        }
        Fl::wait();
    }
    
    r = window->get_image();

    Fl::delete_widget(window);
    Fl::do_widget_deletion();

    return r;
}

}