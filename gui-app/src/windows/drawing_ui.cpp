#include <cstdlib>
#include "misc/dialogs.h"

#include "websocket/tasks.h"
#include "components/xpm/xpm.h"
#include "misc/config.h"
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
    };
}

DrawingWindow::DrawingWindow(image_ptr_t reference_img) : Fl_Window(Fl::w() / 2 - 1024 / 2, Fl::h() / 2 - 640 / 2, 1024, 640, "Image palette - Select an image"),
        SubscriberThis(drawing_ui_events) {
    this->set_modal();

    image_ = reference_img->blur(4)->resizeImage(32, 32)->resizeImage(512, 512);
    
    this->begin();
    image_panel_ = new LayerDrawingImagePanel(0, 0, 1, 1, "Image");
    image_panel_->view_settings()->add_layer(newImage(512, 512, true));
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
            int w = image_panel_->view_settings()->at(1)->w();
            int h = image_panel_->view_settings()->at(1)->h();
            image_panel_->view_settings()->at(1)->replace_image(image_->duplicate());
            image_panel_->view_settings()->at(1)->w(w);
            image_panel_->view_settings()->at(1)->h(h);
        }
    }));
    btnPinSeed_.reset(new Button(xpm::image(xpm::img_24x24_green_pin), [this] {
        // does nothing
    }));
    btnPinSeed_->enableDownUp();
    brush_size_ = new Fl_Choice(0, 0, 1, 1, "Brush size");

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
    settings_panel_->hide();
    align_components();
    load_arch_models();
    arch_input_->callback(cb_widget, this);
    brush_size_->callback(cb_widget, this);
    brush_size_selected();
}

DrawingWindow::~DrawingWindow() {

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
    printf("Arch models: %s\n", model_archs.c_str());
    while ((pos = model_archs.find(",")) != std::string::npos) {
        std::string arch_model = model_archs.substr(0, pos);
        model_archs = model_archs.substr(pos + 1);
        std::string::size_type pos2 = arch_model.find(":");
        if (pos2 != std::string::npos) {
            std::string arch = arch_model.substr(0, pos2);
            std::string model = arch_model.substr(pos2 + 1);
            printf("Arch: %s Model: %s\n", arch.c_str(), model.c_str());
            arch_models_.push_back(std::make_pair(arch, model));
        }
    }

    arch_input_->clear();
    for (auto &arch_model : arch_models_) {
        arch_input_->add(arch_model.first.c_str());
    }
    if (arch_models_.size() > 0) {
        arch_input_->value(0);
        update_model_list();
    }
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

    settings_panel_->size(300, 288);
    settings_panel_->position(right_panel_->x() + 5, right_panel_->y() + 5);
    
    prompt_input_->resize(settings_panel_->x() + 5, settings_panel_->y() + 25, settings_panel_->w() - 10, 200);
    seed_input_->resize(prompt_input_->x(), prompt_input_->y() + prompt_input_->h() + 25, (settings_panel_->w() - 15) / 2, 30);
    btnRandomSeed_->position(seed_input_->x() + seed_input_->w() + 5, seed_input_->y());
    btnRandomSeed_->size(30, 30);
    arch_input_->position(seed_input_->x(), seed_input_->y() + seed_input_->h() + 25);
    arch_input_->size(prompt_input_->w(), 30);
    model_input_->position(arch_input_->x(), arch_input_->y() + arch_input_->h() + 25);
    model_input_->size(arch_input_->w(), 30);
    
    brush_size_->position(color_pal_group_->x(), color_pal_group_->y() + color_pal_group_->h() + 25);
    brush_size_->size(right_panel_->w() - 10, 30);
    btnFirstPass_->position(brush_size_->x(), brush_size_->y() + brush_size_->h() + 5);
    btnFirstPass_->size(brush_size_->w(), 30);
    btnSecondPass_->position(btnFirstPass_->x(), btnFirstPass_->y() + btnFirstPass_->h() + 5);
    btnSecondPass_->size(btnFirstPass_->w(), 30);
    btnBtnResetImage_->position(btnSecondPass_->x(), btnSecondPass_->y() + btnSecondPass_->h() + 5);
    btnBtnResetImage_->size(btnSecondPass_->w(), 30);
    btnPinSeed_->position(btnBtnResetImage_->x(), btnBtnResetImage_->y() + btnBtnResetImage_->h() + 5);
    btnPinSeed_->size(btnBtnResetImage_->w(), 30);

    // the buttons at the bottom right corner
    btnOk_->position(this->w() - 215, this->h() - 40);
    btnOk_->size(100, 30);
    btnCancel_->position(btnOk_->x() + btnOk_->w() + 2, btnOk_->y());
    btnCancel_->size(100, 30);

    color_palette_->aligncomponents();
}

void DrawingWindow::cb_widget(Fl_Widget *widget, void *data) {
    DrawingWindow *self = static_cast<DrawingWindow*>(data);
    self->cb_widget(widget);
}

void DrawingWindow::cb_widget(Fl_Widget *widget) {
    if (widget == arch_input_) {
        update_model_list();
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
    std::string prompt = prompt_input_->value();
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
    params.steps = second_pass ? 8 : 8;
    params.correct_colors = false;
    params.batch_size = 1;
    
    params.cfg = 0.0;
    params.scheduler =  "";
    params.width = 512;
    params.height = 512;
    params.use_lcm = false;
    params.use_float16 = true;
    params.image_strength = second_pass ? 0.25 : 0.85;
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