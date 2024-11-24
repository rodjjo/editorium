#include <FL/fl_ask.H>

#include "misc/utils.h"
#include "misc/dialogs.h"
#include "misc/config.h"
#include "images/image_palette.h"
#include "components/xpm/xpm.h"

#include "websocket/tasks.h"
#include "windows/sapiens_ui.h"
#include "windows/chatbot_ui.h"
#include "windows/copy_region_ui.h"

#include "windows/diffusion_ui.h"



namespace editorium
{
    namespace {
        const std::list<event_id_t> diffusion_tool_events = {
            event_generator_next_image,
            event_generator_previous_image,
            event_generator_accept_image,
            event_generator_accept_partial_image,
            event_generator_save_current_image,
            event_generator_send_to_palette,
            event_image_frame_new_mask,
            event_image_frame_open_mask,
            event_image_frame_mode_selected,
            event_prompt_architecture_selected,
            event_prompt_improve_requested,
            event_prompt_improve_requested2,
            event_prompt_interrogate_requested,
            event_image_frame_seg_gdino,
            event_image_frame_seg_sapiens,
            event_layer_mask_color_picked
        };

       const char *page_names[page_type_count] = {
            "Prompts",
            "Image",
            "Controlnet 1",
            "Controlnet 2",
            "Controlnet 3",
            "Controlnet 4",
            "Ip Adapter 1",
            "Ip Adapter 2",
            "Generate"
        };
    }

    DiffusionWindow::DiffusionWindow() : Fl_Double_Window(
        Fl::w() / 2 - 860 / 2, Fl::h() / 2 - 640 / 2,
        860, 640, "Image generator"
    ), view_settings_(NULL), SubscriberThis(diffusion_tool_events)  {
        after_constructor();
    }

    DiffusionWindow::DiffusionWindow(ViewSettings *view_settings) : Fl_Double_Window(
        Fl::w() / 2 - 860 / 2, Fl::h() / 2 - 640 / 2,
        860, 640, "Image generator"
    ), view_settings_(view_settings), SubscriberThis(diffusion_tool_events) {
        after_constructor();
    }

    void DiffusionWindow::after_constructor() {
        this->size_range(this->w(), this->h());
        this->begin();
        bottom_panel_ = new Fl_Group(0, 0, 1, 1);
        bottom_panel_->box(FL_DOWN_BOX);
        {
            bottom_panel_->begin();
            btnOk_.reset(new Button(xpm::image(xpm::img_24x24_ok), [this] {
                accept_current_image();
            }));

            btnCancel_.reset(new Button(xpm::image(xpm::img_24x24_abort), [this] {
                check_accept_current_image();
            }));

            bottom_panel_->end();
        }
        this->begin();
        btnGenerate_.reset(new Button(xpm::image(xpm::img_24x24_magic_wand), [this] {
            generate();
        }));
        right_panel_ = new Fl_Group(0, 0, 1, 1);
        right_panel_->box(FL_DOWN_BOX);
        { // page selector
            right_panel_->begin();
            selector_ = new Fl_Select_Browser(0, 0, 1, 1);
            for (int i = 0; i < page_type_count; i++) {
                selector_->add(page_names[i]);
            }
            selector_->callback(page_cb, this);
            right_panel_->end();
        }

        image_ptr_t reference_mask;
        image_ptr_t reference_img;
        if (view_settings_) {
            reference_mask = view_settings_->get_selected_image()->create_mask_from_alpha_channel();
            auto selected_img = view_settings_->get_selected_image();
            if (selected_img->format() == img_rgba) {
                reference_img = selected_img->to_rgb();
            } else {
                reference_img = selected_img;
            }
        }

        page_type_t where;
        for (int i = (page_type_t)0; i < page_type_count; i++) {
            where = static_cast<page_type_t>(i);
            images_[where] = NULL;
            char buffer[128];
            sprintf(buffer, "DiffusionWindow_%d", i);
            this->begin();
            pages_[where] = new Fl_Group(0, 0, 1, 1);
            pages_[where]->box(FL_DOWN_BOX);
            pages_[where]->begin();

            if (i != page_type_prompt) {
                if (where == page_type_image) {
                    images_[where] = new ColoredMaskEditableImagePanel(0, 0, 1, 1, titles_[where].c_str());        
                    if (reference_img) {
                        images_[where]->view_settings()->set_image(reference_img);
                        images_[where]->cancel_refresh();
                        if (reference_mask) {
                            images_[where]->view_settings()->set_mask();
                            if (images_[where]->view_settings()->layer_count() > 2) {
                                images_[where]->view_settings()->at(2)->replace_image(reference_mask);
                            }
                        }
                    }
                } else {
                    images_[where] = new NonEditableImagePanel(0, 0, 1, 1, titles_[where].c_str());
                }
                titles_[where] = buffer;
                if (i == page_type_image) {
                    image_frame_.reset(new ImageFrame(pages_[where],  images_[where]));
                } else if (i == page_type_results) {
                    result_frame_.reset(new ResultFrame(pages_[where],  images_[where]));
                } else if (i >= page_type_controlnet1 && i <= page_type_controlnet4) {
                    control_frames_[where] = std::unique_ptr<ControlnetFrame>(new ControlnetFrame(pages_[where],  images_[where], images_[page_type_image], false));
                } else {
                    control_frames_[where] = std::unique_ptr<ControlnetFrame>(new ControlnetFrame(pages_[where],  images_[where], images_[page_type_image], true));
                }
            } else {
                prompt_frame_.reset(new PromptFrame(pages_[where]));
            }

            pages_[where]->end();
            pages_[where]->hide();
        }
        this->end();

        alignComponents();
        selector_->value(1);
        prompt_frame_->refresh_models();
        set_architecture_view();

        if (reference_img) {
            selector_->value(2);
            image_frame_->enable_mode();
        }

        show_current_page();
    }

    DiffusionWindow::~DiffusionWindow() {

    }

    int DiffusionWindow::handle(int event){
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

    void DiffusionWindow::resize(int x, int y, int w, int h) {
        Fl_Double_Window::resize(x, y, w, h);
        if (last_size_w_ != w || last_size_h_ != h) {
            last_size_w_ = w;
            last_size_h_ = h;
            alignComponents();
        }
    }

    void DiffusionWindow::alignComponents() {
        bottom_panel_->resize(0, 0, w() - 6, 40);
        bottom_panel_->position(3, h() - bottom_panel_->h() - 3);

        btnOk_->position(this->w() - 215, this->h() - 37);
        btnOk_->size(100, 30);
        btnCancel_->position(btnOk_->x() + btnOk_->w() + 2, btnOk_->y());
        btnCancel_->size(100, 30);
        btnGenerate_->size(100, 30);
        btnGenerate_->position(w() - btnGenerate_->w() - 5, 5);
        right_panel_->resize(0, 0, btnGenerate_->w(), h() - bottom_panel_->h() - btnGenerate_->h() - 20);
        right_panel_->position(w() - right_panel_->w() - 5, btnGenerate_->y() + btnGenerate_->h() + 5);
        selector_->resize(
            right_panel_->x(),
            right_panel_->y(),
            right_panel_->w(),
            right_panel_->h()
        );

        page_type_t where;
        for (int i = (page_type_t)0; i < page_type_count; i++) {
            where = static_cast<page_type_t>(i);
            pages_[where]->resize(5, 5, w() - right_panel_->w() - 17, h() - bottom_panel_->h() - 14);
            pages_[where]->position(7, 7);
            if (images_[where]) {
                images_[where]->resize(5, 5, pages_[where]->w() - 150, pages_[where]->h() - 14);
                images_[where]->position(pages_[where]->w() - images_[where]->w() - 7, pages_[where]->y() + 7);
            }
        }

        prompt_frame_->alignComponents();
        image_frame_->alignComponents();
        result_frame_->alignComponents();
        for (int i = page_type_controlnet1; i < page_type_controlnet4 + 1; i++)  {
            where = static_cast<page_type_t>(i);
            control_frames_[where]->alignComponents();
        }
        for (int i = page_type_ip_adapter1; i < page_type_ip_adapter2 + 1; i++)  {
            where = static_cast<page_type_t>(i);
            control_frames_[where]->alignComponents();
        }
    }
    
    void DiffusionWindow::page_cb(Fl_Widget* widget, void *cbdata) {
        static_cast<DiffusionWindow*>(cbdata)->page_cb(widget);
    }

    void DiffusionWindow::page_cb(Fl_Widget* widget) {
        if (selecting_page_) {
            return;
        }
        selecting_page_ = true;
        int idx = selector_->value();
        if (idx > 0)  {
            show_current_page();
        } 
        selector_->deselect();
        selector_->select(idx);
        selecting_page_ = false;
    }

    void DiffusionWindow::show_current_page() {
        page_type_t where;
        for (int i = (page_type_t)0; i < page_type_count; i++) {
            where = static_cast<page_type_t>(i);
            pages_[where]->hide();
        }
        int idx = selector_->value() - 1;
        if (idx >= visible_pages_.size()) {
            idx = visible_pages_.size() - 1;
        }
        if (idx >= 0)  {
            where = visible_pages_[idx];
            pages_[where]->show();
            if (images_[where] && !images_[where]->shown()) {
                if (idx == page_type_image) {
                    if (image_frame_->enabled()) {
                        images_[where]->show();    
                    }
                } else if ((idx >= page_type_controlnet1 && idx <= page_type_controlnet4)||
                            (idx >= page_type_ip_adapter1 && idx <= page_type_ip_adapter2)) {
                    images_[where]->show();
                    control_frames_[where]->alignComponents();
                } else {
                    if (control_frames_[where]->enabled()) {
                        images_[where]->show();
                    }
                }
            }
            if (where == page_type_image) {
                btnOk_->show();
            } else {
                btnOk_->hide();
            }
        }
    }

    void DiffusionWindow::enable_masking(ImagePanel *panel) {
        if (panel->view_settings()->layer_count() < 3) {
            panel->enable_color_mask_editor(true);
            panel->view_settings()->set_mask();
            panel->view_settings()->at(0)->pinned(true);
            panel->view_settings()->at(0)->focusable(false);
            panel->view_settings()->at(2)->pinned(true);
            panel->view_settings()->at(2)->focusable(false);
            panel->enable_color_mask_editor(false);
        }
    }

    void DiffusionWindow::dfe_handle_event(void *sender, event_id_t event, void *data) {
        if (sender == prompt_frame_.get()) {
            switch (event) {
                case event_prompt_architecture_selected:
                    set_architecture_view();
                break;
                case event_prompt_improve_requested:
                case event_prompt_improve_requested2:
                    improve_prompt(event == event_prompt_improve_requested2);
                    break;
                case event_prompt_interrogate_requested:
                    interrogate_image();
                    break;
            }
        } else if (sender == result_frame_.get()) {
            switch (event) {
                case event_generator_next_image:
                    if (result_index_ + 1 < results_.size()) {
                        result_index_ += 1;
                        show_current_result();
                    }
                break;

                case event_generator_previous_image:
                    if (result_index_ > 0) {
                        result_index_ -= 1;
                        show_current_result();
                    }
                break;

                case event_generator_accept_image:
                    if (images_[page_type_results] && images_[page_type_image]) {
                        if (images_[page_type_results]->view_settings()->layer_count() > 0) {
                            image_generated_ = true;
                            auto img = images_[page_type_results]->view_settings()->at(0)->getImage()->duplicate();
                            if (images_[page_type_image]->view_settings()->layer_count() > 0) {
                                images_[page_type_image]->view_settings()->at(0)->replace_image(
                                    img
                                );
                            } else {
                                images_[page_type_image]->view_settings()->add_layer(
                                    img
                                );
                            }
                            image_frame_->enable_mode();
                        }
                    }
                break;

                case event_generator_accept_partial_image:
                    accept_current_image_partial();
                break;

                case event_generator_save_current_image:
                    if (images_[page_type_results] && images_[page_type_results]->view_settings()->layer_count() > 0) {
                        auto img = images_[page_type_results]->view_settings()->at(0)->getImage()->duplicate();
                        choose_and_save_image("generator_results", img);
                    }
                break;

                case event_generator_send_to_palette:
                    if (images_[page_type_results] && images_[page_type_results]->view_settings()->layer_count() > 0) {
                        auto img = images_[page_type_results]->view_settings()->at(0)->getImage()->duplicate();
                        add_image_palette(img);
                    } else {
                        show_error("No image to send to the palette!");
                    }
                break;
            };
        } else if (sender == image_frame_.get()) {
            switch (event) {
                case event_image_frame_new_mask:
                    if (images_[page_type_image]->view_settings()->layer_count() < 1) {
                        show_error("Open or generate an image first!");
                    } else {
                        if (ask("Do you want to clear current mask?")) {
                            auto pn = images_[page_type_image];
                            enable_masking(pn);
                            pn->view_settings()->set_mask();
                        }
                    }
                break;

                case event_image_frame_open_mask:
                    if (images_[page_type_image]->view_settings()->layer_count() < 1) {
                        show_error("Open or generate an image first!");
                    } else {
                        auto r = choose_and_open_image("inpaint_mask");
                        if (!r) {
                            return;
                        }
                        auto pn = images_[page_type_image];
                        enable_masking(pn);
                        images_[page_type_image]->view_settings()->at(2)->replace_image(r);
                    }
                break;

                case event_image_frame_seg_gdino:
                    if (images_[page_type_image]->view_settings()->layer_count() < 1) {
                        show_error("Open or generate an image first!");
                    } else {
                        auto classes = fl_input("Enter the classes to segment (comma separated)", "");
                        if (!classes) {
                            return;
                        }
                        auto r = ws::diffusion::run_seg_ground_dino(classes, {images_[page_type_image]->view_settings()->at(0)->getImage()->duplicate()});
                        if (r.empty()) {
                            return;
                        }
                        r[0] = r[0]->dilate(3);
                        auto pn = images_[page_type_image];
                        enable_masking(pn);
                        pn->view_settings()->at(2)->replace_image(r[0]);
                    }
                break;

                case event_image_frame_seg_sapiens:
                    if (images_[page_type_image]->view_settings()->layer_count() < 1) {
                        show_error("Open or generate an image first!");
                    } else {
                        std::string classes = select_sapien_classes();
                        if (classes.empty()) {
                            return;
                        }
                        auto r = ws::diffusion::run_seg_sapiens(classes, {images_[page_type_image]->view_settings()->at(0)->getImage()->duplicate()});
                        if (r.empty()) {
                            return;
                        }
                        r[0] = r[0]->dilate(3);
                        auto pn = images_[page_type_image];
                        enable_masking(pn);
                        pn->view_settings()->at(2)->replace_image(r[0]);
                    }
                break;

                case event_image_frame_mode_selected:
                    images_[page_type_image]->view_settings()->brush_size(image_frame_->get_brush_size());
                    if (image_frame_->get_mode() == img2img_text || images_[page_type_image]->view_settings()->layer_count() < 1) {
                        return;
                    } else if (image_frame_->get_mode() == img2img_img2img) {
                        if (images_[page_type_image]->view_settings()->layer_count() > 1) {
                            images_[page_type_image]->view_settings()->at(1)->visible(false);
                        }
                    } else {
                        if (images_[page_type_image]->view_settings()->layer_count() < 3) {
                            images_[page_type_image]->view_settings()->set_mask();
                        }
                        images_[page_type_image]->view_settings()->at(1)->visible(true);
                    }

                break;
            }
        } else {
            switch (event) {
                case event_layer_mask_color_picked:
                    image_frame_->handle_event(event, sender);
                break;
            }
        }
    }

    void DiffusionWindow::accept_current_image_partial() {
        if (!images_[page_type_results] || !images_[page_type_image]) {
            return;
        }
        if (images_[page_type_results]->view_settings()->layer_count() < 1 || images_[page_type_image]->view_settings()->layer_count() < 1) {
            show_error("You need a image in the images pages and a new generated one!\nIt's required two images to proceed!");
            return;
        }
        auto new_generated_image = images_[page_type_results]->view_settings()->at(0)->getImage()->duplicate();
        auto original_image = images_[page_type_image]->view_settings()->at(0)->getImage()->duplicate();
        auto merged = copy_image_region(original_image, new_generated_image);
        if (merged) {
            images_[page_type_image]->view_settings()->at(0)->replace_image(merged);
        }
    }

    void DiffusionWindow::accept_current_image() {
        if (images_[page_type_image]->view_settings()->layer_count() < 1) {
            show_error("You should generate an image and send it to image page first!");
            return;
        }
        confirm_ = true;
        this->hide();
    }

    void DiffusionWindow::check_accept_current_image() {
        if (image_generated_) {
            if (ask("Do you want to accept the generated image?")) {
                accept_current_image();
            }
        } 
        this->hide();
    }

    image_ptr_t DiffusionWindow::get_current_image() {
        if (images_[page_type_image]->view_settings()->layer_count() > 0) {
            auto img1 = images_[page_type_image]->view_settings()->at(0)->getImage()->duplicate();
            return img1;
        }
        return image_ptr_t();
    }

    bool DiffusionWindow::was_confirmed() {
        return confirm_;
    }

    image_ptr_t DiffusionWindow::choose_and_open_image(const char * scope) {
        std::string result = choose_image_to_open_fl(scope);
        if (!result.empty()) {
            auto dir = filepath_dir(result);
            return ws::filesystem::load_image(result.c_str());
        }
        return image_ptr_t();
    }

    void DiffusionWindow::choose_and_save_image(const char * scope, image_ptr_t image) {
        std::string result = choose_image_to_save_fl(scope);
        if (!result.empty()) {
            auto dir = filepath_dir(result);
            ws::filesystem::save_image(result.c_str(), image, result.find(".png") != std::string::npos);
        }
    }

    bool DiffusionWindow::page_visible(page_type_t page) {
        for (auto & p : visible_pages_) {
            if (p == page) {
                return true;
            }
        }
        return false;
    }
    
    void DiffusionWindow::improve_prompt(bool second_pass) {
        std::string current_prompt = prompt_frame_->positive_prompt();
        if (current_prompt.empty()) {
                        show_error("Please fill the positive prompt first!");
        } else {
            auto arch = prompt_frame_->get_arch();
            auto system_prompt = get_prompts_for_chat("Configuration - for prompt improvement - arch " + arch, arch + "::improve-prompt" + (second_pass ? "::second-pass" : ""));
            if (!system_prompt.empty()) {
                ws::chatbots::chatbot_request_t req;
                req.context = system_prompt;
                req.prompt = current_prompt;
                auto result = ws::chatbots::chat_bot(req);
                if (!result.empty()) {
                    prompt_frame_->positive_prompt(result, true);
                }
            }
        }
    }

    void DiffusionWindow::interrogate_image() {
        if (images_[page_type_image]->view_settings()->layer_count() < 1) {
            show_error("Open or generate an image first!");
        } else {
            auto img = images_[page_type_image]->view_settings()->at(0)->getImage()->duplicate();
            auto arch = prompt_frame_->get_arch();
            auto sys_user_prompt = get_prompts_for_vision("Configuration - for image interrogation - arch " + arch , arch + "::prompt-from-image");
            if (!sys_user_prompt.first.empty() && !sys_user_prompt.second.empty()) {
                ws::chatbots::vision_chat_request_t req;
                req.system_prompt = sys_user_prompt.first;
                req.prompt = sys_user_prompt.second;
                req.image = img;
                auto result = ws::chatbots::chat_bot_vision(req);
                if (!result.empty()) {
                    prompt_frame_->positive_prompt(result, true);
                }
            }
        }
    }

    void DiffusionWindow::generate() {
        if (image_frame_->get_mode() != img2img_text) {
            if (images_[page_type_image]->view_settings()->layer_count() < 1) {
                show_error("Base Image is enabled. A image is necessary to proceed !");
                return;
            }
            if (image_frame_->get_mode() != img2img_img2img) {
                if (images_[page_type_image]->view_settings()->layer_count() < 2) {
                    show_error("Base Image in inpainting mode. A mask is necessary to proceed !");
                    return;
                }
            }
            if (!prompt_frame_->validate()) {
                return;
            }
        }
        
        ws::diffusion::diffusion_request_t params;

        params.model_type = prompt_frame_->get_arch();
        params.prompt = prompt_frame_->positive_prompt();
        params.negative_prompt = prompt_frame_->negative_prompt();
        params.seed = prompt_frame_->get_seed();
        params.steps = prompt_frame_->get_steps();
        params.correct_colors = prompt_frame_->get_correct_colors();
        params.batch_size = prompt_frame_->get_batch_size();
        params.model_name = prompt_frame_->get_model();
        params.cfg = prompt_frame_->get_cfg();
        params.scheduler =  prompt_frame_->get_scheduler();
        params.width = prompt_frame_->get_width();
        params.height = prompt_frame_->get_height();
        params.use_lcm = prompt_frame_->use_lcm_lora();
        params.use_float16 = get_config()->use_float16(); 
        params.image_strength = image_frame_->get_strength();
        params.inpaint_mode = "original"; // TODO: get from config

        int original_width = params.width;
        int original_height = params.height;

        if (image_frame_->get_mode() != img2img_text) {
            auto img1 = images_[page_type_image]->view_settings()->at(0)->getImage()->duplicate();
            if (images_[page_type_image]->view_settings()->layer_count() > 2) {
                auto img2 = images_[page_type_image]->view_settings()->at(1)->getImage();
                img1->pasteAt(0, 0, img2);
            }
            params.images = {img1};


            original_width = params.images[0]->w();
            original_height = params.images[0]->h();
            params.images[0] = params.images[0]->ensureMultipleOf8();
            params.width = params.images[0]->w();
            params.height = params.images[0]->h();
            size_t mask_index = images_[page_type_image]->view_settings()->layer_count() > 2 ? 2 : 1;
            if (image_frame_->get_mode() == img2img_inpaint_masked) {
                params.masks = {images_[page_type_image]->view_settings()->at(mask_index)->getImage()->rgba_mask_into_black_white()};
            } else if (image_frame_->get_mode() == img2img_inpaint_not_masked) {
                params.masks = {images_[page_type_image]->view_settings()->at(mask_index)->getImage()->rgba_mask_into_black_white(true)};
            }
            
            if (!params.masks.empty()) {
                params.masks[0] = params.masks[0]->ensureMultipleOf8();
            }
        }

        params.loras = prompt_frame_->get_loras();

        page_type_t controlnet_pages[] = {
            page_type_controlnet1, 
            page_type_controlnet2, 
            page_type_controlnet3, 
            page_type_controlnet4
        };
        for (int i = 0; i < sizeof(controlnet_pages) / sizeof(controlnet_pages[0]); i++) {
            auto & frame = control_frames_[controlnet_pages[i]];
            if (!frame->enabled() || !page_visible(controlnet_pages[i])) {
                continue;
            }
            ws::diffusion::control_image_t controlnet;
            controlnet.first.first = frame->getModeStr();
            controlnet.first.second = frame->getStrength();
            controlnet.second = frame->getImage();
            if (controlnet.second) {
                params.controlnets.push_back(controlnet);
            }
        }
        
        page_type_t ipadapter_pages[] = {
            page_type_ip_adapter1, 
            page_type_ip_adapter2, 
        };
        for (int i = 0; i < sizeof(ipadapter_pages) / sizeof(ipadapter_pages[0]); i++) {
            auto & frame = control_frames_[ipadapter_pages[i]];
            if (!frame->enabled() || !page_visible(ipadapter_pages[i])) {
                continue;
            }
            ws::diffusion::control_image_t ip_adapter;
            ip_adapter.first.first = frame->getModeStr();
            ip_adapter.first.second = frame->getStrength();
            ip_adapter.second = frame->getImage();
            if (ip_adapter.second) {
                params.ip_adapters.push_back(ip_adapter);
            }
        }

        auto result = run_diffusion(params);

        if (!result.empty()) {
            if (image_frame_->get_mode() == img2img_inpaint_masked || 
                image_frame_->get_mode() == img2img_inpaint_not_masked) {
            }
            size_t index = results_.size();
            for (auto & img : result) {
                results_.push_back(img);            
            }
            if (results_.size() > 16) {
                results_.erase(results_.begin(), results_.begin() + (results_.size() - 17));
            }
            if (index >= results_.size()) {
                index = results_.size() - 1;
            }
            result_index_ = index;
            show_current_result();

            prompt_frame_->save_profile();
        }
    }

    void DiffusionWindow::show_current_result() {
        if (result_index_ < results_.size()) {
            images_[page_type_results]->view_settings()->set_image(results_[result_index_]);
            char buffer[100];
            sprintf(buffer, "%d of %d", (int)result_index_ + 1, (int)results_.size());
            result_frame_->set_page_text(buffer);
        }
    }

    const char *DiffusionWindow::get_mode() {
        if (image_frame_->get_mode() == img2img_text) {
            return "txt2img";
        }
        if (image_frame_->get_mode() == img2img_img2img) {
            return "img2img";
        }
        return "inpaint";
    }

    void DiffusionWindow::set_architecture_view() {
        auto caps = ws::diffusion::get_architecture_features(prompt_frame_->get_arch());
        image_frame_->inpaint_enabled(caps.support_inpaint);
        for (int i = page_type_controlnet1; i < page_type_controlnet4 + 1; i++) {
            auto & frame = control_frames_[static_cast<page_type_t>(i)];
            if (frame) {
                frame->supported_modes(caps.controlnet_types);
            }
        }
        for (int i = page_type_ip_adapter1; i < page_type_ip_adapter2 + 1; i++) {
            auto & frame = control_frames_[static_cast<page_type_t>(i)];
            if (frame) {
                frame->supported_modes(caps.ip_adapter_types);
            }
        }
        visible_pages_.clear();
        visible_pages_.push_back(page_type_prompt);
        visible_pages_.push_back(page_type_image);
        if (caps.controlnet_types.size() > 0) {
            for (int i = 0; i < caps.controlnet_types.size() && i < 4; i++) {
                visible_pages_.push_back(static_cast<page_type_t>(page_type_controlnet1 + i));
            }
        }
        if (caps.ip_adapter_types.size() > 0) {
            for (int i = 0; i < caps.ip_adapter_types.size() && i < 2; i++) {
                visible_pages_.push_back(static_cast<page_type_t>(page_type_ip_adapter1 + i));
            }
        }
        visible_pages_.push_back(page_type_results);
        
        auto selector_index = selector_->value();
        auto selected_text = selector_->value() > 0 ? selector_->text(selector_index) : "";
        selector_->clear();
        for (auto & page : visible_pages_) {
            selector_->add(page_names[page]);
        }
        selector_->value(1);
        for (int i = 0; i < visible_pages_.size(); i++) {
            if (selected_text == page_names[visible_pages_[i]]) {
                selector_->value(i + 1);
                break;
            }
        }
    }

    image_ptr_t generate_image_(bool modal, ViewSettings* view_settings) {
        image_ptr_t r;
        DiffusionWindow *window = view_settings ? new DiffusionWindow(view_settings) : new DiffusionWindow();
        if (modal) {
            window->set_modal();
        }
        window->show();
        while (true) {
            if (!window->visible_r()) {
                break;
            }
            Fl::wait();
        }
        if (window->was_confirmed()) {
            r = window->get_current_image();
        }
        Fl::delete_widget(window);
        Fl::do_widget_deletion();
        return r;
    }


    image_ptr_t generate_image(bool modal) {
        return generate_image_(modal, NULL);
    }

    image_ptr_t generate_image(bool modal, ViewSettings* view_settings) {
        return generate_image_(modal, view_settings);
    }

    

} // namespace editorium
