
#include "misc/utils.h"
#include "misc/dialogs.h"
#include "misc/config.h"
#include "components/xpm/xpm.h"

#include "windows/diffusion_ui.h"
#include "websocket/tasks.h"

namespace editorium
{
    namespace {
        const std::list<event_id_t> diffusion_tool_events = {
            event_generator_next_image,
            event_generator_previous_image,
            event_generator_accept_image,
            event_generator_accept_partial_image,
            event_generator_save_current_image,
            event_image_frame_new_mask,
            event_image_frame_open_mask,
            event_image_frame_mode_selected,
            event_prompt_architecture_selected
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
                this->hide();        
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

        image_ptr_t reference_img;
        if (view_settings_) {
            reference_img = view_settings_->get_selected_image();
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
                    images_[where] = new MaskEditableImagePanel(0, 0, 1, 1, titles_[where].c_str());        
                    if (reference_img) {
                        images_[where]->view_settings()->set_image(reference_img);
                        images_[where]->cancel_refresh();
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
        alignComponents();
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
                } else if (idx >= page_type_controlnet1 && idx <= page_type_controlnet4) {
                    images_[where]->show();
                } else if (idx >= page_type_ip_adapter1 && idx <= page_type_ip_adapter2) {
                    images_[where]->show();
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

    void DiffusionWindow::dfe_handle_event(void *sender, event_id_t event, void *data) {
        if (sender == prompt_frame_.get()) {
            switch (event) {
                case event_prompt_architecture_selected:
                    set_architecture_view();
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
                break;

                case event_generator_save_current_image:
                    if (images_[page_type_results] && images_[page_type_results]->view_settings()->layer_count() > 0) {
                        auto img = images_[page_type_results]->view_settings()->at(0)->getImage()->duplicate();
                        choose_and_save_image("generator_results", img);
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
                            images_[page_type_image]->view_settings()->set_mask();
                        }
                    }
                break;

                case event_image_frame_open_mask:
                    if (images_[page_type_image]->view_settings()->layer_count() < 1) {
                        show_error("Open or generate an image first!");
                    } else {
                        auto r = choose_and_open_image("inpaint_mask");
                        if (images_[page_type_image]->view_settings()->layer_count() < 2) {
                            images_[page_type_image]->view_settings()->add_layer(r);
                        } else {
                            images_[page_type_image]->view_settings()->at(1)->replace_image(r);
                        }
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
                        if (images_[page_type_image]->view_settings()->layer_count() < 2) {
                            images_[page_type_image]->view_settings()->set_mask();
                        }
                        images_[page_type_image]->view_settings()->at(1)->visible(true);
                    }

                break;
            }
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

    image_ptr_t DiffusionWindow::get_current_image() {
        image_ptr_t r;
        if (images_[page_type_image]->view_settings()->layer_count() > 0) {
            r = images_[page_type_image]->view_settings()->at(0)->getImage()->duplicate();
        }
        return r;
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
        }
        
        ws::diffusion::diffusion_request_t params;

        params.model_type = prompt_frame_->get_arch();
        params.prompt = prompt_frame_->positive_prompt();
        params.negative_prompt = prompt_frame_->negative_prompt();
        params.seed = prompt_frame_->get_seed();
        params.steps = prompt_frame_->get_steps();
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
        image_ptr_t original_mask;

        if (image_frame_->get_mode() != img2img_text) {
            params.images = {images_[page_type_image]->view_settings()->at(0)->getImage()->duplicate()};
            original_width = params.images[0]->w();
            original_height = params.images[0]->h();
            params.images[0] = params.images[0]->ensureMultipleOf8();
            params.width = params.images[0]->w();
            params.height = params.images[0]->h();

            if (image_frame_->get_mode() == img2img_inpaint_masked) {
                params.masks = {images_[page_type_image]->view_settings()->at(1)->getImage()->removeAlpha()};
            } else if (image_frame_->get_mode() == img2img_inpaint_not_masked) {
                params.masks = {images_[page_type_image]->view_settings()->at(1)->getImage()->duplicate()};
            }
            
            if (!params.masks.empty()) {
                original_mask = params.masks[0]->duplicate();
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
            if (!frame->enabled()) {
                continue;
            }
            ws::diffusion::control_image_t controlnet;
            controlnet.first.first = frame->getModeStr();
            controlnet.first.second = 1.0; // TODO: get from config
            controlnet.second = frame->getImage();
            if (controlnet.second) {
                params.controlnets.push_back(controlnet);
            }
        }

        auto result = run_diffusion(params);

        if (!result.empty()) {
            if (image_frame_->get_mode() == img2img_inpaint_masked || 
                image_frame_->get_mode() == img2img_inpaint_not_masked) {
                for (auto & img : result) {
                    params.masks = {original_mask->removeAlpha()->blur(4.0)->resizeCanvas(img->w(), img->h())};
                    img->pasteAt(0, 0, params.masks[0].get(), params.images[0].get());
                    img = img->getCrop(0, 0, original_width, original_height);
                    img.swap(img);
                }
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

    image_ptr_t generate_image_(ViewSettings* view_settings) {
        image_ptr_t r;
        DiffusionWindow *window = view_settings ? new DiffusionWindow(view_settings) : new DiffusionWindow();
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


    image_ptr_t generate_image() {
        return generate_image_(NULL);
    }

    image_ptr_t generate_image(ViewSettings* view_settings) {
        return generate_image_(view_settings);
    }

    

} // namespace editorium
