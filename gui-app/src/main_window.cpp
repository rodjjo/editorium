#include <map>
#include <FL/Fl.H>
#include <FL/fl_ask.H>
#include "windows/progress_ui.h"
#include "windows/settings_ui.h"
#include "windows/diffusion_ui.h"
#include "windows/upscaler_ui.h"
#include "windows/size_ui.h"
#include "misc/dialogs.h"
#include "misc/config.h"
#include "misc/utils.h"
#include "websocket/tasks.h"

#include "main_window.h"


namespace editorium
{
    namespace
    {
        bool stopped = false;
        const std::list<event_id_t> main_window_events = {
            event_main_menu_clicked,
            event_main_menu_file_new_art,
            event_main_menu_file_open,
            event_main_menu_file_save,
            event_main_menu_file_open_layer,
            event_main_menu_file_close,
            event_main_menu_edit_settings,
            event_main_menu_layers_duplicate,
            event_main_menu_layers_remove_selected,
            event_main_menu_layers_merge_all,
            event_main_menu_layers_remove_background,
            event_main_menu_layers_remove_background_sapiens,
            event_main_menu_layers_remove_background_gdino,
            event_main_menu_layers_flip_horizontal,
            event_main_menu_layers_flip_vertical,
            event_main_menu_layers_rotate_clock,
            event_main_menu_layers_reset_zoom,
            event_main_menu_layers_reset_scroll,
            event_main_menu_enhance_upscaler,
            event_main_menu_enhance_resize,
            event_main_menu_enhance_correct_colors,
            event_main_menu_selection_generate, 
            event_main_menu_layers_from_selection,
            event_main_menu_layers_from_generated,
            event_main_menu_resizeSelection_0,
            event_main_menu_resizeSelection_256,
            event_main_menu_resizeSelection_512,
            event_main_menu_resizeSelection_768,
            event_main_menu_resizeSelection_1024,
            event_main_menu_resizeSelection_2048,
            event_main_menu_resizeSelection_all,           
            event_layer_count_changed,
            event_layer_selected,
            event_layer_after_draw,
            event_main_menu_exit
        };
    } // namespace

    MainWindow::MainWindow() : Fl_Menu_Window(640, 480, "Editorium"), SubscriberThis(main_window_events)
    {
        auto wnd_ = this;

        { // menu
            menuPanel_ = new Fl_Group(0, 20, this->w(), 20);
            menuPanel_->end();
            menuPanel_->box(FL_BORDER_BOX);
            menuPanel_->begin();
            menu_ = new MainMenu(this->w(), 20);
            menuPanel_->end();

            menu_->addItem(event_main_menu_file_new_art, "", "File/New Art", "^n", 0, xpm::img_24x24_new);
            menu_->addItem(event_main_menu_file_open, "", "File/Open", "^o", 0, xpm::img_24x24_open);
            menu_->addItem(event_main_menu_file_save, "", "File/Save", "^s", 0, xpm::img_24x24_flash_drive);
            menu_->addItem(event_main_menu_file_open_layer, "", "File/Open as Layer", "^l", 0, xpm::img_24x24_open_layer);
            menu_->addItem(event_main_menu_file_close, "", "File/Close", "^x", 0, xpm::img_24x24_close);
            menu_->addItem(event_main_menu_exit, "", "File/Exit", "", 0, xpm::img_24x24_exit);
            menu_->addItem(event_main_menu_edit_settings, "", "Edit/Settings", "", 0, xpm::img_24x24_settings);
            menu_->addItem(event_main_menu_layers_duplicate, "", "Layers/Duplicate", "^d", 0, xpm::img_24x24_copy);
            menu_->addItem(event_main_menu_layers_from_selection, "", "Layers/From selection", "", 0, xpm::no_image);
            menu_->addItem(event_main_menu_layers_from_generated, "", "Layers/From generation", "", 0, xpm::no_image);
            menu_->addItem(event_main_menu_layers_remove_selected, "", "Layers/Remove", "", 0, xpm::img_24x24_remove);
            menu_->addItem(event_main_menu_layers_merge_all, "", "Layers/Merge", "^m", 0, xpm::img_24x24_load);
            menu_->addItem(event_main_menu_layers_reset_zoom, "", "Layers/Reset Zoom", "", 0, xpm::no_image);
            menu_->addItem(event_main_menu_layers_reset_scroll, "", "Layers/Reset Scroll", "", 0, xpm::no_image);
            menu_->addItem(event_main_menu_layers_remove_background, "", "Layers/Background/Remove", "", 0, xpm::img_24x24_picture);
            menu_->addItem(event_main_menu_layers_remove_background_sapiens, "", "Layers/Background/Extract human", "", 0, xpm::img_24x24_picture);
            menu_->addItem(event_main_menu_layers_remove_background_gdino, "", "Layers/Background/Extract object", "", 0, xpm::img_24x24_picture);
            menu_->addItem(event_main_menu_layers_flip_horizontal, "", "Layers/Flip/Horizontal", "", 0, xpm::img_24x24_left_right);
            menu_->addItem(event_main_menu_layers_flip_vertical, "", "Layers/Flip/Vertical", "", 0, xpm::img_24x24_up_down);
            menu_->addItem(event_main_menu_layers_rotate_clock, "", "Layers/Flip/Rotate", "", 0, xpm::img_24x24_redo);
            menu_->addItem(event_main_menu_enhance_upscaler, "", "Enhancements/Upscaler", "", 0, xpm::img_24x24_zoom);
            menu_->addItem(event_main_menu_enhance_resize, "", "Enhancements/Resize Image", "^r", 0, xpm::img_24x24_text_preview);
            menu_->addItem(event_main_menu_enhance_correct_colors, "", "Enhancements/Correct Colors", "", 0, xpm::img_24x24_text_preview);
            menu_->addItem(event_main_menu_selection_generate, "", "Selection/Generate Image", "^i", 0, xpm::img_24x24_bee);
            menu_->addItem(event_main_menu_resizeSelection_0, "", "Selection/Expand/Custom", "^e");
            menu_->addItem(event_main_menu_resizeSelection_256, "", "Selection/Expand/256x256", "^0");
            menu_->addItem(event_main_menu_resizeSelection_512, "", "Selection/Expand/512x512", "^1");
            menu_->addItem(event_main_menu_resizeSelection_768, "", "Selection/Expand/768x768", "^2");
            menu_->addItem(event_main_menu_resizeSelection_1024, "", "Selection/Expand/1024x1024", "^3");
            menu_->addItem(event_main_menu_resizeSelection_2048, "", "Selection/Expand/2048x2048", "^4");
            menu_->addItem(event_main_menu_resizeSelection_all, "", "Selection/Expand/Select All", "^a");
        } // menu

        { // image panels
            wnd_->begin();

            image_ = new ImagePanel(0, 0, 1, 1, "MainWindowImagePanel");
            layers_ = new Fl_Select_Browser(0, 0, 1, 1);
            layers_->callback(layer_cb, this);
            removeLayer_.reset(new Button(xpm::image(xpm::img_24x24_remove), [this] {
                remove_selected_layer();
            }));
            removeAllLayers_.reset(new Button(xpm::image(xpm::img_24x24_erase), [this] {
                clear_layers();
            }));
            removeLayer_->tooltip("Remove selected layer");
            removeAllLayers_->tooltip("Remove all the layers");
            removeLayer_->size(28, 28);
            removeAllLayers_->size(28, 28);
        } 
        
        { // status bar
            wnd_->begin();

            bottomPanel_ = new Fl_Group(0, 0, 1, 1);
            bottomPanel_->begin();
            lblImageSize_ = new Fl_Box(0, 0, 1, 1);
            lblZoomSize_ = new Fl_Box(0, 0, 1, 1);
            lblLayerSize_ = new Fl_Box(0, 0, 1, 1);
            lblSelectionSize_ = new Fl_Box(0, 0, 1, 1);
            bottomPanel_->end();
        }

        wnd_->end();
        wnd_->position(Fl::w() / 2 - wnd_->w() / 2, Fl::h() / 2 - wnd_->h() / 2);
        wnd_->size_range(860, 480);
        wnd_->show();
        alignComponents();
    }

    MainWindow::~MainWindow(){
    };

    int MainWindow::dfe_run()
    {
        puts("Starting user interface...");
        Fl::scheme("gtk+");
        MainWindow *wnd = new MainWindow();
        
        get_config();


        while (!stopped)
        {
            Fl::wait(0.33);
            if (!wnd->shown())
            {
                break;
            }
        }
        return 0;
    }
    void MainWindow::layer_cb(Fl_Widget* widget, void *cbdata) {
        static_cast<MainWindow *>(cbdata)->layer_cb(widget);
    }

    void MainWindow::layer_cb(Fl_Widget* widget) {
        if (widget == layers_) {
            if (in_layer_callback_) {
                return;
            }
            in_layer_callback_ = true;
            int idx = layers_->value();
            if (idx > 0)  {
                image_->view_settings()->select(idx - 1);
            } 
            layers_->deselect();
            layers_->select(idx);
            in_layer_callback_ = false;
        }
    }

    void MainWindow::dfe_stop()
    {
        dfe_hideProgress();
        stopped = true;
    }

    void MainWindow::dfe_close()
    {
        stopped = true;
    }

    void MainWindow::dfe_showProgress()
    {
        show_progress_window();
    }

    void MainWindow::dfe_hideProgress()
    {
        hide_progress_window();
    }

    void MainWindow::dfe_show_error(const char *message) {
        fl_alert("%s", message);
    }

    void MainWindow::alignComponents() {
        menuPanel_->position(0, 0);
        int w = this->w();
        int h = this->h();
        menuPanel_->size(w, menu_->h());
        bottomPanel_->size(w, menu_->h());
        bottomPanel_->position(0, h - bottomPanel_->h());
        menu_->position(0, 0);
        menu_->size(w, menuPanel_->h());
        if (image_->view_settings()->layer_count() > 0) {
            removeAllLayers_->position(w - removeAllLayers_->w() - 1, menuPanel_->h() + 1);
            removeLayer_->position(removeAllLayers_->x() - removeLayer_->w() - 2,  removeAllLayers_->y());
            layers_->size(100, h - (menuPanel_->h() + removeAllLayers_->h() + bottomPanel_->h()) - 2);
            layers_->position(w - 1 - layers_->w(), removeAllLayers_->h() + removeAllLayers_->y() + 1);
            image_->size(w - 5 - layers_->w(), h - menuPanel_->h() - bottomPanel_->h() - 2);
            layers_->show();
            removeLayer_->show();
            removeAllLayers_->show();
        } else {
            layers_->hide();
            removeLayer_->hide();
            removeAllLayers_->hide();
            image_->size(w - 2, h - menuPanel_->h() - bottomPanel_->h() - 2);
        }
        image_->position(1, menuPanel_->h() + 1);

        lblImageSize_->size(100, bottomPanel_->h() - 2);
        lblZoomSize_->size(170, bottomPanel_->h() - 2);
        lblLayerSize_->size(300, bottomPanel_->h() - 2);
        lblSelectionSize_->size(300, bottomPanel_->h() - 2);

        lblImageSize_->position(bottomPanel_->x() + 1,  bottomPanel_->y() + 1);
        lblZoomSize_->position(lblImageSize_->x() + 2 + lblImageSize_->w(),  lblImageSize_->y());
        lblLayerSize_->position(lblZoomSize_->x() + 2 + lblZoomSize_->w(),  lblImageSize_->y());
        lblSelectionSize_->position(lblLayerSize_->x() + 2 + lblLayerSize_->w(),  lblImageSize_->y());
    }

    void MainWindow::resize(int x, int y, int w, int h)
    {
        Fl_Menu_Window::resize(x, y, w, h);
        alignComponents();
    }

    int MainWindow::handle(int event)
    {
        switch (event)
        {
        case FL_KEYUP:
        {
            if (Fl::event_key() == FL_Escape)
            {
                return 1;
            }
            if (Fl::event_key() == FL_Delete) {
                remove_selected_layer();
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

        return Fl_Menu_Window::handle(event);
    }

    void MainWindow::dfe_handle_event(void *sender, event_id_t id, void *data)
    {
        switch (id)
        {
        case event_main_menu_clicked:
            break;
        case event_main_menu_file_new_art:
            create_image(false);
            break;
        case event_main_menu_selection_generate:
            create_image(true);
            break;
        case event_main_menu_layers_from_selection:
            convert_selection_into_layer();
            break;
        case event_main_menu_layers_from_generated:
            layer_generate_in_selection();
            break;
        case event_main_menu_resizeSelection_0:
            resizeSelection(0);
            break;
        case event_main_menu_resizeSelection_256:
            resizeSelection(256);
            break;
        case event_main_menu_resizeSelection_512:
            resizeSelection(512);
            break;
        case event_main_menu_resizeSelection_768:
            resizeSelection(768);
            break;
        case event_main_menu_resizeSelection_1024:
            resizeSelection(1024);
            break;
        case event_main_menu_resizeSelection_2048:
            resizeSelection(2048);
            break;
        case event_main_menu_resizeSelection_all:
            resizeSelection(-1);
            break;
        case event_main_menu_file_open:
            choose_file_and_open(true);
            break;
        case event_main_menu_file_save:
            choose_file_and_save();
            break;
        case event_main_menu_file_open_layer:
            choose_file_and_open(false);
            break;
        case event_main_menu_file_close:
            clear_layers();
            break;
        case event_layer_count_changed:
            if (sender == image_) {
                update_layer_list();
            }
            break;
        case event_layer_selected:
            if (sender == image_ && !in_layer_callback_) {
                layers_->value(image_->view_settings()->selected_layer_index() + 1);
            }
            break;
        case event_main_menu_edit_settings:
            edit_settings();
            break;
        case event_main_menu_layers_duplicate:
            image_->view_settings()->duplicate_selected();
            break;
        case event_main_menu_layers_remove_background:
            image_->view_settings()->remove_background_selected(remove_using_default);
            break;
        case event_main_menu_layers_remove_background_sapiens:
            image_->view_settings()->remove_background_selected(remove_using_sapiens);
            break;
        case event_main_menu_layers_remove_background_gdino:
            image_->view_settings()->remove_background_selected(remove_using_gdino);
            break;
        case event_main_menu_layers_remove_selected:
            remove_selected_layer();
            break;
        case event_main_menu_layers_merge_all:
            merge_all_layers();
            break;
        case event_main_menu_layers_reset_zoom:
            image_->view_settings()->setZoom(100);
            break;
        case event_main_menu_layers_reset_scroll:
            image_->clear_scroll();
            break;
        case event_main_menu_layers_flip_horizontal:
            image_->view_settings()->flip_horizoltal_selected();
            break;
        case event_main_menu_layers_flip_vertical:
            image_->view_settings()->flip_vertical_selected();
            break;
        case event_main_menu_layers_rotate_clock:
            image_->view_settings()->rotate_selected();
            break;
        case event_main_menu_enhance_upscaler:
            upscale_current_image();
            break;
        case event_main_menu_enhance_resize:
            resize_image();
            break;
        case event_main_menu_enhance_correct_colors:
            correct_colors();
            break;
        case event_main_menu_exit:
            this->hide();
            break;
        case event_layer_after_draw:
            if (sender == image_) {
                Fl::remove_timeout(MainWindow::update_status, this);
                Fl::add_timeout(0.033, MainWindow::update_status, this);
            }
            break;
        }
    }

    void MainWindow::update_status(void *cbdata) {
        static_cast<MainWindow *>(cbdata)->update_status();
    }

    void MainWindow::update_status() {
        char buffer[256] = "";
        
        if ((int)image_->view_settings()->layer_count()) {
            int x = 0, y = 0, w = 0, h = 0;
            image_->view_settings()->get_image_area(&x, &y, &w, &h);
            sprintf(buffer, " Size: %d x %d ", w, h);
        } else {
            sprintf(buffer, " Size: 0 x 0 ");
        }
        lblImageSize_->copy_label(buffer);
        
        sprintf(buffer, " Zoom: %d %% ", image_->view_settings()->getZoom());
        lblZoomSize_->copy_label(buffer);

        if (image_->view_settings()->selected_layer()) {
            auto l = image_->view_settings()->selected_layer();
            sprintf(buffer, " Layer: x: %d y: %d w: %d x h: %d count: %d", l->x(), l->y(), l->w(), l->h(), (int)image_->view_settings()->layer_count());
        } else {
            sprintf(buffer, " No layer selected (count: %d) ", (int)image_->view_settings()->layer_count());
        }
        lblLayerSize_->copy_label(buffer);
        int sx, sy, sw, sh;
        if (image_->view_settings()->get_selected_area(&sx, &sy, &sw, &sh)) {
            sprintf(buffer, "Selection: x: %d, y: %d, w: %d, h: %d", sx, sy, sw, sh);
        } else {
            sprintf(buffer, "No Selection");
        }
        lblSelectionSize_->copy_label(buffer);
    }

    void MainWindow::choose_file_and_open(bool clear_layers) {
        std::string result = choose_image_to_open_fl("main_window_picture");
        if (!result.empty()) {
            auto dir = filepath_dir(result);
            if (clear_layers) {
                image_->view_settings()->clear_layers();
            }
            printf("[MainWindow] Opening file: %s\n", result.c_str());
            image_->view_settings()->add_layer(result.c_str());
        }
    }

    void MainWindow::choose_file_and_save() {
        std::string result = choose_image_to_save_fl("main_window_picture");
        if (!result.empty()) {
            printf("[MainWindow] Saving file: %s\n", result.c_str());
            auto img = image_->view_settings()->merge_layers_to_image();
            if (img) {
                ws::filesystem::save_image(result, img, result.find(".png") != std::string::npos);
            }
        }
    }

    void MainWindow::update_layer_list() {
        alignComponents();
        layers_->clear();
        for (size_t i = 0; i < image_->view_settings()->layer_count(); i++) {
            layers_->add(image_->view_settings()->at(i)->name());
        }
    }

    void MainWindow::remove_selected_layer() {
        if (ask("Do you want to remove the selected layer ?")) {
            image_->view_settings()->remove_layer(layers_->value() - 1);
        }
    }

    void MainWindow::merge_all_layers() {
        if (ask("Do you want to merge all the layers ?")) {
            auto img = image_->view_settings()->merge_layers_to_image();
            if (img) {
                image_->view_settings()->clear_layers();
                image_->view_settings()->add_layer(img);
            }
        }
    }

    void MainWindow::clear_layers() {
        const char *message = image_->view_settings()->layer_count() > 1 ? "Do you want to close all the layers ?" : "Do you want to close the image ?";
        if (ask(message)) {
            image_->view_settings()->clear_layers();
        }
    }

    void MainWindow::convert_selection_into_layer() {
        if (image_->view_settings()->has_selected_area()) {
            int sx, sy, unused;
            image_->view_settings()->get_selected_area(&sx, &sy, &unused, &unused);
            image_->view_settings()->add_layer(image_->view_settings()->get_selected_image());
            image_->view_settings()->clear_selected_area();
            image_->view_settings()->at(image_->view_settings()->layer_count() - 1)->x(sx);
            image_->view_settings()->at(image_->view_settings()->layer_count() - 1)->y(sy);
        } else {
            fl_alert("No selection to create a layer");
        }
    }

    void MainWindow::upscale_current_image() {
        if (image_->view_settings()->layer_count() < 1) {
            show_error("Open an image first!");
            return;
        } else if (image_->view_settings()->layer_count() > 1) {
            show_error("Upscaling is only available for single layer images!\nMerge all the layers first!");
            return;
        }
        image_->view_settings()->clear_selected_area();
        auto img = image_->view_settings()->merge_layers_to_image();
        float scale = 2.0;
        float weight = 1.0;
        bool restore_bg = true;
        if (get_gfpgan_upscaler_params(scale, weight, restore_bg)) {
            auto img_list = ws::upscalers::upscale_gfpgan(scale, weight, restore_bg, {img});
            if (img_list.size() > 0) {
                image_->view_settings()->clear_layers();
                image_->view_settings()->add_layer(img_list[0]);
            }
        }
    }

    void MainWindow::resize_image() {
        if (image_->view_settings()->layer_count() < 1) {
            show_error("Open an image first!");
            return;
        } else if (image_->view_settings()->layer_count() > 1) {
            show_error("Resizing is only available for single layer images!\nMerge all the layers first!");
            return;
        }
        int w = 0, h = 0, unused = 0;
        image_->view_settings()->get_image_area(&unused, &unused, &w, &h);
        if (getSizeFromDialog("Resize the image", &w, &h)) {
            image_->view_settings()->clear_selected_area();
            auto img = image_->view_settings()->merge_layers_to_image();
            if (img) {
                auto resized = img->resizeImage(w, h);
                image_->view_settings()->clear_layers();
                image_->view_settings()->add_layer(resized);
            }
        }
    }

    void MainWindow::correct_colors() {
        if (image_->view_settings()->layer_count() < 1) {
            show_error("Open an image first!");
            return;
        } else if (image_->view_settings()->layer_count() > 1) {
            show_error("Correcting colors is only available for single layer images!\nMerge all the layers first!");
            return;
        } else if (!image_->view_settings()->has_selected_area()) {
            show_error("The color correction use the selected area as reference.\nIt corrects the colors outside the selected area!");
            return;
        }
        auto img = image_->view_settings()->merge_layers_to_image();
        auto selected_img = image_->view_settings()->get_selected_image();
        auto corrected = ws::upscalers::correct_colors({img}, {selected_img});
        if (corrected.size() > 0) {
            image_->view_settings()->clear_layers();
            image_->view_settings()->add_layer(corrected[0]);
            image_->view_settings()->clear_selected_area();
        }
    }

    void MainWindow::layer_generate_in_selection() {
        if (image_->view_settings()->has_selected_area()) {
            int sx, sy, unused;
            image_->view_settings()->get_selected_area(&sx, &sy, &unused, &unused);
            auto img = generate_image(true, image_->view_settings());
            if (img) {
                image_->view_settings()->add_layer(img);
                image_->view_settings()->clear_selected_area();
                image_->view_settings()->at(image_->view_settings()->layer_count() - 1)->x(sx);
                image_->view_settings()->at(image_->view_settings()->layer_count() - 1)->y(sy);
            }
        } else {
            fl_alert("No selection to generate a layer");
        }
    }

    void MainWindow::create_image(bool selection) {
        if (selection) {
            if (image_->view_settings()->layer_count() < 1) {
                show_error("Open an image first!");
                return;
            }
            if (!image_->view_settings()->has_selected_area()) {
                show_error("There is no selection!\nSelect a area in the image first!");
                return;
            }
        }
        auto img = selection ? generate_image(true, image_->view_settings()) : generate_image(true);
        if (img) {
            image_->view_settings()->fuse_image(img);
        } 
    }

      void MainWindow::resizeSelection(int width) {
        int x1 = 0, x2 = 0, y1 = 0, y2 = 0;
        image_->view_settings()->get_selected_area(&x1, &y1, &x2, &y2);
        x2 += x1;
        y2 += y1;
        if (x1 == x2 && y1 == y2 && width != -1) {
            show_error("No selection");
            return;
        }

        int w = x2 - x1;
        int h = y2 - y1;
        if (width > 0) {
            int sx, sy, sw, sh;
            image_->view_settings()->get_image_area(&sx, &sy, &sw, &sh);
            if (width > sw) {
                width = sw;
            }
            if (width > sh) {
                width = sh;
            }
            printf("Resizing selection to %d x %d\n", width, width);
            int diff_w = (width - w) / 2; 
            int diff_h = (width - h) / 2;
            x1 -= diff_w;
            y1 -= diff_h;
            image_->view_settings()->set_selected_area(x1, y1, width, width);
            image_->view_settings()->get_selected_area(&x1, &y1, &x2, &y2);
            x2 += x1;
            y2 += y1;

            if (x1 < 0) {
                x2 += (-x1);
                x1 = 0;
            }
            if (y1 < 0) {
                y2 += (-y1);
                y1 = 0;
            }

            if (x2 > sw) {
                auto diff = x2 - sw;
                x2 -= diff;
                x1 -= diff;
            }
            if (y2 > sh) {
                auto diff = y2 - sh;
                y2 -= diff;
                y1 -= diff;
            }
            printf("Moving selection to %d x %d\n", x1, y1);
            printf("Resizing selection to %d x %d\n", x2 - x1, y2 - y1);
            image_->view_settings()->set_selected_area(x1, y1, x2 - x1, y2 - y1);
        } else if (width == -1) {
            int x = 0, y = 0;
            int w = 0, h = 0;
            image_->view_settings()->get_image_area(&x, &y, &w, &h);
            image_->view_settings()->set_selected_area(0, 0, w, h);
        } else if (getSizeFromDialog("Resize the selection area", &w, &h)) {
            image_->view_settings()->set_selected_area(x1, y1, w, h);
        } 
      }

} // namespace editorium
