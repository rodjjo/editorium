#include "misc/dialogs.h"
#include "windows/image_palette_ui.h"
#include "components/xpm/xpm.h"
#include "images/image_palette.h"

namespace editorium
{

ImagePalleteWindow::ImagePalleteWindow() : Fl_Window(Fl::w() / 2 - 640 / 2, Fl::h() / 2 - 290 / 2, 768, 640, "Image palette - Select an image") {
    this->set_modal();
    this->begin();
    img_ = new ImagePanel(0, 0, 1, 1, "ImagePaletteImagePanel");
    pinned_ = new Fl_Check_Button(0, 0, 1, 1, "Pin this image");

    btnPrior_.reset(new Button(xpm::image(xpm::img_24x24_back), [this] {
        go_prior_image();
    }));

    btnNext_.reset(new Button(xpm::image(xpm::img_24x24_forward), [this] {
        go_next_image();
    }));

    btnOk_.reset(new Button(xpm::image(xpm::img_24x24_ok), [this] {
        this->confirmed_ = true;
        this->hide();
    }));
    btnCancel_.reset(new Button(xpm::image(xpm::img_24x24_abort), [this] {
        this->hide();        
    }));

    this->end();    
    align_components();
    
    selected_index_ = 0;
    show_current_image();

    pinned_->callback(widget_cb, this);
}    

ImagePalleteWindow::~ImagePalleteWindow() {
}

void ImagePalleteWindow::widget_cb(Fl_Widget* widget, void *cbdata) {
    static_cast<ImagePalleteWindow*>(cbdata)->widget_cb(widget);
}

void ImagePalleteWindow::widget_cb(Fl_Widget* widget) {
    if (widget == pinned_) {
        if (ignore_pinned_cb_) {
            return;
        }
        if (selected_index_ < get_image_palette_count()) {
            if (pinned_->value()) {
                pin_image_palette(selected_index_);
            } else {
                unpin_image_palette(selected_index_);
            }
        }
    } 
}

void ImagePalleteWindow::update_title() {
    std::string title = "Image palette - Select an image";
    if (selected_index_ < get_image_palette_count()) {
        title += " (" + std::to_string(selected_index_ + 1) + "/" + std::to_string(get_image_palette_count()) + ")";
    }
    this->copy_label(title.c_str());
}

void ImagePalleteWindow::align_components() {
    // the img_ filling the remaining area of the window with a margin of 5 pixels
    img_->position(5, 5);
    img_->size(this->w() - 10, this->h() - 50);
    
    // the pinned check box at the botton center
    pinned_->position(this->w() / 2 - 50, this->h() - 40);
    pinned_->size(100, 30);

    // butons prior and next side by side at the left bottom
    btnPrior_->position(10, this->h() - 40);
    btnPrior_->size(30, 30);
    btnNext_->position(btnPrior_->x() + btnPrior_->w() + 2, btnPrior_->y());
    btnNext_->size(30, 30);

    btnOk_->position(this->w() - 215, this->h() - 40);
    btnOk_->size(100, 30);
    btnCancel_->position(btnOk_->x() + btnOk_->w() + 2, btnOk_->y());
    btnCancel_->size(100, 30);
}

image_ptr_t ImagePalleteWindow::get_picked_image() {
    image_ptr_t r;
    if (confirmed_ && selected_index_ < get_image_palette_count()) {
        r = get_image_palette(selected_index_);
    }
    return r;
}

void ImagePalleteWindow::show_current_image() {
    if (selected_index_ < get_image_palette_count()) {
        img_->view_settings()->clear_layers();
        img_->view_settings()->add_layer(get_image_palette(selected_index_));
        ignore_pinned_cb_ = true;
        pinned_->value(is_pinned_at_image_palette(selected_index_));
        ignore_pinned_cb_ = false;
    }
    update_title();
}

void ImagePalleteWindow::go_next_image() {
    if (selected_index_ < get_image_palette_count() - 1) {
        selected_index_++;
    }
    if (selected_index_ < get_image_palette_count()) {
        show_current_image();
    }
    
}

void ImagePalleteWindow::go_prior_image() {
    if (selected_index_ > 0) {
        selected_index_--;
    }
    if (selected_index_ < get_image_palette_count()) {
        show_current_image();
    }
}


image_ptr_t pickup_image_from_palette() {
    image_ptr_t r;

    if (get_image_palette_count() < 1) {
        show_error("There is no images in the palette!");
        return r;
    }

    auto window = new ImagePalleteWindow();
    window->show();
    while (true) {
        if (!window->visible_r()) {
            break;
        }
        Fl::wait();
    }
    
    r = window->get_picked_image();

    Fl::delete_widget(window);
    Fl::do_widget_deletion();

    return r;
}

} // namespace editorium
