#include "misc/dialogs.h"
#include "windows/copy_region_ui.h"
#include "components/xpm/xpm.h"


namespace editorium
{

CopyRegionWindow::CopyRegionWindow(image_ptr_t original_img, image_ptr_t new_img) : Fl_Window(Fl::w() / 2 - 640 / 2, Fl::h() / 2 - 290 / 2, 768, 640, "Image palette - Select an image") {
    this->set_modal();

    original_img_ = original_img;
    new_img_ = new_img;
    tabs_ = new Fl_Tabs(0, 0, 1, 1);
    tabs_->begin();

    page_original_ = new Fl_Group(0, 0, 1, 1, "The original image");
    page_original_->begin();
    original_panel_ = new ImagePanel(0, 0, 1, 1, "ImagePaletteImagePanel1");
    original_panel_->view_settings()->add_layer(original_img_);
    original_panel_->view_settings()->set_mask();
    page_original_->end();

    this->begin();
    page_new_img_ = new Fl_Group(0, 0, 1, 1, "The merged image");
    page_new_img_->begin();
    new_panel_ = new ImagePanel(0, 0, 1, 1, "ImagePaletteImagePanel2");
    new_panel_->view_settings()->add_layer(new_img_->duplicate());
    page_new_img_->end();

    this->begin();
    

    btnOk_.reset(new Button(xpm::image(xpm::img_24x24_ok), [this] {
        this->confirmed_ = true;
        this->hide();
    }));
    btnCancel_.reset(new Button(xpm::image(xpm::img_24x24_abort), [this] {
        this->hide();        
    }));
    this->end();    
    align_components();
}    

CopyRegionWindow::~CopyRegionWindow() {
}

void CopyRegionWindow::align_components() {
    // all the component positions are absolute and relative to the window (this)
    // tabs filling the remaining area of the window with a margin of 5 pixels
    tabs_->position(5, 5);
    tabs_->size(this->w() - 10, this->h() - 50);
    // the page_original and page_new_img filling the tabs area with no margins, however 30 pixels from the top
    page_original_->position(tabs_->x(), tabs_->y() + 30);
    page_original_->size(tabs_->w() - 10, tabs_->h() - 35);
    page_new_img_->position(tabs_->x(), tabs_->y() + 30);
    page_new_img_->size(tabs_->w() - 10, tabs_->h() - 35);
    // the original_panel and new_panel filling the respective pages area with no margins
    original_panel_->position(page_original_->x(), page_original_->y());
    original_panel_->size(page_original_->w(), page_original_->h());
    new_panel_->position(page_new_img_->x(), page_new_img_->y());
    new_panel_->size(page_new_img_->w(), page_new_img_->h());
    // the buttons at the bottom right corner
    btnOk_->position(this->w() - 215, this->h() - 40);
    btnOk_->size(100, 30);
    btnCancel_->position(btnOk_->x() + btnOk_->w() + 2, btnOk_->y());
    btnCancel_->size(100, 30);
}

image_ptr_t CopyRegionWindow::get_merged_image() {
    image_ptr_t r;

    return r;
}
  

image_ptr_t copy_image_region(image_ptr_t original, image_ptr_t new_image) {
    image_ptr_t r;

    if (!original || !new_image) {
        show_error("You need two images to merge!");
        return r;
    }

    if (original->w() != new_image->w() || original->h() != new_image->h()) {
        show_error("The images must have the same size!");
        return r;
    }

    auto window = new CopyRegionWindow(original, new_image);
    window->show();
    while (true) {
        if (!window->visible_r()) {
            break;
        }
        Fl::wait();
    }
    
    r = window->get_merged_image();

    Fl::delete_widget(window);
    Fl::do_widget_deletion();

    return r;
}

} // namespace editorium
