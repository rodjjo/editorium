#include <FL/fl_ask.H>
#include "components/xpm/xpm.h"
#include "websocket/tasks.h"
#include "misc/dialogs.h"
#include "windows/sapiens_ui.h"

#include "windows/copy_region_ui.h"


namespace editorium
{

namespace {


    const uint8_t brushes_sizes[] = {
        4, 8, 16, 32, 64, 128
    };

    uint8_t brush_size_count() {
        return sizeof(brushes_sizes) / sizeof(brushes_sizes[0]);
    }

}

CopyRegionWindow::CopyRegionWindow(image_ptr_t original_img, image_ptr_t new_img) : Fl_Window(Fl::w() / 2 - 640 / 2, Fl::h() / 2 - 290 / 2, 768, 640, "Image palette - Select an image") {
    this->set_modal();

    original_img_ = original_img;
    new_img_ = new_img;
    
    this->begin();
    tabs_ = new Fl_Tabs(0, 0, 1, 1);
    tabs_->begin();

    page_original_ = new Fl_Group(0, 0, 1, 1, "The original image");
    page_original_->begin();
    original_panel_ = new MaskEditableImagePanel(0, 0, 1, 1, "ImagePaletteImagePanel1");
    original_panel_->view_settings()->add_layer(new_img_);
    original_panel_->view_settings()->set_mask();
    page_original_->end();
    
    tabs_->begin();
    page_new_img_ = new Fl_Group(0, 0, 1, 1, "The merged image");
    page_new_img_->begin();
    new_panel_ = new NonEditableImagePanel(0, 0, 1, 1, "ImagePaletteImagePanel2");
    new_panel_->view_settings()->add_layer(original_img_->duplicate());
    page_new_img_->end();

    this->begin();
    
    choice_brush_size_ = new Fl_Choice(1, 1, 1, 1, "Brush size");
    
    btn_seg_dino_.reset(new Button(xpm::image(xpm::img_24x24_alien), [this] {
        this->peform_dino_image_segmentation();
    }));

    btn_seg_sapiens_.reset(new Button(xpm::image(xpm::img_24x24_female), [this] {
        this->peform_sapiens_image_segmentation();
    }));

    btnOk_.reset(new Button(xpm::image(xpm::img_24x24_ok), [this] {
        this->confirmed_ = true;
        this->hide();
    }));
    btnCancel_.reset(new Button(xpm::image(xpm::img_24x24_abort), [this] {
        this->hide();        
    }));
    this->end();    

    btn_seg_dino_->tooltip("Create mask using Grounding Dino segmentation");
    btn_seg_sapiens_->tooltip("Create mask using Sapiens segmentation (from facebook)");


    char buffer[64] = "";
    for (int i = 0; i < brush_size_count(); i++) {
        sprintf(buffer, "%d Pixels", brushes_sizes[i]);
        choice_brush_size_->add(buffer);
    }
    choice_brush_size_->align(FL_ALIGN_LEFT);
    choice_brush_size_->value(2);

    align_components();

    choice_brush_size_->callback(cb_widget, this);
    tabs_->callback(cb_widget, this);
    tabs_->when(FL_WHEN_CHANGED);
}    

CopyRegionWindow::~CopyRegionWindow() {
}

void CopyRegionWindow::cb_widget(Fl_Widget *widget, void *data) {
    auto self = static_cast<CopyRegionWindow*>(data);
    self->cb_widget(widget);
}

void CopyRegionWindow::cb_widget(Fl_Widget *widget) {
    if (widget == tabs_) {
       merge_images();
    } else if (widget == choice_brush_size_) {
        auto brush_size = brushes_sizes[choice_brush_size_->value()];
        original_panel_->view_settings()->brush_size(brush_size);
    }
}

void CopyRegionWindow::merge_images() {
    if (original_panel_->view_settings()->layer_count() < 2) {
        return;
    }
    if (original_panel_->view_settings()->at(1)->version() != mask_version_) {
        mask_version_ = original_panel_->view_settings()->at(1)->version();
        auto merged = original_img_->duplicate();
        auto mask = original_panel_->view_settings()->at(1)->getImage();
        merged->pasteAt(0, 0, mask, new_img_.get());
        new_panel_->view_settings()->clear_layers();
        new_panel_->view_settings()->add_layer(merged);
    }
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
    // the choice_brush_size at the bottom left corner
    choice_brush_size_->position(75, this->h() - 40);
    choice_brush_size_->size(150, 30);

    // the buttons btn_seg_dino_ and btn_seg_sapiens_ at the bottom left corner after choice_brush_size_
    btn_seg_dino_->size(30, 30);
    btn_seg_sapiens_->size(30, 30);
    btn_seg_dino_->position(choice_brush_size_->x() + choice_brush_size_->w() + 5, choice_brush_size_->y());
    btn_seg_sapiens_->position(btn_seg_dino_->x() + btn_seg_dino_->w() + 5, btn_seg_dino_->y());
    

    // the buttons at the bottom right corner
    btnOk_->position(this->w() - 215, this->h() - 40);
    btnOk_->size(100, 30);
    btnCancel_->position(btnOk_->x() + btnOk_->w() + 2, btnOk_->y());
    btnCancel_->size(100, 30);
}

void CopyRegionWindow::peform_dino_image_segmentation() {
    auto classes = fl_input("Enter the classes to segment (comma separated)", "");
    if (!classes) {
        return;
    }
    auto r = ws::diffusion::run_seg_ground_dino(classes, {original_img_});
    if (r.empty()) {
        return;
    }
    if (original_panel_->view_settings()->layer_count() < 2) {
        original_panel_->view_settings()->add_layer(r[0]->dilate(3));
    } else {
        original_panel_->view_settings()->at(1)->replace_image(r[0]->dilate(3));
    }
}

void CopyRegionWindow::peform_sapiens_image_segmentation() {
    std::string classes = select_sapien_classes();
    if (classes.empty()) {
        return;
    }
    auto r = ws::diffusion::run_seg_sapiens(classes, {original_img_});
    if (r.empty()) {
        return;
    }
    if (original_panel_->view_settings()->layer_count() < 2) {
        original_panel_->view_settings()->add_layer(r[0]->dilate(3));
    } else {
        original_panel_->view_settings()->at(1)->replace_image(r[0]->dilate(3));
    }
}


image_ptr_t CopyRegionWindow::get_merged_image() {
    image_ptr_t r;
    if (confirmed_) {
        merge_images();
        r = new_panel_->view_settings()->at(0)->getImage()->duplicate();
    }
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
