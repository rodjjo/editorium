#include <cstdio>

#include "components/xpm/xpm.h"
#include "sapiens_ui.h"
/*
SEGMENTATION_CLASSES_NAMES = {
    0: 'background',
    1: 'apparel',
    2: 'face_neck',
    3: 'hair',
    4: 'left_foot',
    5: 'left_hand',
    6: 'left_lower_arm',
    7: 'left_lower_leg',
    8: 'left_shoe',
    9: 'left_sock',
    10: 'left_upper_arm',
    11: 'left_upper_leg',
    12: 'lower_clothing',
    13: 'right_foot',
    14: 'right_hand',
    15: 'right_lower_arm',
    16: 'right_lower_leg',
    17: 'right_shoe',
    18: 'right_sock',
    19: 'right_upper_arm',
    20: 'right_upper_leg',
    21: 'torso',
    22: 'upper_clothing',
    23: 'lower_lip',
    24: 'upper_lip',
    25: 'lower_teeth',
    26: 'upper_teeth',
    27: 'tongue',
}
*/


namespace editorium
{

namespace {
    const char *classes_names[28] = {
        "background",
        "apparel",
        "face_neck",
        "hair",
        "left_foot",
        "left_hand",
        "left_lower_arm",
        "left_lower_leg",
        "left_shoe",
        "left_sock",
        "left_upper_arm",
        "left_upper_leg",
        "lower_clothing",
        "right_foot",
        "right_hand",
        "right_lower_arm",
        "right_lower_leg",
        "right_shoe",
        "right_sock",
        "right_upper_arm",
        "right_upper_leg",
        "torso",
        "upper_clothing",
        "lower_lip",
        "upper_lip",
        "lower_teeth",
        "upper_teeth",
        "tongue",
    };
}

SapiensClassesWindow::SapiensClassesWindow() : Fl_Window(Fl::w() / 2 - 640 / 2, Fl::h() / 2 - 290 / 2, 640, 290, "Sapiens Classes Selector") {
    for (int i = 0; i < 28; i++) {
        check_classes_[i] = new Fl_Check_Button(1, 1, 1, 1, classes_names[i]);
    }
    this->begin();
    btnOk_.reset(new Button(xpm::image(xpm::img_24x24_ok), [this] {
        this->confirmed_ = true;
        this->hide();
    }));
    btnCancel_.reset(new Button(xpm::image(xpm::img_24x24_abort), [this] {
        this->hide();        
    }));

    this->end();

    alignComponents();
    this->set_modal();
}

SapiensClassesWindow::~SapiensClassesWindow() {
}

void SapiensClassesWindow::alignComponents() {
    /*
    align check_classes_ in 7 rows of 4 columns
    */
    int startx = 5;
    int width = (this->w() - 20) / 4;
    int height = 30;
    int x = startx;
    int y = 5;
    for (int i = 0; i < 28; i++) {
        check_classes_[i]->resize(x, y, width, height);
        x += width + 5;
        if (x >= this->w() - 5) {
            x = startx;
            y += height + 5;
        }
    }

    btnOk_->position(this->w() - 215, this->h() - 40);
    btnOk_->size(100, 30);
    btnCancel_->position(btnOk_->x() + btnOk_->w() + 2, btnOk_->y());
    btnCancel_->size(100, 30);
}


std::string SapiensClassesWindow::get_selected_classes() {
    std::string result;
    if (!confirmed_) {
        return result;
    }
    
    for (int i = 0; i < 28; i++) {
        if (check_classes_[i]->value()) {
            
            if (!result.empty()) {
                result += ",";
            }
            result += classes_names[i];
        }
    }
    return result;
}
    
std::string  select_sapien_classes() {
    auto window = new SapiensClassesWindow();
    window->show();
    while (true) {
        if (!window->visible_r()) {
            break;
        }
        Fl::wait();
    }
    
    auto result = window->get_selected_classes();

    Fl::delete_widget(window);
    Fl::do_widget_deletion();

    return result;
}


} // namespace editorium
