#include <FL/Fl.H>
#include <FL/Fl_Native_File_Chooser.H>
#include <FL/Fl_File_Chooser.H>
#include <FL/Fl_Color_Chooser.H>
#include <FL/fl_ask.H>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif 

#include "misc/config.h"
#include "misc/dialogs.h"
#include "misc/profiles.h"

namespace editorium
{
    
namespace {
    const char *kIMAGE_FILES_FILTER = "Image files\t*.{png,bmp,jpeg,webp,gif,jpg}\n";
    const char *kIMAGE_FILES_FILTER_FL = "Image files (*.{png,bmp,jpeg,webp,gif,jpg})";
}

bool path_exists(const char *p) {
#ifdef _WIN32
    return GetFileAttributesA(p) != INVALID_FILE_ATTRIBUTES;
#else 
    return access(p, F_OK) != -1;
#endif
}

bool ask(const char *message) {
    return fl_choice("%s", "No", "Yes", NULL, message) == 1;
}

response_t yes_nc(const char *message) {
    int r = fl_choice("%s", "No", "Cancel", "Yes", message);
    switch (r)
    {
    case 0:
        return r_yes;
        break;
    case 1:
        return r_cancel;
        break;
    default:
        return r_no;
    }
}

void show_error(const char *message) {
    fl_alert("%s", message);
}

void show_info(const char *message) {
    fl_message("%s", message);
}

std::string executeChooser(Fl_File_Chooser *fc) {
    fc->preview(0);
    fc->show();
    fc->position(Fl::w() / 2 - fc->w() / 2, Fl::h() / 2 - fc->h() / 2);
    while (fc->shown()) {
        Fl::wait(0.01);
    }
    if (fc->value()) return fc->value();
    return std::string();
}

std::string choose_image_to_open_fl(const std::string& scope) {
    dialogs_load_profile();
    std::string current_dir = dialogs_profile_get_string({scope.c_str(), "last_open_directory"}, "");

    if (!path_exists(current_dir.c_str())) {
        current_dir = "";
    }
    
    Fl_File_Chooser dialog(current_dir.c_str(), kIMAGE_FILES_FILTER_FL, Fl_File_Chooser::SINGLE, "Open image");
    std::string result = executeChooser(&dialog);
    if (!result.empty()) {
        size_t latest = result.find_last_of("/\\");
        current_dir = result.substr(0, latest);
        dialogs_profile_set_string({scope.c_str(), "last_open_directory"}, current_dir.c_str());
        dialogs_save_profile();
    }
    return result;
}

std::string choose_image_to_save_fl(const std::string& scope) {
    dialogs_load_profile();
    std::string current_dir = dialogs_profile_get_string({scope.c_str(), "last_save_directory"}, "");
    if (!path_exists(current_dir.c_str())) {
        current_dir = "";
    }
    Fl_File_Chooser dialog(current_dir.c_str(), kIMAGE_FILES_FILTER_FL, Fl_File_Chooser::SINGLE | Fl_File_Chooser::CREATE, "Save image");
    std::string result = executeChooser(&dialog);
    if (!result.empty() && path_exists(result.c_str())) {
        if (!ask("Do you want to replace the destination file ?")) {
            result.clear();
        }
    }

    if (!result.empty())  {
        size_t latest = result.find_last_of("/\\");
        current_dir = result.substr(0, latest);
        dialogs_profile_set_string({scope.c_str(), "last_save_directory"}, current_dir.c_str());
        dialogs_save_profile();
    }

    return result;
}

bool pickup_color(const char* title, uint8_t *r, uint8_t *g, uint8_t *b) {
    return fl_color_chooser(title, *r, *g, *b) == 1;
}


} // namespace editorium
