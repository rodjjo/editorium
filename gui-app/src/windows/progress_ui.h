#pragma once

#include <memory>
#include <functional>
#include <FL/Fl_Window.H>
#include <FL/Fl_Progress.H>
#include <FL/Fl_Box.H>

#include "components/button.h"

namespace editorium
{

typedef enum {
    progress_opening_file,
    progress_saving_file,
    progress_list_models,
    progress_downloader,
    progress_generation,
    progress_generation_video,
    progress_upscaler,
    progress_chatbot,
    progress_chatbot_vision,
    progress_correct_colors,
    progress_background,
    progress_preprocessor,
    progress_segmentation
} progress_type;

typedef std::function<bool()> checker_cb_t;

class ProgressWindow {
  public:
    ProgressWindow(progress_type ptype);
    virtual ~ProgressWindow();
  private:
    static void update(void *cbdata);
    void update();
    void set_title(const char *title);
    void set_progress(size_t value, size_t max);
    void set_text(const char *text);

  private:
    Fl_Window *window_;
    Fl_Progress *progress_ = NULL;
    Fl_Box *text_ = NULL;
    std::unique_ptr<Button> btnCancel_;
    size_t version_ = 0;
    progress_type ptype_;
};

void show_progress_window(progress_type ptype, checker_cb_t cancel_cb);
void show_progress_window();
void enable_progress_window(progress_type ptype);
void hide_progress_window();
bool should_cancel();
void set_progress_title(const std::string& title);
void set_progress_text(const std::string& text);
void set_progress(size_t value, size_t max);


    
} // namespace editorium
