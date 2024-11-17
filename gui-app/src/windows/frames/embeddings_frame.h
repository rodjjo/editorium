#pragma once

#include <string>
#include <vector>

#include <FL/Fl_Group.H>
#include <FL/Fl_Select_Browser.H>
#include <FL/Fl_Input.H>


#include "messagebus/messagebus.h"
#include "components/button.h"
#include "components/image_panel.h"

namespace editorium
{

typedef struct {
  std::string name;
  std::string kind;
  std::string filename;
  std::string path;
} embedding_t;

class EmbeddingFrame {
  public:
    EmbeddingFrame(bool lora_embedding, Fl_Group *parent);
    virtual ~EmbeddingFrame();
    void refresh_models(const std::string& architecture);
    void alignComponents();
    embedding_t getSelected();
    bool contains(const std::string& name);
    
  private:
    void goNextConcept();
    void goPreviousConcept();
    void selectImage();
    static void widget_cb(Fl_Widget* widget, void *cbdata);
    void widget_cb(Fl_Widget* widget);
    std::vector<std::string> findModel(const std::vector<std::string>& words);
    int findIndex();

  private:
    bool in_search_callback_ = false;
    std::vector<embedding_t> embeddings_;
    bool lora_embedding_ = false;
    Fl_Input *search_;
    Fl_Select_Browser *embeddings_list_;
    Fl_Group *parent_;
    ImagePanel *img_;
    std::unique_ptr<Button> btnNext_;
    std::unique_ptr<Button> btnUse_;
    std::unique_ptr<Button> btnSetImg_;
    std::unique_ptr<Button> btnPrior_;

};

} // namespace editorium
