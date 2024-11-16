#include <algorithm>

#include "components/xpm/xpm.h"
#include "websocket/tasks.h"
#include "windows/frames/embeddings_frame.h"

namespace editorium
{

namespace {
    std::string cache_arch;
    std::vector<embedding_t> text_inv_cache;
    std::vector<embedding_t> loras_cache;
}

EmbeddingFrame::EmbeddingFrame(bool lora_embedding, Fl_Group *parent) {
    parent_ = parent;
    lora_embedding_ = lora_embedding;
    img_ = new NonEditableImagePanel(0, 0, 1, 1, lora_embedding ? "LoraEmbeddingImage" : "TextualInversionImage");
    search_ = new Fl_Input(0, 0, 1, 1, lora_embedding ? "Lora:" : "Textual Inversion:");
    embeddings_list_ = new Fl_Select_Browser(0, 0, 1, 1);

    btnNext_.reset(new Button(xpm::image(xpm::img_24x24_forward), [this] {
        goNextConcept();
    }));

    btnUse_.reset(new Button(xpm::image(xpm::img_24x24_green_pin), [this] {
        publish_event(this, lora_embedding_ ? event_prompt_lora_selected : event_prompt_textual_selected, NULL);
    }));

    btnSetImg_.reset(new Button(xpm::image(xpm::img_24x24_folder), [this] {
        selectImage();
    }));

    btnPrior_.reset(new Button(xpm::image(xpm::img_24x24_back), [this] {
        goPreviousConcept();
    }));

    btnNext_->tooltip("Next");
    btnPrior_->tooltip("Previous");
    btnUse_->tooltip("Use this concept");
    btnSetImg_->tooltip("Define image");

    alignComponents();
    search_->align(FL_ALIGN_TOP_LEFT);
    search_->callback(&EmbeddingFrame::widget_cb, this);
    embeddings_list_->callback(&EmbeddingFrame::widget_cb, this);
    embeddings_list_->when(FL_WHEN_ENTER_KEY);
    embeddings_list_->hide();
    search_->when(FL_WHEN_CHANGED);
}

EmbeddingFrame::~EmbeddingFrame() {

}

void EmbeddingFrame::alignComponents() {
    search_->resize(parent_->x() + 5, parent_->y() + 15, parent_->w() - 10, 30);
    img_->resize(parent_->x() + 5, search_->y() + search_->h() + 5, search_->w(), 100);

    embeddings_list_->resize(img_->x(), img_->y(), img_->w(), img_->h());
    
    btnPrior_->position(parent_->x() + 5, img_->y() + img_->h() + 5); 
    btnPrior_->size(img_->w() / 2 - 37, 30);
    
    btnUse_->position(btnPrior_->x() + btnPrior_->w() + 5, btnPrior_->y());
    btnUse_->size(30, 30);

    btnSetImg_->position(btnUse_->x() + btnUse_->w() + 5, btnPrior_->y());
    btnSetImg_->size(30, 30);

    btnNext_->position(btnSetImg_->x() + btnSetImg_->w() + 5, btnPrior_->y());
    btnNext_->size(btnPrior_->w(), btnPrior_->h());
}

void EmbeddingFrame::widget_cb(Fl_Widget* widget, void *cbdata) {
    static_cast<EmbeddingFrame *>(cbdata)->widget_cb(widget);
}

void EmbeddingFrame::widget_cb(Fl_Widget* widget) {
    if (widget == search_) {
        if (in_search_callback_) {
            return;
        }
        in_search_callback_ = true;
        std::string value = search_->value();
        std::vector<std::string> words;
        // take the value and split it by spaces
        std::string word;
        auto lower_it = [](unsigned char c){ return std::tolower(c); };
        value += " ";
        for (auto & c : value) {
            if (c == ' ') {
                if (!word.empty()) {
                    std::transform(word.begin(), word.end(), word.begin(), lower_it);
                    words.push_back(word);
                    word.clear();
                }
            } else {
                word.push_back(c);
            }
        }

        std::vector<std::string> model_list = findModel(words);
        embeddings_list_->clear();
        for (auto & e : model_list) {
            embeddings_list_->add(e.c_str());
        }
        if (model_list.empty()) {
            embeddings_list_->hide();
            img_->show();
            img_->redraw();
        } else {
            embeddings_list_->show();
            img_->hide();
            embeddings_list_->redraw();
        }
        

        in_search_callback_ = false;
    } else if (widget == embeddings_list_) {
        int index = embeddings_list_->value();
        if (index >= 0) {
            in_search_callback_ = true;
            search_->value(embeddings_list_->text(index));
            in_search_callback_ = false;
            embeddings_list_->hide();
            img_->show();
            img_->redraw();
        }
    }
}

void EmbeddingFrame::selectImage() {

}

int EmbeddingFrame::findIndex() {
    if (embeddings_.empty()) {
        return -1;
    }
    std::string current = search_->value();
    for (size_t i = 0; i < embeddings_.size(); i++) {
        if (embeddings_[i].name == current) {
            return (int)i;
        }
    }
    return -1;
}

void EmbeddingFrame::goNextConcept() {
    if (embeddings_.empty()) {
        return;
    }
    int next_index = findIndex() + 1;
    if (next_index < embeddings_.size()) {
        in_search_callback_ = true;
        search_->value(embeddings_[next_index].name.c_str());
        in_search_callback_ = false;
    }
}

void EmbeddingFrame::goPreviousConcept() {
    if (embeddings_.empty()) {
        return;
    }
    int next_index = findIndex() - 1;
    if (next_index >= 0) {
        in_search_callback_ = true;
        search_->value(embeddings_[next_index].name.c_str());
        in_search_callback_ = false;
    } 
}

std::vector<std::string> EmbeddingFrame::findModel(const std::vector<std::string>& words) {
    std::vector<std::string> result;
    if (words.empty()) {
        for (auto & e : embeddings_) {
            result.push_back(e.name);
        }
        std::sort(result.begin(), result.end());
        return result;
    }
    std::string comp;
    auto lower_it = [](unsigned char c){ return std::tolower(c); };
    size_t found_count = 0;
    for (auto & e : embeddings_ ) {
        comp = e.name;
        std::transform(comp.begin(), comp.end(), comp.begin(), lower_it);
        found_count = 0;
        for (auto name_lower : words) {
            if (comp.find(name_lower) != std::string::npos) {
                found_count++;
            }
        }
        if (found_count == words.size()) {
            result.push_back(e.name);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

embedding_t EmbeddingFrame::getSelected() {
    int index = findIndex();
    if (index >= 0) {
        return embeddings_[index];
    }
    embedding_t result;
    return result;
}

void EmbeddingFrame::refresh_models(const std::string& architecture) {
    embeddings_.clear();
    in_search_callback_ = true;
    search_->value("");
    in_search_callback_ = false;
    embeddings_list_->clear();
    if (cache_arch != architecture) {
        cache_arch = architecture;
        text_inv_cache.clear();
        loras_cache.clear();
    }
    if (lora_embedding_) {
        embeddings_ = loras_cache;
    } else {
        embeddings_ = text_inv_cache;
    }
    if (!embeddings_.empty()) {
        embeddings_list_->clear();
        for (auto & e: embeddings_) {
            embeddings_list_->add(e.name.c_str());
        }
        return;
    }
    if (!lora_embedding_) {
        printf("Textual inversion embeddings are disabled for now\n");
        return;
    }
    auto embeddings = ws::models::list_models(architecture, true);
    try{

        for (auto & e: embeddings) {
            embedding_t value;
            value.name = e;
            value.filename = e;
            value.path = e;
            value.kind = "lora";
            if (value.filename.empty() || value.name.empty() || value.path.empty()) {
                continue;
            }
            embeddings_.push_back(value);
            embeddings_list_->add(value.name.c_str());
        }
        if (lora_embedding_) {
            loras_cache = embeddings_;
        } else {
            text_inv_cache = embeddings_;
        }
    } catch(std::exception e) {
        printf("Error refreshing embedding model list %s", e.what());
    }
}

} // namespace editorium
