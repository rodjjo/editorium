#pragma once

#include <list>
#include <functional>

namespace editorium
{

/*
    This message buss is not thread safe.
    Should only be called from the gui thread
*/

typedef enum {
    event_none = 0,
    event_main_menu_clicked,
    event_main_menu_file_new_art,
    event_main_menu_file_open,
    event_main_menu_file_save,
    event_main_menu_file_open_layer,
    event_main_menu_file_close,
    event_main_menu_edit_settings,
    event_main_menu_layers_duplicate,
    event_main_menu_layers_from_selection,
    event_main_menu_layers_from_generated,
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
    event_main_menu_selection_generate,
    event_main_menu_resizeSelection_0,
    event_main_menu_resizeSelection_256,
    event_main_menu_resizeSelection_512,
    event_main_menu_resizeSelection_768,
    event_main_menu_resizeSelection_1024,
    event_main_menu_resizeSelection_2048,
    event_main_menu_resizeSelection_all,
    event_main_menu_exit,
    event_layer_count_changed,
    event_layer_selected,
    event_layer_after_draw,
    event_generator_next_image,
    event_generator_previous_image,
    event_generator_accept_image,
    event_generator_accept_partial_image,
    event_generator_save_current_image,
    event_image_frame_new_mask,
    event_image_frame_open_mask,
    event_image_frame_seg_gdino,
    event_image_frame_seg_sapiens,
    event_image_frame_mode_selected,
    event_prompt_lora_selected,
    event_prompt_textual_selected,
    event_prompt_architecture_selected,
} event_id_t;

typedef std::function<void(void *sender, event_id_t event, void *data)> event_handler_t;

class Subscriber {
    event_handler_t handler_;
public:
    Subscriber(const std::list<event_id_t>& events, event_handler_t handler);
    virtual ~Subscriber();
    void operator() (void *sender, event_id_t event, void *data) {
        handler_(sender, event, data);
    }
};

class SubscriberThis: public Subscriber {
    public:
        SubscriberThis(const std::list<event_id_t>& events) : Subscriber(
            events, 
            [this] (void *sender, event_id_t event, void *data) {
                this->dfe_handle_event(sender, event, data);
            }) {

        }
        virtual void dfe_handle_event(void *sender, event_id_t event, void *data) = 0;
    
};

void publish_event(void *sender, event_id_t event, void *data);

} // namespace editorium
