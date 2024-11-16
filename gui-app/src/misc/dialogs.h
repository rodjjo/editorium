#pragma once

#include <inttypes.h>
#include <string>

namespace editorium
{
    
typedef enum {
    r_yes = 0,
    r_no,
    r_cancel
} response_t;

bool ask(const char *message);
response_t yes_nc(const char *message);
void show_error(const char *message);
void show_info(const char *message);

std::string choose_image_to_open_fl(const std::string& scope);
std::string choose_image_to_save_fl(const std::string& scope);
bool pickup_color(const char* title, uint8_t *r, uint8_t *g, uint8_t *b);
    
} // namespace editorium
