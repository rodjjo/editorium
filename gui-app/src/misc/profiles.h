#pragma once

#include <vector>
#include <string>

namespace editorium {

void prompt_load_profile();
void prompt_save_profile();
void chatbot_load_profile();
void chatbot_save_profile();

void prompt_profile_set_string(const std::vector<std::string>& key, const std::string& value);
void prompt_profile_set_int(const std::vector<std::string>& key, int value);
void prompt_profile_set_float(const std::vector<std::string>& key, float value);
void prompt_profile_set_boolean(const std::vector<std::string>& key, float value);

std::string prompt_profile_get_string(const std::vector<std::string>& key, const std::string& default_value = "");
int prompt_profile_get_int(const std::vector<std::string>& key, int default_value = 0);
float prompt_profile_get_float(const std::vector<std::string>& key, float default_value = 0.0);
bool prompt_profile_get_boolean(const std::vector<std::string>& key, bool default_value = false);

std::string chatbot_profile_get_string(const std::vector<std::string>& key, const std::string& default_value = "");
void chatbot_profile_set_string(const std::vector<std::string>& key, const std::string& value);

void dialogs_load_profile();
void dialogs_save_profile();
std::string dialogs_profile_get_string(const std::vector<std::string>& key, const std::string& default_value = "");
void dialogs_profile_set_string(const std::vector<std::string>& key, const std::string& value);


}