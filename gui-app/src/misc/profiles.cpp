#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "profiles.h"
#include "misc/config.h"


namespace editorium {

using json = nlohmann::json;

namespace {
    std::string prompt_profile_name = "profile.json";
    std::string chatbot_profile_name = "chatbot.json";
    json prompt_profile_data;
    json chatbot_profile_data;
}

void load_profile_file(const std::string& filename, json &data) {
    if (!std::filesystem::exists(filename)) {
        return;
    }

    try {
        std::ifstream ifile(filename);
        ifile >> data;
        ifile.close();
    } catch (std::exception e) {
        printf("Error loading profile: %s\n", e.what());
        data = json();
    }
}

void save_profile_file(const std::string& filename, const json &data) {
    if (!std::filesystem::exists(get_config()->profiles_dir())) {
        std::filesystem::create_directories(get_config()->profiles_dir());
    }

    std::string filepath = get_config()->profiles_dir() + "/" + filename;
    
    try {
        std::ofstream ofile(filepath);
        ofile << data.dump(4);
        ofile.close();
    } catch (std::exception e) {
        printf("Error saving profile: %s\n", e.what());
    }
}

void prompt_load_profile() {
    std::string profile_dir = get_config()->profiles_dir();
    if (profile_dir.empty()) {
        return;
    }
    
    std::string profile_path = get_config()->profiles_dir() + "/" + prompt_profile_name; 
    if (!std::filesystem::exists(profile_path)) {
       return;
    }
    
    load_profile_file(profile_path, prompt_profile_data);
}

void prompt_save_profile() {
    save_profile_file(prompt_profile_name, prompt_profile_data);
}

void chatbot_load_profile() {
    std::string profile_dir = get_config()->profiles_dir();
    if (profile_dir.empty()) {
        return;
    }
    
    std::string profile_path = get_config()->profiles_dir() + "/" + chatbot_profile_name; 
    if (!std::filesystem::exists(profile_path)) {
       return;
    }
    
    load_profile_file(profile_path, chatbot_profile_data);
}

void chatbot_save_profile() {
    save_profile_file(chatbot_profile_name, chatbot_profile_data);
}

std::string key_to_name(const std::vector<std::string>& key) {
    std::string key_name;
    for (size_t i = 0; i < key.size(); i++) {
        key_name += key[i];
        if (i < key.size() - 1) {
            key_name += "::";
        }
    }
    return key_name;
}

void prompt_profile_set_string(const std::vector<std::string>& key, const std::string& value) {
    std::string key_name = key_to_name(key);
    prompt_profile_data[key_name] = value;
}


void prompt_profile_set_int(const std::vector<std::string>& key, int value) {
    std::string key_name = key_to_name(key);
    prompt_profile_data[key_name] = value;    
}

void prompt_profile_set_float(const std::vector<std::string>& key, float value) {
    std::string key_name = key_to_name(key);
    prompt_profile_data[key_name] = value;
}

void prompt_profile_set_boolean(const std::vector<std::string>& key, float value) {
    std::string key_name = key_to_name(key);
    prompt_profile_data[key_name] = value;
}

std::string prompt_profile_get_string(const std::vector<std::string>& key, const std::string& default_value) {
    std::string key_name = key_to_name(key);
    if (prompt_profile_data.contains(key_name)) {
        return prompt_profile_data[key_name].get<std::string>();
    }
    return default_value;
}

int prompt_profile_get_int(const std::vector<std::string>& key, int default_value) {
    std::string key_name = key_to_name(key);
    if (prompt_profile_data.contains(key_name)) {
        return prompt_profile_data[key_name].get<int>();
    }
    return default_value;
}

float prompt_profile_get_float(const std::vector<std::string>& key, float default_value) {
    std::string key_name = key_to_name(key);
    if (prompt_profile_data.contains(key_name)) {
        return prompt_profile_data[key_name].get<float>();
    }
    return default_value;
}

bool prompt_profile_get_boolean(const std::vector<std::string>& key, bool default_value) {
    std::string key_name = key_to_name(key);
    if (prompt_profile_data.contains(key_name)) {
        return prompt_profile_data[key_name].get<bool>();
    }
    return default_value;
}

std::string chatbot_profile_get_string(const std::vector<std::string>& key, const std::string& default_value) {
    std::string key_name = key_to_name(key);
    if (chatbot_profile_data.contains(key_name)) {
        return chatbot_profile_data[key_name].get<std::string>();
    }
    return default_value;
}

void chatbot_profile_set_string(const std::vector<std::string>& key, const std::string& value) {
    std::string key_name = key_to_name(key);
    chatbot_profile_data[key_name] = value;
}

}