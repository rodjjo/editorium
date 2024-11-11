#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "profiles.h"
#include "misc/config.h"


namespace editorium {

using json = nlohmann::json;

namespace {
    std::string prompt_profile_name = "profile.json";
    json prompt_profile_data;
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

    try {
        std::ifstream ifile(profile_path);
        ifile >> prompt_profile_data;
        ifile.close();
    } catch (std::exception e) {
        printf("Error loading profile: %s\n", e.what());
        prompt_profile_data = json();
    }
}

void prompt_save_profile() {
    if (!std::filesystem::exists(get_config()->profiles_dir())) {
        std::filesystem::create_directories(get_config()->profiles_dir());
    }

    std::string profile_path = get_config()->profiles_dir() + "/" + prompt_profile_name; 
    std::ofstream ofile(profile_path);
    ofile << prompt_profile_data.dump(4);
    ofile.close();
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


}