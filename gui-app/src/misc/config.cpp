#include <fstream>
#include <nlohmann/json.hpp>
#include "misc/config.h"
#include "misc/utils.h"

namespace editorium
{
    using json = nlohmann::json;

    json load_config() {
        std::ifstream file(configPath());
        if (!file.is_open()) {
            return json();
        }
        try{
            json cf;
            file >> cf;
            return cf;
        } catch(std::exception e) {
            printf("Failed to load the config: %s", e.what());
            return json();
        }
    }

    void save_config(const json &config) {
        std::ofstream file(configPath());
        if (!file.is_open()) {
            return;
        }
        file << config.dump(4);
    }

    namespace 
    {
        std::unique_ptr<Config> conf;
    } // namespace 
    
    Config* get_config() {
        if (!conf) {
            conf.reset(new Config());
            conf->load();
        }
        return conf.get();
    }

    Config::Config() {
    }

    Config::~Config() {
    }

    bool Config::load() {
        json cf = load_config(); // load the configuration

        if (cf.is_null()) {
            puts("Configuration empty, using the default values...");
            return true; // there is no config saved yet use the defaults
        }

        try {
            auto load_map = [&cf] (const char *key, std::map<std::string, std::string> & output) {
                if (cf.contains(key) && cf[key].is_object()) {
                    auto hist =  cf[key];
                    output.clear();
                    for (auto it = hist.begin(); it != hist.end(); ++it) {
                        output[it.key()] = it.value().get<std::string>();
                    }
                }
            };

            load_map("open_history", last_open_dirs);
            load_map("save_history", last_save_dirs);

            auto load_key = [&cf] (const char *key, const char *sub_key, const char *default_value="") -> std::string {
                std::string r = default_value;
                if (!cf.contains(key)) {
                    return r;
                }
                auto sub = cf[key];
                if (sub.contains(sub_key)) {
                    return sub[sub_key].get<std::string>();
                }
                return r;
            };

            profiles_dir_ = load_key("directories", "profiles_dir");
            server_url_ = load_key("urls", "server_url");
            sdxl_base_model_ = load_key("base_models", "sdxl_base_model");
            flux_base_model_ = load_key("base_models", "flux_base_model");
            sd35_base_model_ = load_key("base_models", "sd35_base_model");

            chat_bot_repo_id_ = load_key("chat_bot", "repo_id", chat_bot_repo_id_.c_str());
            chat_bot_model_name_ = load_key("chat_bot", "model_name", chat_bot_model_name_.c_str());
            chat_bot_template_ = load_key("chat_bot", "template", chat_bot_template_.c_str());
            chat_bot_response_after_ = load_key("chat_bot", "response_after", chat_bot_response_after_.c_str());

            if (cf.contains("chat_bot")) {
                json value = cf["chat_bot"];
                if (value.contains("max_new_tokens")) {
                    chat_bot_max_new_tokens_ = value["max_new_tokens"];
                }
                if (value.contains("temperature")) {
                    chat_bot_temperature_ = value["temperature"];
                }
                if (value.contains("top_p")) {
                    chat_bot_top_p_ = value["top_p"];
                }
                if (value.contains("top_k")) {
                    chat_bot_top_k_ = value["top_k"];
                }
                if (value.contains("repetition_penalty")) {
                    chat_bot_repetition_penalty_ = value["repetition_penalty"];
                }
            }

            chat_vision_repo_id_ = load_key("chat_vision", "repo_id", chat_vision_repo_id_.c_str());
            if (cf.contains("chat_vision")) {
                json value = cf["chat_vision"];
                if (value.contains("temperature")) {
                    chat_vision_temperature_ = value["temperature"];
                }
            }

            if (cf.contains("float16_enabled")) {
                use_float16_ = cf["float16_enabled"];
            }
            if (cf.contains("private_mode_enabled")) {
                private_mode_ = cf["private_mode_enabled"];
            }
            if (cf.contains("keep_in_memory")) {
                keep_in_memory_ = cf["keep_in_memory"];
            }
        } catch(std::exception e) {
            printf("Failed to load the config: %s", e.what());
            return false;
        }

        return true;
    }

    bool Config::save() {
        json cf;
        try {
            auto store_map = [&cf] (const char *key, const std::map<std::string, std::string> & input) {
                json d;
                for (const auto & v: input) {
                    d[v.first.c_str()] = v.second;
                }
                cf[key] = d;
            };
            store_map("open_history", last_open_dirs);
            store_map("save_history", last_save_dirs);
            
            json dirs;
            dirs["profiles_dir"] = profiles_dir_;
            cf["directories"] = dirs;

            json urls;
            urls["server_url"] = server_url_;
            cf["urls"] = urls;

            json base_models;
            base_models["sdxl_base_model"] = sdxl_base_model_;
            base_models["flux_base_model"] = flux_base_model_;
            base_models["sd35_base_model"] = sd35_base_model_;
            cf["base_models"] = base_models;
            
            cf["float16_enabled"] = use_float16_;
            cf["private_mode_enabled"] = private_mode_;
            cf["keep_in_memory"] = keep_in_memory_;

            json chat_bot;
            chat_bot["repo_id"] = chat_bot_repo_id_;
            chat_bot["model_name"] = chat_bot_model_name_;
            chat_bot["template"] = chat_bot_template_;
            chat_bot["response_after"] = chat_bot_response_after_;
            chat_bot["max_new_tokens"] = chat_bot_max_new_tokens_;
            chat_bot["temperature"] = chat_bot_temperature_;
            chat_bot["top_p"] = chat_bot_top_p_;
            chat_bot["top_k"] = chat_bot_top_k_;
            chat_bot["repetition_penalty"] = chat_bot_repetition_penalty_;
            cf["chat_bot"] = chat_bot;

            json chat_vision;
            chat_vision["repo_id"] = chat_vision_repo_id_;
            chat_vision["temperature"] = chat_vision_temperature_;
            cf["chat_vision"] = chat_vision;
        } catch(std::exception e) {
            printf("Failed to save the config: %s", e.what());
            return false;
        }
        save_config(cf);
        return true;
    }

    std::string Config::last_save_directory(const char *scope) {
        auto it = last_save_dirs.find(scope);
        if (last_save_dirs.end() != it) {
            return it->second;
        }
        return std::string();
    }

    std::string Config::last_open_directory(const char *scope) {
        auto it = last_open_dirs.find(scope);
        if (last_open_dirs.end() != it) {
            return it->second;
        }
        return std::string();
    }

    void Config::last_save_directory(const char *scope, const char* value) {
        last_save_dirs[scope] = value;
        save();
    }

    void Config::last_open_directory(const char *scope, const char* value) {
        last_open_dirs[scope] = value;
        save();
    }

    std::string Config::profiles_dir() {
        return profiles_dir_;
    }

    void Config::profiles_dir(const char *value) {
        profiles_dir_ = value;
    }

    std::string Config::server_url() {
        return server_url_;
    }

    void Config::server_url(const char *value) {
        server_url_ = value;
    }

    std::string Config::sdxl_base_model() {
        return sdxl_base_model_;
    }

    std::string Config::flux_base_model() {
        return flux_base_model_;
    }

    std::string Config::sd35_base_model() {
        return sd35_base_model_;
    }

    void Config::sdxl_base_model(const char *value) {
        sdxl_base_model_ = value;
    }

    void Config::flux_base_model(const char *value) {
        flux_base_model_ = value;
    }

    void Config::sd35_base_model(const char *value) {
        sd35_base_model_ = value;
    }


    bool Config::use_float16() {
        return use_float16_;
    }

    bool Config::private_mode() {
        return private_mode_;
    }

    void Config::use_float16(bool value) {
        use_float16_ = value;
    }

    void Config::private_mode(bool value) {
        private_mode_ = value;
    }

    void Config::keep_in_memory(bool value) {
        keep_in_memory_ = value;
    }

    bool Config::keep_in_memory() {
        return keep_in_memory_;
    }

    void Config::chat_bot_repo_id(const char *value) {
        chat_bot_repo_id_ = value;
    }

    void Config::chat_bot_model_name(const char *value) {
        chat_bot_model_name_ = value;
    }

    void Config::chat_bot_template(const char *value) {
        chat_bot_template_ = value;
    }

    void Config::chat_bot_response_after(const char *value) {
        chat_bot_response_after_ = value;
    }

    void Config::chat_bot_max_new_tokens(int value) {
        chat_bot_max_new_tokens_ = value;
    }

    void Config::chat_bot_temperature(float value) {
        chat_bot_temperature_ = value;
    }

    void Config::chat_bot_top_p(float value) {
        chat_bot_top_p_ = value;
    }

    void Config::chat_bot_top_k(int value) {
        chat_bot_top_k_ = value;
    }

    void Config::chat_bot_repetition_penalty(float value) {
        chat_bot_repetition_penalty_ = value;
    }

    std::string Config::chat_bot_repo_id() {
        return chat_bot_repo_id_;
    }

    std::string Config::chat_bot_model_name() {
        return chat_bot_model_name_;
    }

    std::string Config::chat_bot_template() {
        return chat_bot_template_;
    }

    std::string Config::chat_bot_response_after() {
        return chat_bot_response_after_;
    }

    int Config::chat_bot_max_new_tokens() {
        return chat_bot_max_new_tokens_;
    }

    float Config::chat_bot_temperature() {
        return chat_bot_temperature_;
    }

    float Config::chat_bot_top_p() {
        return chat_bot_top_p_;
    }

    int Config::chat_bot_top_k() {
        return chat_bot_top_k_;
    }

    float Config::chat_bot_repetition_penalty() {
        return chat_bot_repetition_penalty_;
    }

    void Config::chat_vision_repo_id(const char *value) {
        chat_vision_repo_id_ = value;
    }

    void Config::chat_vision_temperature(float value) {
        chat_vision_temperature_ = value;
    }

    std::string Config::chat_vision_repo_id() {
        return chat_vision_repo_id_;
    }

    float Config::chat_vision_temperature() {
        return chat_vision_temperature_;
    }

} // namespace editorium
