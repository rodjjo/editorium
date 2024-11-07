#include "misc/config.h"

namespace editorium
{
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
        return false;
    }

    bool Config::save() {
        return false;
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

    std::string Config::add_model_dir() {
        return add_model_dir_;
    }

    std::string Config::add_lora_dir() {
        return add_lora_dir_;
    }

    std::string Config::add_emb_dir() {
        return add_emb_dir_;
    }

    void Config::add_model_dir(const char *value) {
        add_model_dir_ = value;
    }

    void Config::add_lora_dir(const char *value) {
        add_lora_dir_ = value;
    }

    void Config::add_emb_dir(const char *value) {
        add_emb_dir_ = value;
    }

    bool Config::filter_nsfw() {
        return filter_nsfw_;
    }

    bool Config::use_float16() {
        return use_float16_;
    }

    bool Config::private_mode() {
        return private_mode_;
    }

    void Config::filter_nsfw(bool value) {
        filter_nsfw_ = value;
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

} // namespace editorium
