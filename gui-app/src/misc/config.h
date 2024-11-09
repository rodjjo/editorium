#include <memory>
#include <string>
#include <map>

namespace editorium
{

class Config {
    public:
        Config();
        virtual ~Config();
        bool load();
        bool save();
        std::string last_save_directory(const char *scope);
        std::string last_open_directory(const char *scope);
        void last_save_directory(const char *scope, const char* value);
        void last_open_directory(const char *scope, const char* value);
        std::string profiles_dir();
        void profiles_dir(const char *value);

        bool use_float16();
        bool private_mode();
        bool keep_in_memory();

        void use_float16(bool value);
        void private_mode(bool value);
        void keep_in_memory(bool value);

    private:
        std::map<std::string, std::string> last_save_dirs;
        std::map<std::string, std::string> last_open_dirs;
        std::string profiles_dir_;
        bool use_float16_ = true;
        bool private_mode_ = false;
        bool keep_in_memory_ = false;
};
    
Config* get_config();

} // namespace editorium
