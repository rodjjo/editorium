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
        std::string profiles_dir();
        void profiles_dir(const char *value);
        std::string server_url();
        void server_url(const char *value);
        std::string sdxl_base_model();
        std::string flux_base_model();
        std::string lumina_base_model();
        std::string sd35_base_model();
        std::string arch_speed_model();
        void sdxl_base_model(const char *value);
        void flux_base_model(const char *value);
        void lumina_base_model(const char *value);
        void sd35_base_model(const char *value);
        void arch_speed_model(const char *value);
        bool use_float16();
        bool private_mode();
        bool keep_in_memory();

        void use_float16(bool value);
        void private_mode(bool value);
        void keep_in_memory(bool value);

        void chat_bot_repo_id(const char *value);
        void chat_bot_model_name(const char *value);
        void chat_bot_template(const char *value);
        void chat_bot_response_after(const char *value);
        void chat_bot_max_new_tokens(int value);
        void chat_bot_temperature(float value);
        void chat_bot_top_p(float value);
        void chat_bot_top_k(int value);
        void chat_bot_repetition_penalty(float value);
        std::string chat_bot_repo_id();
        std::string chat_bot_model_name();
        std::string chat_bot_template();
        std::string chat_bot_response_after();
        int chat_bot_max_new_tokens();
        float chat_bot_temperature();
        float chat_bot_top_p();
        int chat_bot_top_k();
        float chat_bot_repetition_penalty();

        void chat_vision_repo_id(const char *value);
        void chat_vision_temperature(float value);
        std::string chat_vision_repo_id();
        float chat_vision_temperature();

        std::string chat_story_repo_id();

    private:
        std::string profiles_dir_;
        std::string server_url_ = "ws://localhost:5001";
        std::string sdxl_base_model_;
        std::string flux_base_model_;
        std::string lumina_base_model_;
        std::string sd35_base_model_;
        std::string arch_speed_model_;

        std::string chat_bot_repo_id_ = "TheBloke/Nous-Hermes-13B-GPTQ";
        std::string chat_bot_model_name_ = "model";
        std::string chat_bot_template_ = "### Instruction:\n{context}\n### Input:\n{input}\n### Response:\n";
        std::string chat_bot_response_after_;
        int chat_bot_max_new_tokens_ = 512;
        float chat_bot_temperature_ = 1;
        float chat_bot_top_p_ = 1;
        int chat_bot_top_k_ = 0;
        float chat_bot_repetition_penalty_ = 1;
        
        // chat vision
        std::string chat_vision_repo_id_ = "openbmb/MiniCPM-Llama3-V-2_5-int4";
        float chat_vision_temperature_ = 0.7;

        bool use_float16_ = true;
        bool private_mode_ = false;
        bool keep_in_memory_ = false;
};
    
Config* get_config();

} // namespace editorium
