#include <vector>
#include <string>
#include <fstream>
#include <set>
#include <map>
#include <regex>
#include "workflow_parser.h"


namespace editorium {

std::vector<std::string> get_lines_from_file(const std::string &path) {
    std::vector<std::string> result;
    std::ifstream ifile(path);
    if (ifile.is_open()) {
        std::string line;
        std::string line_sum;
        bool include_started = false;
        while (std::getline(ifile, line)) {
            line = line.substr(0, line.find_last_not_of(" \t") + 1);
            if (line.find("#include ") == 0) {
                include_started = true;
            }
            if (line.back() == '\\' && include_started) {
                line = line.substr(0, line.size() - 1);
                line_sum += line;
            } else {
                line_sum += line;
                include_started = false;
                result.push_back(line_sum);
                line_sum.clear();
            }
        }
        if (!line_sum.empty()) {
            result.push_back(line_sum);
        }
    }
    return result;
}

std::vector<std::string> read_worflow_file(const std::string & path) {
    std::vector<std::string> result = get_lines_from_file(path);
    
    return result;
}



/*


std::vector<std::string> read_worflow_file(
    const std::string& include_dir, 
    const std::string& path, 
    std::set<std::string>& already_included, 
    std::map<std::string, std::string> & replace_input,
    const std::string& suffix
) {
    std::vector<std::string> result;
    std::string include_dir_path;
    if (include_dir.empty()) {
        include_dir_path = path;
    } else {
        include_dir_path = include_dir + "/" + path;
    }
    std::string full_path = include_dir_path;
    include_dir_path = include_dir_path.substr(0, include_dir_path.find_last_of("/"));
    if (!std::filesystem::exists(full_path)) {
        throw std::runtime_error("File " + full_path + " not found");
    }
    std::string included_track = full_path + "-" + suffix;
    if (already_included.find(included_track) != already_included.end()) {
        throw std::runtime_error("File " + full_path + " already included");
    }
    already_included.insert(included_track);
    std::vector<std::string> file_content = get_lines_from_file(full_path);
    if (!suffix.empty()) {
        std::regex replace_include_from("<insert_task:([^<>]+)>");
        std::regex replace_suffix("task://([^<>:]+)(:|$)");
        for (std::string &line : file_content) {
            line = line.substr(0, line.find_last_not_of(" \t") + 1);
            if (line.find("#name=") == 0) {
                line = line + "-" + suffix;
                continue;
            }
            std::smatch match;
            if (std::regex_search(line, match, replace_include_from)) {
                line = std::regex_replace(line, replace_include_from, "<insert_task:" + match[1].str() + "-" + suffix + ">");
            }
            if (std::regex_search(line, match, replace_suffix)) {
                line = std::regex_replace(line, replace_suffix, "task://" + match[1].str() + "-" + suffix + match[2].str());
            }
        }
    }
    if (!replace_input.empty()) {
        for (const auto & [key, value] : replace_input) {
            for (std::string &line : file_content) {
                line = line.substr(0, line.find_last_not_of(" \t") + 1);
                if (line.find("task://<" + key + ">") != std::string::npos) {
                    line = std::regex_replace(line, std::regex("task://<" + key + ">"), "task://" + value);
                }
                while (line.find("<insert_task:<" + key + ">>") != std::string::npos) {
                    line = std::regex_replace(line, std::regex("<insert_task:<" + key + ">>"), "<insert_task:" + value + ">");
                }
            }
        }
    }
    std::regex capture_inputs1("#input=([^#]+)");
    std::regex capture_inputs2("#input\\.([^=]+)=([^#]+)");
    std::regex capture_path(".*#path=([^$#]+).*");
    std::regex capture_suffix(".*#suffix=([0-9a-zA-Z_\\-]+).*");
    for (const std::string &line : file_content) {
        if (line.find("#comment") == 0) {
            continue;
        }
        if (line.find("#include ") == 0) {
            std::smatch match;
            if (std::regex_search(line, match, capture_inputs1)) {
                std::string input = match[1].str();
            }
            std::vector<std::pair<std::string, std::string>> inputs;
            std::sregex_iterator next(line.begin(), line.end(), capture_inputs2);
            std::sregex_iterator end;
            while (next != end) {
                std::smatch match = *next;
                inputs.push_back({match[1].str(), match[2].str()});
                next++;
            }
            std::smatch match;
            if (std::regex_search(line, match, capture_path)) {
                std::string path = match[1].str();
            }
            if (std::regex_search(line, match, capture_suffix)) {
                std::string suffix = match[1].str();
            }
        } else {
            result.push_back(line);
        }
    }   
    return result;
}
/*

def get_lines_from_file(path: str):
    result = []
    with open(path, 'r') as f:
        lines = f.readlines()
        line_sum = ''
        include_started = False
        for line in lines:
            line = line.strip()
            if line.startswith('#include '):
                include_started = True
                
            if line.endswith('\\') and include_started:
                line = line[:-1]
                line_sum += line
            else: 
                line_sum += line
                include_started = False
                result.append(line_sum)
                line_sum = ''

        if line_sum:
            result.append(line_sum)
    return result


    def read_worflow_file(include_dir: str, path: str, already_included: set, replace_input: dict, suffix: str):
        if include_dir == '':
            path = full_path(path)
            include_dir = os.path.dirname(path)
        else:
            path = full_path(os.path.join(include_dir, path))
            include_dir = os.path.dirname(path)
        if os.path.exists(path) is False:
            raise Exception(f"File {path} not found")
        included_track = f'{path}-{suffix}'
        if included_track in already_included:
            raise Exception(f"File {path} already included")
        already_included.add(included_track)
        parsed_lines = []
        capture_inputs1 = re.compile(r'#input=([^#]+)')
        capture_inputs2 = re.compile(r'#input\.([^=]+)=([^#]+)')
        capture_path = re.compile(r'.*#path=([^$#]+).*')
        capture_suffix = re.compile(r'.*#suffix=([0-9a-zA-Z_\-]+).*')
        
        file_content = get_lines_from_file(path)
            
        if suffix:
            replace_include_from = re.compile(r'<insert_task:([^<>]+)>')
            replace_suffix = re.compile(r'task://([^<>:]+)(:|$)')
            for index, line in enumerate(file_content):
                line = line.strip()
                if line.startswith("#name="):
                    file_content[index] = f'{line.strip()}-{suffix}'
                    continue
                file_content[index] = re.sub(replace_include_from, rf'<insert_task:\g<1>-{suffix}>', line)
                file_content[index] = re.sub(replace_suffix, rf'task://\g<1>-{suffix}\g<2>', line)    
                
        if replace_input:
            print("Replacing inputs of file ", path, " with ", replace_input)
            for key in replace_input:
                for index, line in enumerate(file_content):
                    line = line.strip()
                    if f"task://<{key}>" in line:
                        file_content[index] = line.replace(f"task://<{key}>", f'task://{replace_input[key]}')
                    while f"<insert_task:<{key}>>" in line:
                        line = line.replace(f"<insert_task:<{key}>>", f'<insert_task:{replace_input[key]}>')
                        file_content[index] = line
                        
        for index, line in enumerate(file_content):
            line = line.strip()
            if "<insert_task:<" in line or "from://<" in line or "task://<" in line:
                print(f"[WARNING] Invalid include line: {line} it does not have #suffix=value")

        for line in file_content:
            line = line.strip()  # #include #input=bla #path=bla
            if line.startswith("#comment"):
                continue
            if line.startswith("#include "):
                inputs1 = re.search(capture_inputs1, line)
                inputs2 = re.findall(capture_inputs2, line)
                parsed_inputs = {}
                if inputs1:
                    parsed_inputs["input"] = inputs1.group(1).strip()
                if inputs2:
                    for input in inputs2:
                        parsed_inputs[f'input.{input[0].strip()}'] = input[1].strip()
                print(parsed_inputs)
                parsed_path = re.match(capture_path, line)
                parsed_suffix = re.match(capture_suffix, line)
                if not parsed_suffix:
                    raise Exception(f"Invalid include line: {line} it does not have #suffix=value")
                if not parsed_path:
                    raise Exception(f"Invalid include line: {line} it does not have #path=value")
                
                path = parsed_path.group(1).strip()
                suffix = parsed_suffix.group(1).strip()
                
                parsed_lines += read_worflow_file(include_dir, path, already_included, parsed_inputs, suffix)
            else:
                parsed_lines.append(line)
        return parsed_lines


def extract_workflow_collections(include_dir: str, content: List[str], collections: dict) -> dict:
    in_task = False
    task_type = ''
    config_path = ''
    for line in content:
        line = line.strip()
        if line.startswith('#start'):
            in_task = True
            continue
        if line.startswith('#end'):
            if task_type == 'execute':
                if not config_path:
                    raise Exception(f"Invalid execute task configuration missing path: {line}")
                config_path_complete = full_path(os.path.join(include_dir, config_path))
                if os.path.exists(config_path_complete) is False:
                    raise Exception(f"Invalid execute task configuration path not found: {config_path_complete}")
                if config_path not in collections:
                    collections[config_path] = read_worflow_file(include_dir, config_path, set(), {}, '')
                    collections = extract_workflow_collections(os.path.dirname(config_path_complete), collections[config_path], collections)
            in_task = False
            task_type = ''
            config_path = ''
            continue
        if in_task:
            if line.startswith('#type='):
                task_type = line[6:].strip()
            elif line.startswith('#task_type='):
                task_type = line[11:].strip()
            elif line.startswith('#config.path='):
                config_path = line[13:].strip()
            if task_type and task_type != 'execute':
                config_path = ''            
                
    return collections
*/    /*

json parse_workflow(const std::string &path) {
    json result;
    return result;
}

*/

} // namespace editorium
