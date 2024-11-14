#ifdef _WIN32
#include <Windows.h>
#endif

#include <unistd.h>

#include <GL/gl.h>
#include <FL/Fl.H>
#include <FL/gl.h>

#ifdef _WIN32
#include <Windows.h>
#endif

#include <algorithm>

#include "misc/utils.h"
#include "misc/config.h"


namespace editorium
{
namespace 
{
    std::string executable_dir;
} 

const std::string& executableDir() {
    if (executable_dir.empty()) {
#ifdef _WIN32
    wchar_t path[1024] = { 0, };
    if (GetModuleFileNameW(NULL, path, (sizeof(path) / sizeof(wchar_t)) -1) != 0) {
        executable_dir = path;
    }
#else
        char path[1024] = { 0, };
        if (readlink("/proc/self/exe", path, sizeof(path) - 1) != -1) {
            executable_dir = path;
        }
#endif
        size_t latest = executable_dir.find_last_of("/\\");
        if (latest != std::string::npos) {
            executable_dir = executable_dir.substr(0, latest);
        } else {
            executable_dir = std::string();
        }
    }

    return executable_dir;
}

std::string configPath() {
    return executableDir() + "/editorium-ui-config.json";
}
 
std::string filepath_dir(const std::string & path) {
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos) {
        return path.substr(0, pos + 1);
    }
    return path;
}

void blur_gl_contents(int w, int h, int mouse_x, int mouse_y) {
    if (!get_config()->private_mode()) { 
        return;
    }

    const int half_area = Fl::event_shift() != 0 ? 80 : 40;
    float fx = 2.0 / w;
    float fy = 2.0 / h;
    float hx = half_area * fx;
    float hy = half_area * fy;
    float x1 = (mouse_x * fx - hx) - 1.0;
    float y1 = 1.0 - (mouse_y * fy - hy);
    float x2 = (mouse_x * fx + hx) - 1.0;
    float y2 = 1.0 - (mouse_y * fy + hy);

    glColor4f(0.2, 0.2, 0.2, 1.0);
    glBegin(GL_QUADS);

    // left
    glVertex2f(-1.0, 1.0);
    glVertex2f(x1, 1.0);
    glVertex2f(x1, -1.0);
    glVertex2f(-1.0, -1.0);
    
    // right
    glVertex2f(1.0, 1.0);
    glVertex2f(x2, 1.0);
    glVertex2f(x2, -1.0);
    glVertex2f(1.0, -1.0);

    // center top
    glVertex2f(x1, y1);
    glVertex2f(x1, 1.0);
    glVertex2f(x2, 1.0);
    glVertex2f(x2, y1);

    // center bottom
    glVertex2f(x1, y2);
    glVertex2f(x2, y2);
    glVertex2f(x2, -1.0);
    glVertex2f(x1, -1.0);

    glEnd();
}

bool rectRect(int r1x, int r1y, int r1w, int r1h, int r2x, int r2y, int r2w, int r2h) {

  // are the sides of one rectangle touching the other?

  if (r1x + r1w >= r2x &&    // r1 right edge past r2 left
      r1x <= r2x + r2w &&    // r1 left edge past r2 right
      r1y + r1h >= r2y &&    // r1 top edge past r2 bottom
      r1y <= r2y + r2h) {    // r1 bottom edge past r2 top
        return true;
  }
  return false;
}

std::vector<std::string> list_directory_files(const std::string& path, const std::vector<std::string>& ext) {
    std::vector<std::string> result;
#ifdef _WIN32
    WIN32_FIND_DATAA findFileData;
    HANDLE hFind = FindFirstFileA((path + "/*." + ext).c_str(), &findFileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            result.push_back(findFileData.cFileName);
        } while (FindNextFileA(hFind, &findFileData));
        FindClose(hFind);
    }
#else
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(path.c_str())) != NULL) {
        bool ext_found = false;
        while ((ent = readdir(dir)) != NULL) {
            std::string name = ent->d_name;
            ext_found = false;
            for (const auto& ext_item : ext) {
                if (name.find(ext_item) != std::string::npos) {
                    ext_found = true;
                    break;
                }
            }
            if (ext_found) {
                result.push_back(path + name);
            }
        }
        closedir(dir);
    }
#endif

    // sort the result
    std::sort(result.begin(), result.end());

    return result;
}

std::string extract_directory(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos) {
        return path.substr(0, pos + 1);
    }
    return path;
}

std::string the_item_after(const std::vector<std::string> contents, const std::string& current) {
    for (size_t i = 0; i < contents.size(); i++) {
        if (contents[i] == current) {
            if (i + 1 < contents.size()) {
                return contents[i + 1];
            }
        }
    }
    if (contents.size() > 0) {
        return contents[0];
    }
    return std::string();
}

std::string the_item_before(const std::vector<std::string> contents, const std::string& current) {
    for (size_t i = 0; i < contents.size(); i++) {
        if (contents[i] == current) {
            if (i > 0) {
                return contents[i - 1];
            }
        }
    }
    if (contents.size() > 0) {
        return contents[contents.size() - 1];
    }
    return std::string();
}

} // namespace editorium