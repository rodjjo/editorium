#pragma once

#include <vector>
#include <string>

namespace editorium{

const std::string& executableDir();
std::string configPath();

std::string filepath_dir(const std::string & path);
void blur_gl_contents(int w, int h, int mouse_x, int mouse_y);
bool rectRect(int r1x, int r1y, int r1w, int r1h, int r2x, int r2y, int r2w, int r2h);
std::vector<std::string> list_directory_files(const std::string& path, const std::vector<std::string>& ext);
std::string extract_directory(const std::string& path);
std::string the_item_after(const std::vector<std::string> contents, const std::string& current);
std::string the_item_before(const std::vector<std::string> contents, const std::string& current);

}