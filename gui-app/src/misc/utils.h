#pragma once

#include <string>

namespace editorium{

const std::string& executableDir();
std::string configPath();

std::string filepath_dir(const std::string & path);
void blur_gl_contents(int w, int h, int mouse_x, int mouse_y);
bool rectRect(int r1x, int r1y, int r1w, int r1h, int r2x, int r2y, int r2w, int r2h);
}