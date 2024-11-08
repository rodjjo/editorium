#pragma once

#include <string>
#include <memory>

namespace editorium {

std::pair<std::shared_ptr<unsigned char>, size_t> base64_decode(const char *src, size_t len);
std::string base64_encode(const unsigned char *src, size_t len);

}

