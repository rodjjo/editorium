#include <strings.h>

#include "base64.h"
#include "b64.h"

namespace editorium
{
    std::pair<std::shared_ptr<unsigned char>, size_t> base64_decode(const char *src, size_t len) {
        auto decoded = std::shared_ptr<unsigned char>(b64_decode_ex(src, len, &len));
        return std::make_pair(decoded, len);
    }

    std::string base64_encode(const unsigned char *src, size_t len) {
        const char *encoded = b64_encode(src, len);
        std::string result(encoded);
        free((void*)encoded);
        return result;
    }

} // namespace editorium


