#include "video/video.h"
#include "video/decoder.h"
#include "video/encoder.h"

namespace vs {

// abastract classes destructors:

StreamInfo::~StreamInfo() {}
Decoder::~Decoder() {}
Encoder::~Encoder() {}
// instance creating functions:

std::shared_ptr<vs::Decoder> open_file(const char* path) {
    return std::shared_ptr<vs::Decoder>(new vs::DecoderImp(path, vs::file_source));
}

std::shared_ptr<Encoder> encoder(
    const char *codec_name,
    const char *path,
    unsigned int frame_width,
    unsigned int frame_height,
    int fps_numerator,
    int fps_denominator,
    int bit_rate,
    const char *title,
    const char *author,
    const char *tags
) {
    return std::shared_ptr<vs::Encoder>(new vs::EncoderImp(
        codec_name,
        path,
        title ? title : "",
        author ? author : "",
        tags ? tags : "",
        frame_width,
        frame_height,
        fps_numerator,
        fps_denominator,
        bit_rate
    ));
}

void initialize() {
    // av_register_all();  // linux needs - edit 2025: deprecated not necessary anymore
    avformat_network_init();
}

}  // namespace vs