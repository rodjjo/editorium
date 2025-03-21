#pragma once

#include <string>
#include <memory>

#include "video/video.h"
#include "video/headers.h"
#include "video/guards.h"

namespace vs {

class EncoderImp: public Encoder {
 public:
    EncoderImp(
        const char *codec_name,
        const char *path,
        const char *title,
        const char *author,
        const char *tags,
        unsigned int frame_width,
        unsigned int frame_height,
        int fps_numerator,
        int fps_denominator,
        int bit_rate);
    virtual ~EncoderImp();
    bool frame(const unsigned char* buffer) override;
    const char* error() override;
    bool finish() override;
 private:
    void init_encoder();
    bool find_codec();
    bool allocate_stream();
    const char *find_format();
    bool configure_codec();
    bool open_output_file();
    bool allocate_frame();
    bool allocate_format();
    bool encode_frame(AVFrame *frame_data);
    void report_error(const char *error);
    bool flush_frames();
 protected:
    vs::AVCodecContextPtr codec_ctx_;
    vs::FormatContextPtr format_ctx_;
    vs::AVFramePtr frame_;
    vs::SwsContextPtr output_color_context_;
    const AVCodec *codec_;
    AVStream *stream_;

    std::string error_;
    bool opened_;
    bool finished_;
    bool should_close_file_;

    std::string codec_name_;
    std::string path_;

    std::string title_;
    std::string author_;
    std::string tags_;

    unsigned int frame_width_;
    unsigned int frame_height_;
    int max_bidirectional_frames_;
    int frame_pts_;
    int got_packet_ptr_;
    int frame_align_;
    int key_frame_interval_;
    int fps_numerator_;
    int fps_denominator_;
    int bit_rate_;
};

}  // namespace vs

