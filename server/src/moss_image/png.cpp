#include "png.h"

#include <zlib.h>

#include <algorithm>
#include <cstdlib>
#include <system_error>
#include <thread>
#include <vector>

namespace luce::moss_image {

namespace {

constexpr size_t kStripBytes = 256 * 1024;  // filtered bytes per deflate strip, at least
constexpr int    kMaxStrips  = 16;

void put_u32(std::string & out, const uint32_t v) {
    out += char(v >> 24);
    out += char(v >> 16);
    out += char(v >> 8);
    out += char(v);
}

void put_chunk(std::string & out, const char * type, const std::string & data) {
    put_u32(out, uint32_t(data.size()));
    const size_t start = out.size();
    out.append(type, 4);
    out += data;
    put_u32(out, uint32_t(crc32(0, reinterpret_cast<const Bytef *>(out.data() + start), uInt(out.size() - start))));
}

int paeth(const int a, const int b, const int c) {
    const int p  = a + b - c;
    const int pa = std::abs(p - a);
    const int pb = std::abs(p - b);
    const int pc = std::abs(p - c);
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

struct Strip {
    int                  y0 = 0;
    int                  y1 = 0;
    std::vector<uint8_t> filtered;  // the rows, each a filter byte (4, Paeth) and 3 * width bytes
    std::string          deflated;  // raw deflate, ending on a byte boundary
    uLong                adler = 0;
    bool                 ok    = false;
};

void encode_strip(const uint8_t * rgb, const int width, Strip & s, const bool last) {
    const size_t stride = size_t(width) * 3;
    s.filtered.resize(size_t(s.y1 - s.y0) * (stride + 1));
    uint8_t * dst = s.filtered.data();
    for (int y = s.y0; y < s.y1; ++y) {
        const uint8_t * row = rgb + size_t(y) * stride;
        const uint8_t * up  = y > 0 ? row - stride : nullptr;
        *dst++              = 4;
        for (size_t i = 0; i < stride; ++i) {
            const int a = i >= 3 ? row[i - 3] : 0;
            const int b = up != nullptr ? up[i] : 0;
            const int c = up != nullptr && i >= 3 ? up[i - 3] : 0;
            *dst++      = uint8_t(row[i] - paeth(a, b, c));
        }
    }
    s.adler = adler32(adler32(0, nullptr, 0), s.filtered.data(), uInt(s.filtered.size()));

    z_stream z{};
    if (deflateInit2(&z, 1, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
        return;
    }
    s.deflated.resize(deflateBound(&z, uLong(s.filtered.size())) + 16);
    z.next_in      = s.filtered.data();
    z.avail_in     = uInt(s.filtered.size());
    z.next_out     = reinterpret_cast<Bytef *>(s.deflated.data());
    z.avail_out    = uInt(s.deflated.size());
    const int rc   = deflate(&z, last ? Z_FINISH : Z_SYNC_FLUSH);
    s.ok           = (last ? rc == Z_STREAM_END : rc == Z_OK) && z.avail_in == 0;
    s.deflated.resize(z.total_out);
    deflateEnd(&z);
}

} // namespace

std::string encode_png(const uint8_t * rgb, const int width, const int height) {
    const size_t row_bytes = size_t(width) * 3 + 1;
    const int    hw_threads = int(std::max(1u, std::thread::hardware_concurrency()));
    const int    n_strips   = std::clamp(int(row_bytes * size_t(height) / kStripBytes), 1,
                                         std::min({ kMaxStrips, hw_threads, height }));
    std::vector<Strip> strips(n_strips);
    for (int i = 0; i < n_strips; ++i) {
        strips[i].y0 = int(int64_t(height) * i / n_strips);
        strips[i].y1 = int(int64_t(height) * (i + 1) / n_strips);
    }
    std::vector<std::thread> workers;
    int                      next = 1;
    try {
        for (; next < n_strips; ++next) {
            workers.emplace_back(encode_strip, rgb, width, std::ref(strips[next]), next == n_strips - 1);
        }
    } catch (const std::system_error &) {
        // no thread to spare: this one encodes the strips left
    }
    for (int i = next; i < n_strips; ++i) {
        encode_strip(rgb, width, strips[i], i == n_strips - 1);
    }
    encode_strip(rgb, width, strips[0], n_strips == 1);
    for (std::thread & w : workers) {
        w.join();
    }

    // zlib stream: header (level 1), the strips' deflate data, Adler-32 of all filtered bytes.
    std::string zlib = "\x78\x01";
    uLong       adler = adler32(0, nullptr, 0);
    for (const Strip & s : strips) {
        if (!s.ok) {
            return {};
        }
        zlib += s.deflated;
        adler = adler32_combine(adler, s.adler, z_off_t(s.filtered.size()));
    }
    put_u32(zlib, uint32_t(adler));

    std::string ihdr;
    put_u32(ihdr, uint32_t(width));
    put_u32(ihdr, uint32_t(height));
    ihdr += std::string("\x08\x02\x00\x00\x00", 5);  // 8 bits, RGB, deflate, adaptive filters, no interlace

    std::string png = "\x89PNG\r\n\x1a\n";
    put_chunk(png, "IHDR", ihdr);
    put_chunk(png, "IDAT", zlib);
    put_chunk(png, "IEND", {});
    return png;
}

} // namespace luce::moss_image
