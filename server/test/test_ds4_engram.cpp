// DeepSeek V4.1 Engram addressing and row decode against golden vectors.
//
// The expected row ids and decoded values were produced by a numpy port of
// DeepSeek's inference/engram.py (NgramHashState.forward) and of the
// reference row decode (E4M3 x 2^(E8M0 - 127), rounded to bf16), on the
// synthetic constants below. The same port agreed with this implementation on
// 240,000 row ids for the released constants and on 153,600 values read from
// a real table.

#include "deepseek4/deepseek4_engram.h"
#include "CppUnitTestFramework.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

using namespace luce::common;

namespace {

struct DeepSeek4EngramFixture {};

constexpr int kLayers = 2;
constexpr int kHeads = 3;
constexpr int kMaxNgram = 4;
constexpr int kCols = (kMaxNgram - 1) * kHeads;
constexpr int32_t kPad = 1;
constexpr int kVocab = 40;

const std::vector<uint64_t> kMultipliers = {
    1000000000039ull, 777777777777777ull, 123456789012345ull, 98765432109877ull,
    3ull, 5555555555555555ull, 31415926535897ull, 27182818284591ull,
};
const std::vector<uint64_t> kPrimes = {
    1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051,
    2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069,
};
const std::vector<int32_t> kTokens = {
    7, 0, 39, 12, 12, 5, 33, 18, 2, 27, 9, 9, 9, 31, 4, 16, 38, 21, 1, 11, 30, 6, 25, 14,
};
// [token][layer][col]
const std::vector<uint32_t> kExpectedRows = {
    580, 1570, 2262, 3547, 5047, 6073, 6550, 8177, 9075,
    277, 3858, 4579, 7162, 9728, 11077, 13805, 14939, 16397,
    342, 1383, 2144, 3072, 4753, 5891, 6137, 7420, 8612,
    1681, 2502, 5103, 6146, 8681, 12038, 14139, 14822, 16651,
    574, 1964, 2223, 4041, 4619, 5213, 6378, 7494, 9052,
    503, 3056, 4790, 7802, 9216, 10990, 12734, 15565, 16259,
    633, 1655, 2666, 3169, 4614, 5750, 6426, 8208, 8683,
    758, 3266, 5015, 6227, 8634, 10728, 12238, 14496, 17878,
    629, 1561, 2077, 4023, 4509, 5242, 6825, 7465, 9028,
    1796, 3600, 5203, 6959, 9798, 10664, 12584, 15198, 17310,
    979, 1521, 2692, 3099, 4207, 5103, 6641, 7841, 8846,
    1816, 3620, 5223, 6403, 9698, 12022, 12173, 14688, 17736,
    497, 1916, 2975, 3746, 4523, 5263, 6756, 7404, 8878,
    508, 2033, 4780, 6184, 8759, 10720, 12376, 16042, 17200,
    631, 1213, 2580, 3215, 5016, 5986, 6943, 7854, 9071,
    1417, 2031, 4218, 7637, 8478, 11784, 14050, 15005, 17844,
    226, 1014, 2523, 3395, 4308, 5917, 6773, 7690, 8739,
    1034, 2721, 5897, 7229, 8429, 11737, 13616, 16232, 17647,
    473, 1162, 2452, 3236, 4436, 5106, 6545, 7339, 8803,
    1492, 3341, 4929, 7097, 8512, 11456, 13818, 14484, 17263,
    296, 1782, 2086, 3175, 4657, 5432, 6634, 7556, 8897,
    1377, 2631, 4511, 7093, 9361, 11597, 13513, 14189, 17412,
    125, 1107, 2916, 3243, 4846, 5194, 7128, 7812, 9030,
    715, 2835, 5290, 8029, 9248, 10106, 13912, 15653, 18230,
    125, 1107, 2916, 3685, 4271, 5952, 6909, 8114, 8450,
    715, 2835, 5290, 8019, 9838, 10914, 13319, 14469, 16695,
    492, 1177, 2210, 3133, 4401, 5320, 6913, 7481, 8293,
    735, 2855, 5310, 8039, 9858, 10934, 13412, 14414, 16307,
    150, 1850, 2761, 3777, 4084, 5097, 6659, 7944, 9126,
    1464, 3313, 4901, 6464, 9571, 11945, 14052, 14422, 17475,
    563, 1011, 2370, 3829, 4419, 5538, 6499, 7336, 8375,
    1408, 2022, 4209, 6067, 8236, 10522, 13272, 15087, 18047,
    99, 1206, 2306, 3855, 4167, 5985, 6659, 8043, 8687,
    1983, 2371, 5693, 6070, 9598, 11557, 13791, 14987, 16724,
    900, 1597, 2304, 3206, 4832, 5139, 6963, 7423, 8777,
    690, 2810, 5265, 7472, 9811, 10108, 13052, 14875, 16365,
    868, 1269, 2917, 3043, 4706, 5965, 7030, 7899, 8514,
    34, 3408, 5732, 6197, 9437, 11621, 13798, 15027, 17036,
    556, 1394, 2414, 3068, 4998, 5167, 6759, 7252, 9232,
    378, 3319, 4347, 6572, 8538, 11384, 13918, 15062, 17219,
    84, 1231, 2352, 3708, 4676, 5656, 6632, 7375, 8933,
    1857, 3273, 5582, 7802, 9237, 11663, 13446, 15783, 16791,
    605, 1696, 2966, 3203, 5050, 5895, 6566, 7960, 8867,
    440, 3381, 4409, 6762, 8092, 12011, 13696, 15770, 17524,
    1001, 1177, 3029, 3656, 4819, 5207, 6518, 7651, 8744,
    570, 2483, 4524, 6759, 9900, 11122, 13453, 15933, 16637,
    890, 1717, 2699, 3122, 5079, 5137, 6980, 8193, 8321,
    248, 2206, 4232, 7254, 8606, 10815, 13383, 14461, 18086,
};

std::vector<int32_t> token_map() {
    std::vector<int32_t> map(kVocab);
    for (int i = 0; i < kVocab; ++i) map[(size_t) i] = (i * 13 + 5) % 29 + 3;
    return map;
}

void running_sums(std::vector<uint64_t> & offsets, std::vector<uint64_t> & rows) {
    offsets.assign(kPrimes.size(), 0);
    rows.assign(kLayers, 0);
    for (int l = 0; l < kLayers; ++l) {
        for (int c = 0; c < kCols; ++c) {
            offsets[(size_t) (l * kCols + c)] = rows[(size_t) l];
            rows[(size_t) l] += kPrimes[(size_t) (l * kCols + c)];
        }
    }
}

bool make_hasher(DeepSeek4EngramHasher & hasher, std::string * err) {
    std::vector<uint64_t> offsets, rows;
    running_sums(offsets, rows);
    return hasher.init_raw({1, 14}, kHeads, kMaxNgram, kPad, token_map(),
                           kMultipliers, kPrimes, offsets, rows, err);
}

// Row bytes of the two synthetic table rows: 256 E4M3 codes (NaN codes
// replaced by 0x7e) then 8 E8M0 block scales.
std::vector<uint8_t> synthetic_row(int which) {
    static const uint8_t scales[2][8] = {
        {127, 120, 130, 0, 1, 254, 127, 100},
        {127, 126, 128, 140, 110, 127, 127, 90},
    };
    std::vector<uint8_t> row(DeepSeek4EngramTable::kRowBytes);
    for (int j = 0; j < DeepSeek4EngramTable::kDim; ++j) {
        uint8_t code = which == 0 ? (uint8_t) j : (uint8_t) ((j * 37 + 11) % 256);
        if ((code & 127) == 127) code = 0x7e;
        row[(size_t) j] = code;
    }
    std::memcpy(row.data() + DeepSeek4EngramTable::kDim, scales[which], 8);
    return row;
}

struct ExpectedValue { int index; uint32_t bits; };  // index into [2][256]
const ExpectedValue kExpectedValues[] = {
    {0, 0x00000000u},
    {17, 0x3d100000u},
    {34, 0x3aa00000u},
    {51, 0x3bb00000u},
    {68, 0x41c00000u},
    {85, 0x42d00000u},
    {102, 0x02e00000u},
    {119, 0x03f00000u},
    {136, 0x80020000u},
    {153, 0x80090000u},
    {170, 0xfe200000u},
    {187, 0xff300000u},
    {204, 0xc0c00000u},
    {221, 0xc1d00000u},
    {238, 0xb5600000u},
    {255, 0x36600000u},
    {272, 0x41b00000u},
    {289, 0xc0800000u},
    {306, 0x3fd00000u},
    {323, 0xc0200000u},
    {340, 0x3f700000u},
    {357, 0xc4c00000u},
    {374, 0x44100000u},
    {391, 0xb4600000u},
    {408, 0x33400000u},
    {425, 0x43800000u},
    {442, 0xc2d00000u},
    {459, 0x42200000u},
    {476, 0xc1700000u},
    {493, 0x2e400000u},
    {510, 0xad900000u},
};

uint32_t float_bits(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

struct TempTable {
    std::string path;
    uint64_t offset = 4096;   // the table does not start at byte 0 in a GGUF either
    explicit TempTable(const std::vector<std::vector<uint8_t>> & rows) {
        char name[] = "/tmp/ds4_engram_XXXXXX";
        const int fd = mkstemp(name);
        path = name;
        if (fd < 0) return;
        std::vector<uint8_t> header(offset, 0xAB);
        (void) !write(fd, header.data(), header.size());
        for (const auto & row : rows) (void) !write(fd, row.data(), row.size());
        close(fd);
    }
    ~TempTable() { if (!path.empty()) unlink(path.c_str()); }
};

}  // namespace

TEST_CASE(DeepSeek4EngramFixture, hash_matches_reference) {
    DeepSeek4EngramHasher hasher;
    std::string err;
    CHECK(make_hasher(hasher, &err));
    CHECK(hasher.cols() == kCols);
    CHECK(hasher.layer_index(14) == 1 && hasher.layer_index(2) == -1);

    std::vector<uint32_t> out(kTokens.size() * kLayers * kCols);
    DeepSeek4EngramHistory history;
    hasher.hash(history, kTokens.data(), nullptr, kTokens.size(), out.data());
    CHECK(out == kExpectedRows);
}

TEST_CASE(DeepSeek4EngramFixture, hash_carries_history_across_calls) {
    DeepSeek4EngramHasher hasher;
    CHECK(make_hasher(hasher, nullptr));
    std::vector<uint32_t> out(kTokens.size() * kLayers * kCols);
    DeepSeek4EngramHistory history;
    const size_t per_token = (size_t) kLayers * kCols;
    size_t done = 0;
    for (size_t chunk : {1u, 5u, 2u, 16u}) {
        hasher.hash(history, kTokens.data() + done, nullptr, chunk, out.data() + done * per_token);
        done += chunk;
    }
    CHECK(done == kTokens.size());
    CHECK(out == kExpectedRows);

    // A copied history is a snapshot: hashing from it again reproduces the rows.
    DeepSeek4EngramHistory start;
    std::vector<uint32_t> first(per_token), again(per_token);
    DeepSeek4EngramHistory snap = start;
    hasher.hash(start, kTokens.data(), nullptr, 1, first.data());
    hasher.hash(snap, kTokens.data(), nullptr, 1, again.data());
    CHECK(first == again);
}

TEST_CASE(DeepSeek4EngramFixture, dead_slot_blocks_longer_ngrams) {
    DeepSeek4EngramHasher hasher;
    CHECK(make_hasher(hasher, nullptr));
    // A dead position hashes like the start of a sequence for the token after it.
    const int32_t tokens[3] = {7, 0, 39};
    const uint8_t dead[3] = {0, 1, 0};
    std::vector<uint32_t> blocked(3 * kLayers * kCols), fresh(kLayers * kCols);
    DeepSeek4EngramHistory h1, h2;
    hasher.hash(h1, tokens, dead, 3, blocked.data());
    hasher.hash(h2, tokens + 2, nullptr, 1, fresh.data());
    CHECK(std::equal(fresh.begin(), fresh.end(), blocked.begin() + 2 * kLayers * kCols));
}

TEST_CASE(DeepSeek4EngramFixture, init_rejects_inconsistent_constants) {
    DeepSeek4EngramHasher hasher;
    std::vector<uint64_t> offsets, rows;
    running_sums(offsets, rows);
    std::string err;
    std::vector<uint64_t> even = kMultipliers;
    even[0] += 1;
    CHECK(!hasher.init_raw({1, 14}, kHeads, kMaxNgram, kPad, token_map(), even, kPrimes, offsets, rows, &err));
    std::vector<uint64_t> bad_rows = rows;
    bad_rows[1] += 1;
    CHECK(!hasher.init_raw({1, 14}, kHeads, kMaxNgram, kPad, token_map(), kMultipliers, kPrimes, offsets,
                           bad_rows, &err));
    CHECK(!hasher.present());
}

TEST_CASE(DeepSeek4EngramFixture, decode_matches_reference) {
    for (const ExpectedValue & e : kExpectedValues) {
        const std::vector<uint8_t> row = synthetic_row(e.index / DeepSeek4EngramTable::kDim);
        const int j = e.index % DeepSeek4EngramTable::kDim;
        const float v = deepseek4_engram_decode_value(row[(size_t) j], row[(size_t) (DeepSeek4EngramTable::kDim + j / 32)]);
        CHECK(float_bits(v) == e.bits);
    }
}

TEST_CASE(DeepSeek4EngramFixture, table_read_sorts_dedups_and_decodes) {
    const std::vector<uint8_t> r0 = synthetic_row(0), r1 = synthetic_row(1);
    TempTable file({r0, r1});
    DeepSeek4EngramTable table;
    std::string err;
    CHECK(table.open(file.path, file.offset, 2, &err));
    const uint32_t ids[4] = {1, 0, 1, 1};
    std::vector<float> vals(4 * DeepSeek4EngramTable::kDim);
    CHECK(table.read(ids, 4, vals.data(), 1));
    for (int slot = 0; slot < 4; ++slot) {
        const std::vector<uint8_t> & row = ids[slot] == 0 ? r0 : r1;
        for (int j = 0; j < DeepSeek4EngramTable::kDim; ++j) {
            const float want = deepseek4_engram_decode_value(row[(size_t) j], row[(size_t) (DeepSeek4EngramTable::kDim + j / 32)]);
            CHECK(float_bits(vals[(size_t) (slot * DeepSeek4EngramTable::kDim + j)]) == float_bits(want));
        }
    }
    const uint32_t out_of_range = 2;
    CHECK(!table.read(&out_of_range, 1, vals.data(), 1));
    CHECK(!table.open(file.path, file.offset, 3, &err));   // past the end of the file
}

TEST_CASE(DeepSeek4EngramFixture, table_read_rejects_nan_codes) {
    std::vector<uint8_t> row = synthetic_row(0);
    row[5] = 0x7f;
    TempTable file({row});
    DeepSeek4EngramTable table;
    CHECK(table.open(file.path, file.offset, 1, nullptr));
    std::vector<float> vals(DeepSeek4EngramTable::kDim);
    const uint32_t id = 0;
    CHECK(!table.read(&id, 1, vals.data(), 1));
}
