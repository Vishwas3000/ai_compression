#include "jpegai_entropy.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

constexpr size_t kDistributionCount = 32;
constexpr size_t kStateCount = 256;
constexpr uint8_t kMaxZ = 63;
constexpr uint64_t kHighMasks[] = {
    0x00ffffffffffffffULL, 0x01ffffffffffffffULL,
    0x03ffffffffffffffULL, 0x07ffffffffffffffULL,
    0x0fffffffffffffffULL, 0x1fffffffffffffffULL,
    0x3fffffffffffffffULL, 0x7fffffffffffffffULL,
};

uint64_t load64(const uint8_t *bytes) {
    uint64_t value;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

struct BitStream {
    const uint8_t *begin = nullptr;
    const uint8_t *end = nullptr;
    uint64_t head = 0;
    unsigned bitPosition = 0;
    uint8_t state1 = 0;
    uint8_t state2 = 0;

    uint64_t read(unsigned count) {
        if (count > bitPosition) throw std::runtime_error("truncated entropy stream");
        bitPosition -= count;
        const uint64_t result = head >> bitPosition;
        head ^= result << bitPosition;
        return result;
    }

    void flush() {
        const size_t step = 7U ^ (bitPosition >> 3U);
        if (end < begin + step) throw std::runtime_error("truncated entropy stream");
        end -= step;
        head = load64(end) & kHighMasks[bitPosition & 7U];
        bitPosition |= 56U;
    }

    void initialize(const uint8_t *start, const uint8_t *finish) {
        begin = start;
        end = finish - 8;
        head = load64(end);
        if (head == 0) throw std::runtime_error("invalid entropy stream");
        bitPosition = 63U ^ static_cast<unsigned>(__builtin_clzll(head));
        head ^= 1ULL << bitPosition;
        state1 = static_cast<uint8_t>(read(8));
        state2 = static_cast<uint8_t>(read(8));
        flush();
    }
};

class Decoder {
public:
    Decoder(const uint8_t *bytes, size_t size) : memory_(size + 8, 0) {
        if (bytes == nullptr || size == 0) throw std::invalid_argument("empty entropy stream");
        std::memcpy(memory_.data() + 8, bytes, size);
        stream_.initialize(memory_.data(), memory_.data() + memory_.size());
        for (int i = 511, bits = 0; i > 0; --i) {
            if ((i << bits) < 256) ++bits;
            zPreprocess_[i] = static_cast<uint16_t>(((i << bits) ^ 256) | (bits << 8));
        }
    }

    void setSGMTables(const uint32_t *transitions, const uint8_t *bounds, size_t count) {
        if (transitions == nullptr || bounds == nullptr || count == 0 || count > kDistributionCount) {
            throw std::invalid_argument("invalid SGM tables");
        }
        transitionCount_ = count;
        yTransitions_.assign(transitions, transitions + count * kStateCount);
        std::copy(bounds, bounds + count, yBounds_.begin());
    }

    void decodeSGM(const uint8_t *indexes, int16_t *values, const uint8_t *masks, size_t count) {
        if (indexes == nullptr || values == nullptr || masks == nullptr || transitionCount_ == 0) {
            throw std::invalid_argument("invalid SGM input");
        }
        for (size_t i = 0; i < count; ++i) {
            if (masks[i] && indexes[i] >= transitionCount_) {
                throw std::invalid_argument("SGM distribution index out of range");
            }
        }

        size_t index = 0;
        while (index + 3 < count) {
            if (masks[index]) decodeYInbound(stream_.state1, indexes, values, index);
            if (masks[index + 1]) decodeYInbound(stream_.state2, indexes, values, index + 1);
            if (masks[index + 2]) decodeYInbound(stream_.state1, indexes, values, index + 2);
            if (masks[index + 3]) decodeYInbound(stream_.state2, indexes, values, index + 3);
            stream_.flush();
            for (size_t i = index; i < index + 4; ++i) {
                if (masks[i]) decodeYOutbound(indexes, values, i);
            }
            index += 4;
        }
        if (count & 2U) {
            if (masks[index]) {
                decodeYInbound(stream_.state1, indexes, values, index);
                decodeYOutbound(indexes, values, index);
            }
            if (masks[index + 1]) {
                decodeYInbound(stream_.state2, indexes, values, index + 1);
                decodeYOutbound(indexes, values, index + 1);
            }
            index += 2;
            stream_.flush();
        }
        if (count & 1U) {
            if (masks[index]) {
                decodeYInbound(stream_.state1, indexes, values, index);
                decodeYOutbound(indexes, values, index);
            }
            stream_.flush();
        }
    }

    void decodeFactorized(const uint8_t *cdfs, uint8_t *values, size_t channels, size_t count) {
        if (cdfs == nullptr || values == nullptr || channels == 0 || count == 0) {
            throw std::invalid_argument("invalid factorized input");
        }
        for (size_t channel = 0; channel < channels; ++channel) {
            const uint8_t *cdf = cdfs + channel * kMaxZ;
            uint8_t previous = 0;
            for (size_t i = 0; i < kMaxZ; ++i) {
                if (cdf[i] < previous) throw std::invalid_argument("invalid factorized CDF");
                previous = cdf[i];
            }
            if (previous != 255) throw std::invalid_argument("incomplete factorized CDF");
            decodeFactorizedRow(cdf, values + channel * count, count);
        }
    }

private:
    void decodeYInbound(uint8_t &state, const uint8_t *indexes, int16_t *values, size_t index) {
        const uint32_t transition = yTransitions_[indexes[index] * kStateCount + state];
        state = static_cast<uint8_t>((transition >> 16) | stream_.read(transition >> 24));
        values[index] = static_cast<int16_t>(transition);
    }

    void decodeYOutbound(const uint8_t *indexes, int16_t *values, size_t index) {
        const int bound = yBounds_[indexes[index]];
        if (!(values[index] + bound)) {
            const uint64_t temporary = stream_.read(stream_.read(1) ? 3 : 16);
            const int sign = temporary & 1;
            values[index] = static_cast<int16_t>(((temporary >> 1) + (bound - sign)) ^ -sign);
            stream_.flush();
        }
    }

    void setZDistribution(const uint8_t *cdf) {
        int current = 0;
        for (int symbol = 0; symbol < kMaxZ; ++symbol) {
            int probability = cdf[symbol] - current;
            for (int state = current; state < cdf[symbol]; ++state) {
                zTransitions_[state] = (static_cast<uint32_t>(zPreprocess_[probability++]) << 16) | symbol;
            }
            current = cdf[symbol];
        }
        zTransitions_[255] = (static_cast<uint32_t>(zPreprocess_[1]) << 16) | kMaxZ;
    }

    void decodeZInbound(uint8_t &state, uint8_t *values, size_t index) {
        const uint32_t transition = zTransitions_[state];
        state = static_cast<uint8_t>((transition >> 16) | stream_.read(transition >> 24));
        values[index] = static_cast<uint8_t>(transition);
    }

    void decodeZOutbound(uint8_t *values, size_t index) {
        if (values[index] == kMaxZ) values[index] = static_cast<uint8_t>(stream_.read(6));
    }

    void decodeFactorizedRow(const uint8_t *cdf, uint8_t *values, size_t count) {
        setZDistribution(cdf);
        size_t index = 0;
        while (index + 3 < count) {
            decodeZInbound(stream_.state1, values, index);
            decodeZInbound(stream_.state2, values, index + 1);
            decodeZInbound(stream_.state1, values, index + 2);
            decodeZInbound(stream_.state2, values, index + 3);
            for (size_t i = index; i < index + 4; ++i) decodeZOutbound(values, i);
            stream_.flush();
            index += 4;
        }
        if (count & 2U) {
            decodeZInbound(stream_.state1, values, index);
            decodeZOutbound(values, index);
            decodeZInbound(stream_.state2, values, index + 1);
            decodeZOutbound(values, index + 1);
            index += 2;
        }
        if (count & 1U) {
            decodeZInbound(stream_.state1, values, index);
            decodeZOutbound(values, index);
        }
        stream_.flush();
    }

    std::vector<uint8_t> memory_;
    BitStream stream_;
    size_t transitionCount_ = 0;
    std::vector<uint32_t> yTransitions_;
    std::array<uint8_t, kDistributionCount> yBounds_{};
    std::array<uint32_t, kStateCount> zTransitions_{};
    std::array<uint16_t, 512> zPreprocess_{};
};

} // namespace

struct JPEGAIANSDecoder { Decoder decoder; };

extern "C" JPEGAIANSDecoder *jpegai_ans_decoder_create(const uint8_t *bytes, size_t size) {
    try {
        return new JPEGAIANSDecoder{Decoder(bytes, size)};
    } catch (...) {
        return nullptr;
    }
}

extern "C" void jpegai_ans_decoder_destroy(JPEGAIANSDecoder *decoder) { delete decoder; }

extern "C" int jpegai_ans_decoder_set_sgm_tables(
    JPEGAIANSDecoder *decoder, const uint32_t *transitions, const uint8_t *bounds, size_t count
) {
    if (decoder == nullptr) return -1;
    try { decoder->decoder.setSGMTables(transitions, bounds, count); return 0; } catch (...) { return -1; }
}

extern "C" int jpegai_ans_decoder_decode_sgm(
    JPEGAIANSDecoder *decoder, const uint8_t *indexes, int16_t *values,
    const uint8_t *masks, size_t count
) {
    if (decoder == nullptr) return -1;
    try { decoder->decoder.decodeSGM(indexes, values, masks, count); return 0; } catch (...) { return -1; }
}

extern "C" int jpegai_ans_decoder_decode_factorized(
    JPEGAIANSDecoder *decoder, const uint8_t *cdfs, uint8_t *values,
    size_t channels, size_t valuesPerChannel
) {
    if (decoder == nullptr) return -1;
    try { decoder->decoder.decodeFactorized(cdfs, values, channels, valuesPerChannel); return 0; } catch (...) { return -1; }
}
