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

void store64(uint8_t *bytes, uint64_t value) {
    std::memcpy(bytes, &value, sizeof(value));
}

struct EncodeBitStream {
    uint8_t *begin = nullptr;
    uint8_t *current = nullptr;
    uint8_t *limit = nullptr;
    uint64_t head = 0;
    unsigned bitPosition = 0;
    uint8_t state1 = 0;
    uint8_t state2 = 0;

    void write(uint64_t value, unsigned count) {
        if (count > 32 || bitPosition + count > 63) {
            throw std::runtime_error("entropy stream overflow");
        }
        head |= value << bitPosition;
        bitPosition += count;
    }

    void flush() {
        if (current > limit) throw std::runtime_error("entropy stream overflow");
        store64(current, head);
        const size_t bytes = bitPosition >> 3U;
        if (bytes > static_cast<size_t>(limit - current)) {
            throw std::runtime_error("entropy stream overflow");
        }
        current += bytes;
        head >>= bitPosition & ~7U;
        bitPosition &= 7U;
    }

    size_t closeWithStates() {
        write(65536U | (static_cast<uint64_t>(state1) << 8U) | state2, 17);
        if (current > limit) throw std::runtime_error("entropy stream overflow");
        store64(current, head);
        const size_t size = static_cast<size_t>(current - begin) + ((bitPosition + 7U) >> 3U);
        if (size > static_cast<size_t>(limit - begin)) {
            throw std::runtime_error("entropy stream overflow");
        }
        return size;
    }
};

class Encoder {
public:
    explicit Encoder(size_t capacity) : memory_(capacity + 8, 0), capacity_(capacity) {
        if (capacity == 0) throw std::invalid_argument("empty entropy buffer");
        stream_.begin = memory_.data();
        stream_.current = memory_.data();
        stream_.limit = memory_.data() + capacity;
        int bits = 1;
        for (int probability = 255; probability > 0; --probability) {
            if ((probability << bits) < 256) ++bits;
            deltaBits_[probability] = static_cast<uint16_t>(
                (bits << 8) - (probability << bits) + 256
            );
        }
    }

    void setSGMTables(
        const uint32_t *transitions, const uint8_t *bounds,
        const uint8_t *stateMaps, size_t count
    ) {
        if (transitions == nullptr || bounds == nullptr || stateMaps == nullptr ||
            count == 0 || count > kDistributionCount) {
            throw std::invalid_argument("invalid SGM tables");
        }
        transitionCount_ = count;
        yTransitions_.assign(transitions, transitions + count * kStateCount);
        yStateMaps_.assign(stateMaps, stateMaps + count * kStateCount);
        std::copy(bounds, bounds + count, yBounds_.begin());
        for (size_t i = 0; i < count; ++i) {
            if (yBounds_[i] == 0 || yBounds_[i] > 128) {
                throw std::invalid_argument("invalid SGM bound");
            }
        }
    }

    void encodeSGM(
        const uint8_t *indexes, int16_t *values, const uint8_t *masks, size_t count
    ) {
        ensureOpen();
        if (indexes == nullptr || values == nullptr || masks == nullptr || transitionCount_ == 0) {
            throw std::invalid_argument("invalid SGM input");
        }
        for (size_t i = 0; i < count; ++i) {
            if (masks[i] && indexes[i] >= transitionCount_) {
                throw std::invalid_argument("SGM distribution index out of range");
            }
        }

        size_t index = count;
        if (index & 1U) {
            --index;
            if (masks[index]) {
                encodeYOutbound(indexes, values, index);
                encodeYInbound(stream_.state1, indexes, values, index);
            }
            stream_.flush();
        }
        if (index & 2U) {
            index -= 2;
            if (masks[index + 1]) {
                encodeYOutbound(indexes, values, index + 1);
                encodeYInbound(stream_.state2, indexes, values, index + 1);
            }
            if (masks[index]) {
                encodeYOutbound(indexes, values, index);
                encodeYInbound(stream_.state1, indexes, values, index);
            }
            stream_.flush();
        }
        while (index != 0) {
            index -= 4;
            for (size_t i = index + 4; i-- > index;) {
                if (masks[i]) encodeYOutbound(indexes, values, i);
            }
            if (masks[index + 3]) encodeYInbound(stream_.state2, indexes, values, index + 3);
            if (masks[index + 2]) encodeYInbound(stream_.state1, indexes, values, index + 2);
            if (masks[index + 1]) encodeYInbound(stream_.state2, indexes, values, index + 1);
            if (masks[index]) encodeYInbound(stream_.state1, indexes, values, index);
            stream_.flush();
        }
    }

    void encodeFactorized(
        const uint8_t *cdfs, uint8_t *values, size_t channels, size_t count
    ) {
        ensureOpen();
        if (cdfs == nullptr || values == nullptr || channels == 0 || count == 0) {
            throw std::invalid_argument("invalid factorized input");
        }
        for (size_t channel = channels; channel-- > 0;) {
            const uint8_t *cdf = cdfs + channel * kMaxZ;
            uint8_t previous = 0;
            for (size_t i = 0; i < kMaxZ; ++i) {
                if (cdf[i] < previous) throw std::invalid_argument("invalid factorized CDF");
                previous = cdf[i];
            }
            if (previous != 255) throw std::invalid_argument("incomplete factorized CDF");
            encodeFactorizedRow(cdf, values + channel * count, count);
        }
    }

    size_t finish() {
        ensureOpen();
        finished_ = true;
        return stream_.closeWithStates();
    }

    const uint8_t *bytes() const { return memory_.data(); }

private:
    void ensureOpen() const {
        if (finished_) throw std::runtime_error("entropy encoder is closed");
    }

    static void encodeWithTransition(
        EncodeBitStream &stream, uint8_t &state, uint32_t transition
    ) {
        const uint32_t bits = (state + (transition >> 16U)) >> 8U;
        const uint32_t mask = bits == 0 ? 0 : (1U << bits) - 1U;
        stream.write(state & mask, bits);
        state = static_cast<uint8_t>(((state | 256U) >> bits) + transition);
    }

    void encodeYInbound(
        uint8_t &state, const uint8_t *indexes, const int16_t *values, size_t index
    ) {
        const int value = values[index];
        if (value < -128 || value > 127) throw std::runtime_error("invalid SGM symbol");
        const size_t table = indexes[index] * kStateCount;
        encodeWithTransition(stream_, state, yTransitions_[table + value + 128]);
        state = yStateMaps_[table + state];
    }

    void encodeYOutbound(const uint8_t *indexes, int16_t *values, size_t index) {
        const int value = values[index];
        const int bound = yBounds_[indexes[index]];
        if (value >= bound || value <= -bound) {
            const uint64_t coded = value >= bound
                ? static_cast<uint64_t>(value - bound) << 1U
                : (static_cast<uint64_t>(-static_cast<int32_t>(value) - bound) << 1U) | 1U;
            stream_.head |= coded << stream_.bitPosition;
            if (coded >= 8) {
                stream_.bitPosition += 17;
            } else {
                stream_.head |= 1ULL << (stream_.bitPosition + 3U);
                stream_.bitPosition += 4;
            }
            stream_.flush();
            values[index] = static_cast<int16_t>(-bound);
        }
    }

    void setZDistribution(const uint8_t *cdf) {
        int cumulative = 0;
        for (size_t symbol = 0; symbol < kMaxZ; ++symbol) {
            const int probability = cdf[symbol] - cumulative;
            zTransitions_[symbol] = probability
                ? (static_cast<uint32_t>(deltaBits_[probability]) << 16U) |
                    static_cast<uint8_t>(cumulative - probability)
                : 0;
            cumulative = cdf[symbol];
        }
        zTransitions_[kMaxZ] = (static_cast<uint32_t>(deltaBits_[1]) << 16U) | 254U;
    }

    void encodeZInbound(uint8_t &state, uint8_t *values, size_t index) {
        encodeWithTransition(stream_, state, zTransitions_[values[index]]);
    }

    void encodeZOutbound(uint8_t *values, size_t index) {
        if (values[index] > kMaxZ) throw std::invalid_argument("factorized symbol out of range");
        if (zTransitions_[values[index]] == 0) {
            stream_.write(values[index], 6);
            values[index] = kMaxZ;
        }
    }

    void encodeFactorizedRow(const uint8_t *cdf, uint8_t *values, size_t count) {
        setZDistribution(cdf);
        size_t length = count;
        if (length & 1U) {
            --length;
            encodeZOutbound(values, length);
            encodeZInbound(stream_.state1, values, length);
        }
        if (length & 2U) {
            --length;
            encodeZOutbound(values, length);
            encodeZInbound(stream_.state2, values, length);
            --length;
            encodeZOutbound(values, length);
            encodeZInbound(stream_.state1, values, length);
        }
        stream_.flush();
        while (length != 0) {
            length -= 4;
            for (size_t i = length + 4; i-- > length;) encodeZOutbound(values, i);
            encodeZInbound(stream_.state2, values, length + 3);
            encodeZInbound(stream_.state1, values, length + 2);
            encodeZInbound(stream_.state2, values, length + 1);
            encodeZInbound(stream_.state1, values, length);
            stream_.flush();
        }
    }

    std::vector<uint8_t> memory_;
    size_t capacity_;
    EncodeBitStream stream_;
    bool finished_ = false;
    size_t transitionCount_ = 0;
    std::vector<uint32_t> yTransitions_;
    std::vector<uint8_t> yStateMaps_;
    std::array<uint8_t, kDistributionCount> yBounds_{};
    std::array<uint32_t, kMaxZ + 1> zTransitions_{};
    std::array<uint16_t, kStateCount> deltaBits_{};
};

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
struct JPEGAIANSEncoder { Encoder encoder; };

extern "C" JPEGAIANSEncoder *jpegai_ans_encoder_create(size_t capacity) {
    try {
        return new JPEGAIANSEncoder{Encoder(capacity)};
    } catch (...) {
        return nullptr;
    }
}

extern "C" void jpegai_ans_encoder_destroy(JPEGAIANSEncoder *encoder) { delete encoder; }

extern "C" int jpegai_ans_encoder_set_sgm_tables(
    JPEGAIANSEncoder *encoder, const uint32_t *transitions, const uint8_t *bounds,
    const uint8_t *stateMaps, size_t count
) {
    if (encoder == nullptr) return -1;
    try {
        encoder->encoder.setSGMTables(transitions, bounds, stateMaps, count);
        return 0;
    } catch (...) {
        return -1;
    }
}

extern "C" int jpegai_ans_encoder_encode_sgm(
    JPEGAIANSEncoder *encoder, const uint8_t *indexes, int16_t *values,
    const uint8_t *masks, size_t count
) {
    if (encoder == nullptr) return -1;
    try {
        encoder->encoder.encodeSGM(indexes, values, masks, count);
        return 0;
    } catch (...) {
        return -1;
    }
}

extern "C" int jpegai_ans_encoder_encode_factorized(
    JPEGAIANSEncoder *encoder, const uint8_t *cdfs, uint8_t *values,
    size_t channels, size_t valuesPerChannel
) {
    if (encoder == nullptr) return -1;
    try {
        encoder->encoder.encodeFactorized(cdfs, values, channels, valuesPerChannel);
        return 0;
    } catch (...) {
        return -1;
    }
}

extern "C" ptrdiff_t jpegai_ans_encoder_finish(
    JPEGAIANSEncoder *encoder, uint8_t *output, size_t outputCapacity
) {
    if (encoder == nullptr || output == nullptr) return -1;
    try {
        const size_t size = encoder->encoder.finish();
        if (size > outputCapacity || size > static_cast<size_t>(PTRDIFF_MAX)) return -1;
        std::memcpy(output, encoder->encoder.bytes(), size);
        return static_cast<ptrdiff_t>(size);
    } catch (...) {
        return -1;
    }
}

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
