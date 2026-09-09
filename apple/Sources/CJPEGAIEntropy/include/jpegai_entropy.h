/*
 * The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2026, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *   be used to endorse or promote products derived from this software without
 *   specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef JPEG_AI_ENTROPY_H
#define JPEG_AI_ENTROPY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct JPEGAIANSDecoder JPEGAIANSDecoder;

JPEGAIANSDecoder *jpegai_ans_decoder_create(const uint8_t *bytes, size_t size);
void jpegai_ans_decoder_destroy(JPEGAIANSDecoder *decoder);

int jpegai_ans_decoder_set_sgm_tables(
    JPEGAIANSDecoder *decoder,
    const uint32_t *transitions,
    const uint8_t *bounds,
    size_t distribution_count
);

int jpegai_ans_decoder_decode_sgm(
    JPEGAIANSDecoder *decoder,
    const uint8_t *sigma_indexes,
    int16_t *values,
    const uint8_t *masks,
    size_t count
);

int jpegai_ans_decoder_decode_factorized(
    JPEGAIANSDecoder *decoder,
    const uint8_t *cdfs,
    uint8_t *values,
    size_t channels,
    size_t values_per_channel
);

#ifdef __cplusplus
}
#endif

#endif
