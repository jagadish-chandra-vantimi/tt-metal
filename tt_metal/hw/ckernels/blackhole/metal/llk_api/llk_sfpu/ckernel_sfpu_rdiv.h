// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"
#include "ckernel_sfpu_rounding_ops.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, RoundingMode rounding_mode, int ITERATIONS>
inline void calculate_rdiv(const uint value) {
    sfpi::vFloat val = Converter::as_float(value);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        // For |in| >= 2^126 the reciprocal is below 2^-126 and flushes to zero, so the result
        // came out 0 whatever value was: rdiv(x, x) was 0, not 1 (#55128). Take the reciprocal
        // of in / 4 there and multiply by value / 4: the quotient is the same and both
        // scalings are exact. value / 4 only drops bits for |value| < 2^-124, where
        // |value / in| < 2^-250 flushes to zero anyway.
        sfpi::vFloat numerator = val;
        v_if(sfpi::exexp(in) >= 126) {
            in = in * 0.25f;
            numerator = val * 0.25f;
        }
        v_endif;
        sfpi::vFloat recip;
        if constexpr (APPROXIMATION_MODE) {
            recip = sfpu_reciprocal_iter<0>(in);
        } else {
            if constexpr (is_fp32_dest_acc_en) {
                recip = sfpu_reciprocal_iter<2>(in);
            } else {
                recip = sfpu_reciprocal_iter<1>(in);
                recip = sfpi::convert<sfpi::vFloat16b>(recip, sfpi::RoundMode::Nearest);
            }
        }
        sfpi::vFloat result = recip * numerator;

        if constexpr (rounding_mode == RoundingMode::Trunc) {
            result = _trunc_body_(result);
        } else if constexpr (rounding_mode == RoundingMode::Floor) {
            result = _floor_body_(result);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void rdiv_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
