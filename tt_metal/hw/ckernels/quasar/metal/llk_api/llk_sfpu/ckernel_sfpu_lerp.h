// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_binary.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE>
inline void lerp_init() {}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_lerp(
    const uint dst_index_in0,  // input (start)
    const uint dst_index_in1,  // end
    const uint dst_index_in2,  // weight
    const uint dst_index_out) {
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "Unsupported data format for calculate_lerp(). Supported data formats are: Float32, Float16_b.");

    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    // lerp, evaluated the way torch.lerp is (aten/src/ATen/native/Lerp.h):
    //   |weight| <  0.5: out = input + weight * (end - input)
    //   |weight| >= 0.5: out = end - (1 - weight) * (end - input)
    // The second form is anchored at end, so weight = 1 returns end exactly; the input-anchored
    // form alone loses end whenever end - input rounds (#53279). It is written below as
    // end + (weight - 1) * (end - input), the same value with the sign folded into the factor.
    // A NaN weight fails the compare and stays on the first form, which is NaN as torch's is.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat in2 = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi];
        sfpi::vFloat diff = in1 - in0;
        v_if(sfpi::abs(in2) >= 0.5f) {
            in0 = in1;
            in2 = in2 - 1.0f;
        }
        v_endif;
        sfpi::vFloat result = in0 + in2 * diff;
        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
