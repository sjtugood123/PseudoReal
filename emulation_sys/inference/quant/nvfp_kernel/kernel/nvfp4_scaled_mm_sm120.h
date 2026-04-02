#pragma once

#define ENABLE_NVFP4_SM120 1
#if defined ENABLE_NVFP4_SM100 && ENABLE_NVFP4_SM100
void cutlass_scaled_fp4_mm_sm100a(torch::Tensor& D, torch::Tensor const& A,
                                  torch::Tensor const& B,
                                  torch::Tensor const& A_sf,
                                  torch::Tensor const& B_sf,
                                  torch::Tensor const& alpha);
#endif

#if defined ENABLE_NVFP4_SM120 && ENABLE_NVFP4_SM120
void cutlass_scaled_fp4_mm_sm120a(torch::Tensor& D, torch::Tensor const& A,
                                  torch::Tensor const& B,
                                  torch::Tensor const& A_sf,
                                  torch::Tensor const& B_sf,
                                  torch::Tensor const& alpha);
#endif