#pragma once
#include <torch/all.h>

void cutlass_scaled_fp4_mm(torch::Tensor& D, torch::Tensor const& A,
                           torch::Tensor const& B, torch::Tensor const& A_sf,
                           torch::Tensor const& B_sf,
                           torch::Tensor const& alpha);