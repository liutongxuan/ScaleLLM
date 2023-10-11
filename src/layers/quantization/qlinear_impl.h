#pragma once

#include <ATen/core/TensorBody.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "../linear_impl.h"
#include "../model_parallel.h"
#include "model_loader/state_dict.h"
#include "models/args.h"

namespace llm {

// Base QLinear class that handles quantized weights loading.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class ColumnParallelQLinearImpl : public ParallelLinearImpl {
 public:
  ColumnParallelQLinearImpl(int64_t in_features,
                            int64_t out_features,
                            bool bias,
                            int64_t bits,
                            int64_t group_size,
                            int64_t qweight_pack_dim,
                            bool gather_output,
                            const ParallelArgs& parallel_args,
                            torch::ScalarType dtype,
                            const torch::Device& device);

  // verify if the weight is loaded correctly
  void verify_loaded_weights(const std::string& prefix = "") const override;

  // all subclasses must implement this function
  virtual torch::Tensor quant_matmul(const torch::Tensor& input,
                                     const torch::Tensor& qweight,
                                     const torch::Tensor& qzeros_,
                                     const torch::Tensor& scales_) const = 0;

  torch::Tensor forward(torch::Tensor input) const override {
    auto output = quant_matmul(input, qweight_, qzeros_, scales_);
    if (bias_.defined()) {
      output.add_(bias_);
    }
    if (parallel_args_.world_size() > 1 && gather_output_) {
      output = gather_from_model_parallel_region(output, parallel_args_);
    }
    return output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) override;

  // special load_state_dict for fused cases
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string_view>& prefixes) override;

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " qweight=" << qweight_.sizes()
           << " qzeros=" << qzeros_.sizes() << " scales=" << scales_.sizes()
           << " device=" << qweight_.device();
  }

 private:
  // parameter members, must be registered
  torch::Tensor qweight_{nullptr};
  torch::Tensor qzeros_{nullptr};
  torch::Tensor scales_{nullptr};

  torch::Tensor bias_{nullptr};

  bool qweight_is_loaded_ = false;
  bool qzeros_is_loaded_ = false;
  bool scales_is_loaded_ = false;
  bool bias_is_loaded_ = false;
  std::vector<torch::Tensor> qweight_list_;
  std::vector<torch::Tensor> qzeros_list_;
  std::vector<torch::Tensor> scales_list_;
  std::vector<torch::Tensor> bias_list_;

  // whether to gather the output
  bool gather_output_;

  // parallel args
  ParallelArgs parallel_args_;

  // int rank_ = 0;
  // int world_size_ = 0;
};

// Base QLinear class that handles quantized weights loading.
//     The linear layer is defined as Y = XA + b. A is parallelized along
//     its first dimension and X along its second dimension as:
//                -   -
//               | A_1 |
//               | .   |
//           A = | .   |       X = [X_1, ..., X_p]
//               | .   |
//               | A_p |
//                -   -
class RowParallelQLinearImpl : public ParallelLinearImpl {
 public:
  RowParallelQLinearImpl(int64_t in_features,
                         int64_t out_features,
                         bool bias,
                         int64_t bits,
                         int64_t group_size,
                         int64_t qweight_pack_dim,
                         bool input_is_parallelized,
                         const ParallelArgs& parallel_args,
                         torch::ScalarType dtype,
                         const torch::Device& device);

  // all subclasses must implement this function
  virtual torch::Tensor quant_matmul(const torch::Tensor& input,
                                     const torch::Tensor& qweight,
                                     const torch::Tensor& qzeros_,
                                     const torch::Tensor& scales_) const = 0;

  torch::Tensor forward(torch::Tensor input) const override {
    if (!input_is_parallelized_) {
      input = scatter_to_model_parallel_region(input, parallel_args_);
    }

    auto output = quant_matmul(input, qweight_, qzeros_, scales_);
    if (bias_.defined()) {
      output.add_(bias_);
    }
    
    if (parallel_args_.world_size() > 1) {
      output = reduce_from_model_parallel_region(output, parallel_args_);
    }
    return output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) override;

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix = "") const override;

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " qweight=" << qweight_.sizes()
           << " qzeros=" << qzeros_.sizes() << " scales=" << scales_.sizes()
           << " device=" << qweight_.device();
  }

 private:
  // parameter members, must be registered
  torch::Tensor qweight_{nullptr};
  torch::Tensor qzeros_{nullptr};
  torch::Tensor scales_{nullptr};

  torch::Tensor bias_{nullptr};

  bool qweight_is_loaded_ = false;
  bool qzeros_is_loaded_ = false;
  bool scales_is_loaded_ = false;
  bool bias_is_loaded_ = false;

  // whether the input is already parallelized
  bool input_is_parallelized_;

  // parallel args
  ParallelArgs parallel_args_;
  // int rank_ = 0;
  // int world_size_ = 0;
};
}  // namespace llm
