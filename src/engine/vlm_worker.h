#pragma once

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include "common/threadpool.h"
#include "model_loader/state_dict.h"
#include "model_parallel/parallel_args.h"
#include "model_runner.h"
#include "models/causal_vlm.h"
#include "models/model_args.h"
#include "models/parameters.h"
#include "parameters.h"
#include "quantization/quant_args.h"
#include "worker.h"

namespace llm {

class VLMWorker : public Worker {
 public:
  VLMWorker(const ParallelArgs& parallel_args,
            const torch::Device& device,
            const ModelRunner::Options& runner_options);

  ~VLMWorker() override = default;

  torch::Tensor vision_encode(torch::Tensor image, torch::Tensor tokens);
};

}  // namespace llm
