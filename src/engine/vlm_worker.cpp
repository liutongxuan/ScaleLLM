#include "vlm_worker.h"

#include <glog/logging.h>
#include <memory>
#include <torch/torch.h>
#include <utility>

namespace llm {

VLMWorker::VLMWorker(const ParallelArgs& parallel_args,
                     const torch::Device& device,
                     const ModelRunner::Options& runner_options)
    : Worker(parallel_args, device, runner_options) {}

torch::Tensor VLMWorker::vision_encode(torch::Tensor image,
                                       torch::Tensor tokens) {
  auto model = dynamic_cast<CausalVLM*>(model_.get());
  if (model == nullptr) {
    LOG(FATAL) << "invalid causal vlm model";
  } else {
    return model->vision_encode(image, tokens);
  }
}

}  // namespace llm
