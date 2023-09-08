#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <thread>

#include "engine/engine.h"
#include "grpc_server.h"
#include "scheduler/continuous_batching_scheduler.h"

using namespace llm;

DEFINE_string(model_path,
              "/home/michael/code/llama/llama-2-7b",
              "Path to the model file.");
DEFINE_string(tokenizer_path,
              "/home/michael/code/llama/tokenizer.model",
              "Path to the tokenizer file.");

DEFINE_string(device, "cuda:0", "Device to run the model on.");

int main(int argc, char** argv) {
  // glog and glfag will be initialized in folly::init
  folly::Init init(&argc, &argv);

  torch::Device device(FLAGS_device);

  // set the default dtype
  if (device.is_cpu()) {
    // always use float32 on CPU since float16 is not supported
    torch::set_default_dtype(
        torch::scalarTypeToTypeMeta(torch::ScalarType::Float));
    LOG(INFO) << "Using float32 on CPU.";
  } else {
    torch::set_default_dtype(
        torch::scalarTypeToTypeMeta(torch::ScalarType::BFloat16));
  }

  auto engine = std::make_unique<Engine>(std::vector<torch::Device>{device});
  CHECK(engine->init(FLAGS_model_path, FLAGS_tokenizer_path));

  auto scheduler = std::make_unique<ContinuousBatchingScheduler>(engine.get());
  const auto* tokenizer = engine->tokenizer();
  auto completion_handler =
      std::make_unique<CompletionHandler>(scheduler.get(), tokenizer);
  GrpcServer server(std::move(completion_handler));
  GrpcServer::Options options;
  options.address = "localhost";
  options.port = 8888;

  if (!server.start(options)) {
    LOG(ERROR) << "failed to start grpc server";
    return -1;
  }

  // TODO: add graceful shutdown
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    // TODO: update server status
  }
  server.stop();
  return 0;
}