#include <absl/strings/str_split.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <gflags/gflags.h>
#include <pybind11/embed.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>

#include <glog/logging.h>
#include <absl/time/time.h>
#include "engine/llm_engine.h"
#include "engine/engine.h"
#include "request/sequence.h"
#include "request/stopping_criteria.h"
#include "sampling/parameters.h"

DEFINE_string(model_name_or_path,
              "/data/llama2-7b",
              "hf model name or path to the model file.");

DEFINE_string(input_file, "/data/dataset/Chatbot_group_10_2.json", "");

DEFINE_string(model_allow_patterns, "*", "Allow patterns for model files.");

DEFINE_string(device,
              "cuda",
              "Device to run the model on, e.g. cpu, cuda:0, cuda:0,cuda:1, or "
              "auto to use all available gpus.");

DEFINE_int32(batch_size, 1, "Batch size.");
DEFINE_int32(max_seq_len, 100, "Maximum sequence length.");

DEFINE_double(temperature, 1, "Temperature for sampling.");

DEFINE_double(top_p, 0.01, "Top p for sampling.");
DEFINE_int64(top_k, 0, "Top k for sampling.");

DEFINE_double(repetition_penalty, 1.0, "Repetition penalty for sampling.");

DEFINE_double(frequency_penalty, 0.0, "Frequency penalty for sampling.");
DEFINE_double(presence_penalty, 0.0, "Presence penalty for sampling.");

class JsonFileReader {
 public:
  static std::vector<std::string> read_json_input(const std::string& input_file) {
    std::ifstream file(input_file);
    if (file.fail()) {
      LOG(ERROR) << "Unable to open the json file:" << input_file;
    }
    nlohmann::json data = nlohmann::json::parse(file);

    std::vector<std::string> result;
    result.reserve(data.size());
    for (auto& it : data) {
      result.emplace_back(it["prompt"]);
    }
    return result;
  }
};

static constexpr double kMicrosToSeconds = 1000000.0;

class LLM {
 public:
  explicit LLM(const std::string model_path, const llm::SamplingParameter& sp,
               const llm::StoppingCriteria& sc, const std::string device_str) :
        sampling_param_(sp), stopping_criteria_(sc) {
    auto devices = parse_devices(device_str);
    llm::LLMEngine::Options options;
    options.devices(devices);
    engine_ = new llm::LLMEngine(options);
    CHECK(engine_->init(FLAGS_model_name_or_path));
    block_manager_ = engine_->block_manager();
    tokenizer_ = engine_->tokenizer();
  }

  void generate(const std::vector<std::string>& batched_prompt) {
    std::vector<llm::Sequence*> sequences;
    sequences.reserve(batched_prompt.size());

    llm::Sequence::Options options;
    options.sampling_param = sampling_param_;
    options.stopping_criteria = stopping_criteria_;
    options.echo = true;
    const size_t capacity = 100;

    for (size_t i = 0; i < batched_prompt.size(); ++i) {
      // create sequences
      std::vector<int> prompt_tokens;
      tokenizer_->encode(batched_prompt[i], &prompt_tokens);

      auto sequence = new llm::Sequence(batched_prompt[i], prompt_tokens, absl::Now(), capacity, options);
      sequences.emplace_back(sequence); 
    }

    absl::Duration request_cost;

    for (int64_t i = 0; i < FLAGS_max_seq_len; ++i) {
      sequences.erase(std::remove_if(sequences.begin(), sequences.end(),
                      [](llm::Sequence* seq) {
                        return seq->is_finished();
                      }), sequences.end());
      if (sequences.empty()) {
        break;
      }
      CHECK(block_manager_->allocate_blocks_for(sequences));

      llm::Batch batch(sequences);
      auto time_start = absl::Now();
      // run inference
      engine_->execute_model(batch);
      auto time_end = absl::Now();
      request_cost += time_end - time_start;

      // process sequence in batch
      for (int64_t i = 0; i < sequences.size(); ++i) {
        auto sequence = sequences[i];
	if (sequence->is_finished()) {
          block_manager_->release_blocks_for(sequence);
	}
        //std::cout << sequence->decode_delta_text(sequence->token_ids(), *tokenizer_)
        //          << std::flush;
      }
    }
    double cost = absl::ToInt64Microseconds(request_cost) / kMicrosToSeconds;
    std::cout << "request cost:" << cost << std::endl;
  }
 
 private:
  std::vector<torch::Device> parse_devices(const std::string& device_str) {
    std::vector<torch::Device> devices;
    if (device_str == "auto") {
      // use all available gpus if any
      const auto num_gpus = torch::cuda::device_count();
      if (num_gpus == 0) {
        LOG(INFO) << "no gpus found, using cpu.";
        return {torch::kCPU};
      }
      devices.reserve(num_gpus);
      for (int i = 0; i < num_gpus; ++i) {
        devices.emplace_back(torch::kCUDA, i);
      }
      return devices;
    }

    // parse device string
    const std::vector<std::string> device_strs = absl::StrSplit(device_str, ',');
    std::set<torch::DeviceType> device_types;
    devices.reserve(device_strs.size());
    for (const auto& device_str : device_strs) {
      devices.emplace_back(device_str);
      device_types.insert(devices.back().type());
    }
    CHECK(!devices.empty()) << "No devices specified.";
    CHECK(device_types.size() == 1)
        << "All devices must be of the same type. Got: " << FLAGS_device;
    return devices;
  }

 private:
  llm::LLMEngine* engine_;
  llm::BlockManager* block_manager_;
  llm::SamplingParameter sampling_param_;
  llm::StoppingCriteria stopping_criteria_;
  const llm::Tokenizer* tokenizer_;
};

int main(int argc, char* argv[]) {
  // initialize glog and gflags
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_minloglevel = google::INFO;
  
  llm::SamplingParameter sampling_param;
  sampling_param.temperature = 0;
  //sampling_param.temperature = FLAGS_temperature;
  //sampling_param.top_p = FLAGS_top_p;
  //sampling_param.top_k = FLAGS_top_k;
  //sampling_param.repetition_penalty = FLAGS_repetition_penalty;
  //sampling_param.frequency_penalty = FLAGS_frequency_penalty;
  //sampling_param.presence_penalty = FLAGS_presence_penalty;
  
  llm::StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = FLAGS_max_seq_len;
  stopping_criteria.ignore_eos = false;
  // stopping_criteria.eos_token_id = tokenizer->eos_id();

  LLM llm_engine(FLAGS_model_name_or_path, sampling_param,
                 stopping_criteria, FLAGS_device);

  auto input_prompts = JsonFileReader::read_json_input(FLAGS_input_file);
  auto loop_count = ceil(input_prompts.size() / FLAGS_batch_size);

  absl::Time time;
  double total_cost = 0.0;

  for (int i = 0; i < loop_count; ++i) {
    auto begin = input_prompts.begin() + i * FLAGS_batch_size;
    auto end = input_prompts.begin() + (i + 1) * FLAGS_batch_size;
    std::vector<std::string> batched_input_prompt(begin, end);

    auto time_start = absl::Now();
    llm_engine.generate(batched_input_prompt);
    auto time_end = absl::Now();

    auto duration = (time_end - time_start);
    double cost = absl::ToInt64Microseconds(duration) / kMicrosToSeconds;
    std::cout << "one_batch_cost:" << cost << std::endl;
    total_cost += cost;
  }
  std::cout << "average cost:" << total_cost / loop_count << std::endl;
  return 0;
}
