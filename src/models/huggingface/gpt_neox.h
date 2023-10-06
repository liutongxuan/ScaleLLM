#pragma once

#include <torch/torch.h>

#include "layers/activation.h"
#include "layers/attention.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/normalization.h"
#include "layers/pos_embedding.h"
#include "memory/kv_cache.h"
#include "models/args.h"
#include "models/input_parameters.h"

// gpt-neox model compatible with huggingface weights

namespace llm::hf {

class GPTNeoXMLPImpl : public torch::nn::Module {
 public:
  GPTNeoXMLPImpl(const ModelArgs& args,
                 const QuantizationArgs& quant_args,
                 const ParallelArgs& parallel_args,
                 torch::ScalarType dtype,
                 const torch::Device& device) {
    const int64_t hidden_size = args.hidden_size();
    const int64_t intermediate_size = args.intermediate_size();

    act_ = Activation::get(args.hidden_act(), device);
    CHECK(act_ != nullptr);

    // register the weight parameter
    dense_h_to_4h_ =
        register_module("dense_h_to_4h",
                        ColumnParallelLinear(hidden_size,
                                             intermediate_size,
                                             /*bias=*/true,
                                             /*gather_output=*/false,
                                             quant_args,
                                             parallel_args,
                                             dtype,
                                             device));
    dense_4h_to_h_ =
        register_module("dense_4h_to_h",
                        RowParallelLinear(intermediate_size,
                                          hidden_size,
                                          /*bias=*/true,
                                          /*input_is_parallelized=*/true,
                                          quant_args,
                                          parallel_args,
                                          dtype,
                                          device));
  }

  torch::Tensor forward(torch::Tensor x) {
    return dense_4h_to_h_(act_(dense_h_to_4h_(x)));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    dense_h_to_4h_->load_state_dict(state_dict.select("dense_h_to_4h."));
    dense_4h_to_h_->load_state_dict(state_dict.select("dense_4h_to_h."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    dense_h_to_4h_->verify_loaded_weights(prefix + "dense_h_to_4h.");
    dense_4h_to_h_->verify_loaded_weights(prefix + "dense_4h_to_h.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear dense_h_to_4h_{nullptr};
  RowParallelLinear dense_4h_to_h_{nullptr};

  ActFunc act_{nullptr};
};
TORCH_MODULE(GPTNeoXMLP);

class GPTNeoXAttentionImpl : public torch::nn::Module {
 public:
  GPTNeoXAttentionImpl(uint32_t layer_id,
                       const ModelArgs& args,
                       const QuantizationArgs& quant_args,
                       const ParallelArgs& parallel_args,
                       torch::ScalarType dtype,
                       const torch::Device& device)
      : layer_id_(layer_id), parallel_args_(parallel_args) {
    const auto world_size = parallel_args.world_size();
    const int64_t n_local_heads = args.n_heads() / world_size;
    hidden_size_ = args.hidden_size();
    head_dim_ = args.hidden_size() / args.n_heads();

    // register submodules
    query_key_value_ =
        register_module("query_key_value",
                        ColumnParallelLinear(hidden_size_,
                                             3 * hidden_size_,
                                             /*bias=*/true,
                                             /*gather_output=*/false,
                                             quant_args,
                                             parallel_args,
                                             dtype,
                                             device));

    dense_ = register_module("dense",
                             RowParallelLinear(hidden_size_,
                                               hidden_size_,
                                               /*bias=*/true,
                                               /*input_is_parallelized=*/true,
                                               quant_args,
                                               parallel_args,
                                               dtype,
                                               device));

    // initialize positional embedding
    const int64_t rotary_dim =
        static_cast<int64_t>(head_dim_ * args.rotary_pct());
    // initialize attention
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    atten_ = register_module("atten",
                             AttentionWithRoPE(n_local_heads,
                                               n_local_heads,
                                               head_dim_,
                                               scale,
                                               rotary_dim,
                                               args.rope_scaling(),
                                               args.rope_theta(),
                                               args.max_position_embeddings(),
                                               /*interleaved=*/false,
                                               dtype,
                                               device));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    // (num_tokens, dim) x (dim, n_heads * head_dim)
    // => (num_tokens, n_heads * head_dim)
    auto qkv = query_key_value_(x).chunk(/*chunks=*/3, /*dim=*/-1);
    DCHECK_EQ(qkv.size(), 3);
    // calculate attention, output: (num_tokens, n_local_heads * head_dim)
    auto output =
        atten_(qkv[0], qkv[1], qkv[2], positions, kv_cache, input_params);
    return dense_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    query_key_value_->load_state_dict(
        state_dict.select("query_key_value."),
        [this](const torch::Tensor& tensor) { return reshape_qkv_tensor(tensor); });
    dense_->load_state_dict(state_dict.select("dense."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    query_key_value_->verify_loaded_weights(prefix + "query_key_value.");
    dense_->verify_loaded_weights(prefix + "dense.");
  }

 private:
  torch::Tensor reshape_qkv_tensor(const torch::Tensor& tensor) {
    // N.B. Fused qkv weights in GPT-NeoX has the shape of [n_heads * 3 *
    // head_dim, hidden_size], while the desired shape is [3 * n_heads *
    // head_dim, hidden_size].
    CHECK(tensor.dim() == 2 || tensor.dim() == 1)
        << "unexpected tensor dim: " << tensor.dim();
    if (tensor.dim() == 2) {
      return tensor.view({-1, 3, head_dim_, hidden_size_})
          .permute({1, 0, 2, 3})
          .reshape({-1, hidden_size_});
    }
    return tensor.view({-1, 3, head_dim_}).permute({1, 0, 2}).reshape({-1});
  }

  // parameter members, must be registered
  ColumnParallelLinear query_key_value_{nullptr};

  RowParallelLinear dense_{nullptr};

  // module members without parameters
  AttentionWithRoPE atten_{nullptr};

  uint32_t layer_id_;

  ParallelArgs parallel_args_;

  int64_t hidden_size_ = 0;
  int64_t head_dim_ = 0;
};
TORCH_MODULE(GPTNeoXAttention);

class GPTNeoXLayerImpl : public torch::nn::Module {
 public:
  GPTNeoXLayerImpl(uint32_t layer_id,
                   const ModelArgs& args,
                   const QuantizationArgs& quant_args,
                   const ParallelArgs& parallel_args,
                   torch::ScalarType dtype,
                   const torch::Device& device)
      : layer_id_(layer_id),
        use_parallel_residual_(args.use_parallel_residual()),
        parallel_args_(parallel_args) {
    // register submodules
    attention_ = register_module(
        "attention",
        GPTNeoXAttention(
            layer_id, args, quant_args, parallel_args, dtype, device));
    mlp_ = register_module(
        "mlp", GPTNeoXMLP(args, quant_args, parallel_args, dtype, device));
    input_layernorm_ = register_module("input_layernorm",
                                       LayerNorm(args.hidden_size(),
                                                 args.layer_norm_eps(),
                                                 /*bias=*/true,
                                                 dtype,
                                                 device));
    post_attention_layernorm_ = register_module("post_attention_layernorm",
                                                LayerNorm(args.hidden_size(),
                                                          args.layer_norm_eps(),
                                                          /*bias=*/true,
                                                          dtype,
                                                          device));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    auto attn_output =
        attention_(input_layernorm_(x), positions, kv_cache, input_params);

    if (use_parallel_residual_) {
      // parallel residual: x = x + attn(ln1(x)) + mlp(ln2(x))
      return x + attn_output + mlp_(post_attention_layernorm_(x));
    }

    // x = x + attn(ln1(x))
    // x = x + mlp(ln2(x))
    auto h = x + attn_output;
    return h + mlp_(post_attention_layernorm_(h));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    attention_->load_state_dict(state_dict.select("attention."));
    mlp_->load_state_dict(state_dict.select("mlp."));
    input_layernorm_->load_state_dict(state_dict.select("input_layernorm."));
    post_attention_layernorm_->load_state_dict(
        state_dict.select("post_attention_layernorm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    attention_->verify_loaded_weights(prefix + "attention.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
    input_layernorm_->verify_loaded_weights(prefix + "input_layernorm.");
    post_attention_layernorm_->verify_loaded_weights(
        prefix + "post_attention_layernorm.");
  }

 private:
  // parameter members, must be registered
  GPTNeoXAttention attention_{nullptr};

  GPTNeoXMLP mlp_{nullptr};

  LayerNorm input_layernorm_{nullptr};

  LayerNorm post_attention_layernorm_{nullptr};

  uint32_t layer_id_;

  bool use_parallel_residual_;

  ParallelArgs parallel_args_;
};
TORCH_MODULE(GPTNeoXLayer);

class GPTNeoXModelImpl : public torch::nn::Module {
 public:
  GPTNeoXModelImpl(const ModelArgs& args,
                   const QuantizationArgs& quant_args,
                   const ParallelArgs& parallel_args,
                   torch::ScalarType dtype,
                   const torch::Device& device)
      : parallel_args_(parallel_args) {
    // register submodules
    embed_in_ = register_module("embed_in",
                                ParallelEmbedding(args.vocab_size(),
                                                  args.hidden_size(),
                                                  parallel_args,
                                                  dtype,
                                                  device));
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block =
          GPTNeoXLayer(i, args, quant_args, parallel_args, dtype, device);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    final_layer_norm_ = register_module("final_layer_norm",
                                        LayerNorm(args.hidden_size(),
                                                  args.layer_norm_eps(),
                                                  /*bias=*/true,
                                                  dtype,
                                                  device));

    embed_out_ = register_module("embed_out",
                                 ColumnParallelLinear(args.hidden_size(),
                                                      args.vocab_size(),
                                                      /*bias=*/false,
                                                      /*gather_output=*/true,
                                                      parallel_args,
                                                      dtype,
                                                      device));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h = embed_in_(tokens);
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, kv_caches[i], input_params);
    }
    h = final_layer_norm_(h);
    // select last token for each sequence
    h = h.index_select(/*dim=*/0, input_params.last_token_indicies);
    return embed_out_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    embed_in_->load_state_dict(state_dict.select("gpt_neox.embed_in."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.select("gpt_neox.layers." + std::to_string(i) + "."));
    }
    final_layer_norm_->load_state_dict(
        state_dict.select("gpt_neox.final_layer_norm."));
    embed_out_->load_state_dict(state_dict.select("embed_out."));
  }

  void verify_loaded_weights() const {
    embed_in_->verify_loaded_weights("gpt_neox.embed_in.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights("gpt_neox.layers." + std::to_string(i) +
                                        ".");
    }
    final_layer_norm_->verify_loaded_weights("gpt_neox.final_layer_norm.");
    embed_out_->verify_loaded_weights("embed_out.");
  }

 private:
  // parameter members, must be registered
  ParallelEmbedding embed_in_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<GPTNeoXLayer> layers_;

  LayerNorm final_layer_norm_{nullptr};

  ColumnParallelLinear embed_out_{nullptr};

  ParallelArgs parallel_args_;
};
TORCH_MODULE(GPTNeoXModel);

}  // namespace llm::hf