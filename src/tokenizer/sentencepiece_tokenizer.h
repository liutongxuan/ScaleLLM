#pragma once

#include <sentencepiece_processor.h>

#include "tokenizer.h"

namespace llm {

// a tokenizer that uses google/SentencePiece
class SentencePieceTokenizer : public Tokenizer {
 public:
  explicit SentencePieceTokenizer(const std::string& model_path);

  std::vector<int> encode(const std::string_view& text) const override;

  std::string decode(const std::vector<int>& ids) const override;

  int n_words() const { return sp_processor_.GetPieceSize(); }

  int unk_id() const { return sp_processor_.unk_id(); }

  int bos_id() const { return sp_processor_.bos_id(); }

  int eos_id() const { return sp_processor_.eos_id(); }

  int pad_id() const { return sp_processor_.pad_id(); }

 private:
  sentencepiece::SentencePieceProcessor sp_processor_;
};

}  // namespace llm