#include "block_allocator.h"

#include <glog/logging.h>

#include <cstdint>
#include <vector>

namespace llm {

// Memory block represents a contiguous memory region. It is used to track
// memory usage.

Block::Block(MemoryType type,
             uint32_t id,
             BlockAllocator* allocator)
    : type_(type),
      id_(id),
      ref_count_(new uint32_t(1)),
      allocator_(allocator) {}

Block::~Block() {
  // decrease reference count and free memory if necessary
  if (ref_count_ != nullptr && --(*ref_count_) == 0) {
    delete ref_count_;
    allocator_->free(*this);
  }
}

// copy constructor
Block::Block(const Block& other)
    : type_(other.type_),
      id_(other.id_),
      ref_count_(other.ref_count_),
      allocator_(other.allocator_) {
  // increase reference count
  ++(*ref_count_);
}

// copy assignment
Block& Block::operator=(const Block& other) {
  if (this != &other) {
    if (ref_count_ != nullptr && --(*ref_count_) == 0) {
      delete ref_count_;
      allocator_->free(*this);
    }
    type_ = other.type_;
    id_ = other.id_;
    allocator_ = other.allocator_;
    ref_count_ = other.ref_count_;
    ++(*ref_count_);
  }
  return *this;
}

Block::Block(Block&& other) noexcept
    : type_(other.type_),
      id_(other.id_),
      ref_count_(other.ref_count_),
      allocator_(other.allocator_) {
  // reset other
  other.ref_count_ = nullptr;
  other.allocator_ = nullptr;
}

Block& Block::operator=(Block&& other) noexcept {
  if (this != &other) {
    // decrease reference count and free memory if necessary
    if (ref_count_ != nullptr && --(*ref_count_) == 0) {
      delete ref_count_;
      allocator_->free(*this);
    }

    type_ = other.type_;
    id_ = other.id_;
    allocator_ = other.allocator_;
    ref_count_ = other.ref_count_;

    other.ref_count_ = nullptr;
    other.allocator_ = nullptr;
  }
  return *this;
}

uint32_t Block::size() const {
  return allocator_ == nullptr ? 0 : allocator_->block_size_in_bytes();
}

// BlockAllocator is used to allocate memory blocks. It is not thread safe.
BlockAllocator::BlockAllocator(uint32_t num_cpu_blocks,
                               uint32_t num_gpu_blocks,
                               uint32_t block_size_in_bytes)
    : cpu_block_count_(num_cpu_blocks),
      gpu_block_count_(num_gpu_blocks),
      block_size_in_bytes_(block_size_in_bytes) {
  free_cpu_blocks_.reserve(cpu_block_count_);
  for (uint32_t i = 0; i < cpu_block_count_; ++i) {
    // push blocks in reverse order so that smaller block ids are allocated
    // first
    free_cpu_blocks_.push_back(cpu_block_count_ - i - 1);
  }

  free_gpu_blocks_.reserve(gpu_block_count_);
  for (uint32_t i = 0; i < gpu_block_count_; ++i) {
    // push blocks in reverse order so that smaller block ids are allocated
    // first
    free_gpu_blocks_.push_back(gpu_block_count_ - i - 1);
  }
}

// allocate a block of memory
Block BlockAllocator::allocate(MemoryType type) {
  uint32_t block_id = 0;
  if (type == MemoryType::kCPU) {
    CHECK(cpu_block_count_ > 0) << "No more CPU memory blocks available";
    block_id = free_cpu_blocks_[--cpu_block_count_];
  } else if (type == MemoryType::kGPU) {
    CHECK(gpu_block_count_ > 0) << "No more GPU memory blocks available";
    block_id = free_gpu_blocks_[--gpu_block_count_];
  }
  return {type, block_id, this};
}

// free a block of memory, should only be called by Block destructor implicitly
void BlockAllocator::free(const Block& block) {
  CHECK(block.allocator_ == this) << "Block does not belong to this allocator";

  if (block.memory_type() == MemoryType::kCPU) {
    CHECK(cpu_block_count_ < free_cpu_blocks_.size());
    free_cpu_blocks_[cpu_block_count_++] = block.id();
  } else if (block.memory_type() == MemoryType::kGPU) {
    CHECK(gpu_block_count_ < free_gpu_blocks_.size());
    free_gpu_blocks_[gpu_block_count_++] = block.id();
  }
}

}  // namespace llm
