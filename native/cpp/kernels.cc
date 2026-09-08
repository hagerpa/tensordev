#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace tensordev_native_cpu {

constexpr int64_t kMinParallelScalarWork = 32768;
constexpr int64_t kTargetScalarWorkPerWorker = 65536;

bool CheckedMultiply(int64_t left, int64_t right, int64_t* product) {
  if (left < 0 || right < 0) return false;
  if (left != 0 && right > std::numeric_limits<int64_t>::max() / left) {
    return false;
  }
  *product = left * right;
  return true;
}

bool CheckedAdd(int64_t left, int64_t right, int64_t* sum) {
  if (left < 0 || right < 0 ||
      right > std::numeric_limits<int64_t>::max() - left) {
    return false;
  }
  *sum = left + right;
  return true;
}

template <typename Dimensions>
bool CheckedLeadingProduct(Dimensions dimensions, size_t trailing_rank,
                           int64_t* product) {
  if (dimensions.size() < trailing_rank) return false;
  int64_t value = 1;
  const size_t leading_rank = dimensions.size() - trailing_rank;
  for (size_t axis = 0; axis < leading_rank; ++axis) {
    if (!CheckedMultiply(value, dimensions[axis], &value)) return false;
  }
  *product = value;
  return true;
}

template <typename LeftDimensions, typename RightDimensions>
bool SameLeadingDimensions(LeftDimensions left, size_t left_trailing_rank,
                           RightDimensions right,
                           size_t right_trailing_rank) {
  if (left.size() < left_trailing_rank ||
      right.size() < right_trailing_rank) {
    return false;
  }
  const size_t left_rank = left.size() - left_trailing_rank;
  const size_t right_rank = right.size() - right_trailing_rank;
  if (left_rank != right_rank) return false;
  for (size_t axis = 0; axis < left_rank; ++axis) {
    if (left[axis] != right[axis]) return false;
  }
  return true;
}

int64_t MapBatchIndex(int64_t output_index,
                      const std::vector<int64_t>& output_strides,
                      const std::vector<int64_t>& operand_strides) {
  int64_t operand_index = 0;
  int64_t remainder = output_index;
  for (size_t axis = 0; axis < output_strides.size(); ++axis) {
    const int64_t coordinate = remainder / output_strides[axis];
    remainder -= coordinate * output_strides[axis];
    operand_index += coordinate * operand_strides[axis];
  }
  return operand_index;
}

template <typename Dimensions>
bool BuildOutputStrides(Dimensions output_dimensions,
                        size_t output_trailing_rank,
                        std::vector<int64_t>* strides,
                        std::string* error) {
  if (output_dimensions.size() < output_trailing_rank) {
    *error = "output rank is smaller than its logical trailing rank";
    return false;
  }
  const size_t batch_rank =
      output_dimensions.size() - output_trailing_rank;
  strides->assign(batch_rank, 0);
  int64_t stride = 1;
  for (size_t reverse = 0; reverse < batch_rank; ++reverse) {
    const size_t axis = batch_rank - reverse - 1;
    (*strides)[axis] = stride;
    if (!CheckedMultiply(stride, output_dimensions[axis], &stride)) {
      *error = "output batch shape exceeds int64 capacity";
      return false;
    }
  }
  return true;
}

template <typename OperandDimensions, typename OutputDimensions>
bool BuildOperandStrides(OperandDimensions operand_dimensions,
                         size_t operand_trailing_rank,
                         OutputDimensions output_dimensions,
                         size_t output_trailing_rank,
                         std::vector<int64_t>* aligned_strides,
                         std::string* error) {
  if (operand_dimensions.size() < operand_trailing_rank ||
      output_dimensions.size() < output_trailing_rank) {
    *error = "invalid operand rank in batch broadcast";
    return false;
  }
  const size_t operand_rank =
      operand_dimensions.size() - operand_trailing_rank;
  const size_t output_rank =
      output_dimensions.size() - output_trailing_rank;
  if (operand_rank > output_rank) {
    *error = "operand has more batch axes than the output";
    return false;
  }

  std::vector<int64_t> operand_strides(operand_rank, 0);
  int64_t stride = 1;
  for (size_t reverse = 0; reverse < operand_rank; ++reverse) {
    const size_t axis = operand_rank - reverse - 1;
    operand_strides[axis] = stride;
    if (!CheckedMultiply(stride, operand_dimensions[axis], &stride)) {
      *error = "operand batch shape exceeds int64 capacity";
      return false;
    }
  }

  aligned_strides->assign(output_rank, 0);
  const size_t offset = output_rank - operand_rank;
  for (size_t axis = 0; axis < operand_rank; ++axis) {
    const int64_t operand_size = operand_dimensions[axis];
    const int64_t output_size = output_dimensions[offset + axis];
    if (operand_size == 1) {
      (*aligned_strides)[offset + axis] = 0;
    } else if (operand_size == output_size) {
      (*aligned_strides)[offset + axis] = operand_strides[axis];
    } else {
      *error = "operand batch shape is not broadcastable to the output";
      return false;
    }
  }
  return true;
}

template <int64_t alphabet_size>
inline void DecodeSelectedEdge(int64_t edge, int64_t source_count,
                               int64_t* letter, int64_t* source_rank) {
  if constexpr (alphabet_size <= 2 && alphabet_size != 0) {
    *letter = 0;
    *source_rank = edge;
  } else if constexpr (alphabet_size == 3) {
    *letter = edge >= source_count;
    *source_rank = edge - *letter * source_count;
  } else if constexpr (alphabet_size == 4) {
    *letter = edge >= 2 * source_count ? 2 : edge >= source_count;
    *source_rank = edge - *letter * source_count;
  } else {
    *letter = edge / source_count;
    *source_rank = edge - *letter * source_count;
  }
}

ffi::Future CompletedFuture(ffi::Error error = ffi::Error::Success()) {
  ffi::Promise promise;
  ffi::Future future(promise);
  if (error.success()) {
    promise.SetAvailable();
  } else {
    promise.SetError(std::move(error));
  }
  return future;
}

ffi::Future InvalidArgument(std::string message) {
  return CompletedFuture(ffi::Error::InvalidArgument(std::move(message)));
}

constexpr int64_t kRaggedMetadataFields = 16;
constexpr int64_t kMaxRaggedGradeCount = 4096;

enum RaggedMetadataField : int64_t {
  kTotal = 0,
  kWidth = 1,
  kRankCount = 2,
  kDenseWidth = 3,
  kPrimeSource = 4,
  kDoubleSource = 5,
  kDoubleRankCount = 6,
  kDoubleSourceRankCount = 7,
  kSelectedOffset = 8,
  kSelectedCount = 9,
  kCollisionOffset = 10,
  kCollisionCount = 11,
  kTailPrimaryCount = 12,
  kMaxOrder = 13,
  kPrimeAlphabetSize = 14,
  kDoubleAlphabetSize = 15,
};

bool ValidateRaggedMetadata(const int32_t* metadata, int64_t grade_count,
                            int64_t selected_count,
                            int64_t collision_count,
                            int64_t generator_width,
                            std::string* error) {
  const int64_t prime_alphabet_size = metadata[kPrimeAlphabetSize];
  const int64_t double_alphabet_size = metadata[kDoubleAlphabetSize];
  const int64_t max_order = metadata[kMaxOrder];
  if (prime_alphabet_size <= 0 || double_alphabet_size <= 0 ||
      max_order < 0 ||
      prime_alphabet_size + double_alphabet_size != generator_width) {
    *error = "invalid ragged Horner global dimensions";
    return false;
  }

  int64_t total_width = 0;
  for (int64_t grade = 0; grade < grade_count; ++grade) {
    const int32_t* row = metadata + grade * kRaggedMetadataFields;
    if (row[kPrimeAlphabetSize] != prime_alphabet_size ||
        row[kDoubleAlphabetSize] != double_alphabet_size ||
        row[kMaxOrder] != max_order || row[kTotal] < 0 ||
        row[kTotal] > max_order || row[kWidth] <= 0 ||
        row[kRankCount] <= 0 || row[kDenseWidth] <= 0) {
      *error = "invalid ragged Horner grade metadata";
      return false;
    }

    int64_t expected_width;
    if (!CheckedMultiply(row[kRankCount], row[kDenseWidth],
                         &expected_width) ||
        expected_width != row[kWidth] ||
        !CheckedAdd(total_width, row[kWidth], &total_width)) {
      *error = "inconsistent or overflowing ragged grade width";
      return false;
    }

    const int64_t prime_source = row[kPrimeSource];
    const int64_t double_source = row[kDoubleSource];
    if ((prime_source < -1 || prime_source >= grade) ||
        (double_source < -1 || double_source >= grade) ||
        (grade > 0 && prime_source < 0 && double_source < 0)) {
      *error = "invalid ragged Horner predecessor index";
      return false;
    }

    int64_t expected_rank_count = 0;
    if (double_source >= 0) {
      const int32_t* source_row =
          metadata + double_source * kRaggedMetadataFields;
      int64_t edge_count;
      if (source_row[kTotal] + 1 != row[kTotal] ||
          row[kDoubleSourceRankCount] != source_row[kRankCount] ||
          row[kDenseWidth] != source_row[kDenseWidth] ||
          row[kDoubleRankCount] < row[kDoubleSourceRankCount] ||
          !CheckedMultiply(double_alphabet_size,
                           row[kDoubleSourceRankCount], &edge_count) ||
          row[kSelectedCount] !=
              edge_count - row[kDoubleSourceRankCount] ||
          row[kCollisionCount] != edge_count - row[kDoubleRankCount] ||
          row[kTailPrimaryCount] !=
              row[kDoubleRankCount] - row[kDoubleSourceRankCount]) {
        *error = "inconsistent double-prime Horner metadata";
        return false;
      }
      if (row[kSelectedOffset] < 0 || row[kCollisionOffset] < 0 ||
          row[kSelectedCount] < 0 || row[kCollisionCount] < 0 ||
          row[kSelectedOffset] >
              selected_count - row[kSelectedCount] ||
          row[kCollisionOffset] >
              collision_count - row[kCollisionCount]) {
        *error = "ragged Horner plan slice lies outside its buffer";
        return false;
      }
      expected_rank_count += row[kDoubleRankCount];
    } else if (row[kSelectedCount] != 0 || row[kCollisionCount] != 0 ||
               row[kTailPrimaryCount] != 0) {
      *error = "double-prime plan supplied without a predecessor";
      return false;
    }

    if (prime_source >= 0) {
      const int32_t* source_row =
          metadata + prime_source * kRaggedMetadataFields;
      int64_t expected_dense_width;
      if (source_row[kTotal] + 1 != row[kTotal] ||
          !CheckedMultiply(source_row[kDenseWidth], prime_alphabet_size,
                           &expected_dense_width) ||
          expected_dense_width != row[kDenseWidth]) {
        *error = "inconsistent prime Horner metadata";
        return false;
      }
      expected_rank_count += source_row[kRankCount];
    }
    if (grade == 0) {
      if (row[kTotal] != 0 || row[kWidth] != 1 || row[kRankCount] != 1 ||
          row[kDenseWidth] != 1 || prime_source >= 0 ||
          double_source >= 0) {
        *error = "invalid scalar ragged Horner grade";
        return false;
      }
    } else if (expected_rank_count != row[kRankCount]) {
      *error = "Horner predecessor ranks do not fill the output grade";
      return false;
    }
  }
  return true;
}

struct RaggedBatchBroadcastPlan {
  std::vector<int64_t> output_strides;
  std::vector<int64_t> generator_strides;
  std::vector<std::vector<int64_t>> base_strides;

  int64_t GeneratorIndex(int64_t output_index) const {
    return MapBatchIndex(output_index, output_strides, generator_strides);
  }

  int64_t BaseIndex(int64_t grade, int64_t output_index) const {
    return MapBatchIndex(output_index, output_strides,
                         base_strides[grade]);
  }
};

struct RaggedGradePlan {
  int64_t total;
  int64_t width;
  int64_t rank_count;
  int64_t dense_width;
  int64_t prime_source;
  int64_t double_source;
  int64_t double_rank_count;
  int64_t double_source_rank_count;
  int64_t selected_offset;
  int64_t selected_count;
  int64_t collision_offset;
  int64_t collision_count;
  int64_t tail_primary_count;
};

struct RaggedHornerState {
  static ffi::TypeId id;

  std::vector<RaggedGradePlan> grades;
  std::vector<int32_t> selected_edges;
  std::vector<int32_t> collision_targets;
  int64_t generator_width;
  int64_t prime_alphabet_size;
  int64_t double_alphabet_size;
  int64_t max_order;
  int64_t scalar_work_per_batch;
};

ffi::TypeId RaggedHornerState::id = XLA_FFI_UNKNOWN_TYPE_ID;

const ffi::TypeInfo kRaggedHornerStateTypeInfo =
    ffi::MakeTypeInfo<RaggedHornerState>();

ffi::ErrorOr<std::unique_ptr<RaggedHornerState>> InstantiateRaggedHorner(
    ffi::Span<const int32_t> metadata,
    ffi::Span<const int32_t> selected_edges,
    ffi::Span<const int32_t> collision_targets) {
  if (metadata.size() == 0 ||
      metadata.size() % kRaggedMetadataFields != 0) {
    return ffi::Unexpected(ffi::Error::InvalidArgument(
        "invalid ragged Horner metadata attribute"));
  }
  const int64_t grade_count =
      static_cast<int64_t>(metadata.size() / kRaggedMetadataFields);
  if (grade_count > kMaxRaggedGradeCount) {
    return ffi::Unexpected(ffi::Error::InvalidArgument(
        "ragged Horner metadata has too many grades"));
  }
  const int64_t generator_width =
      static_cast<int64_t>(metadata[kPrimeAlphabetSize]) +
      static_cast<int64_t>(metadata[kDoubleAlphabetSize]);
  std::string validation_error;
  if (!ValidateRaggedMetadata(
          metadata.begin(), grade_count,
          static_cast<int64_t>(selected_edges.size()),
          static_cast<int64_t>(collision_targets.size()), generator_width,
          &validation_error)) {
    return ffi::Unexpected(
        ffi::Error::InvalidArgument(std::move(validation_error)));
  }

  auto state = std::make_unique<RaggedHornerState>();
  state->generator_width = generator_width;
  state->prime_alphabet_size = metadata[kPrimeAlphabetSize];
  state->double_alphabet_size = metadata[kDoubleAlphabetSize];
  state->max_order = metadata[kMaxOrder];
  state->grades.reserve(grade_count);
  state->selected_edges.assign(selected_edges.begin(), selected_edges.end());
  state->collision_targets.assign(collision_targets.begin(),
                                  collision_targets.end());

  int64_t scalar_work_per_batch = 1;
  for (int64_t grade = 0; grade < grade_count; ++grade) {
    const int32_t* row = metadata.begin() + grade * kRaggedMetadataFields;
    state->grades.push_back(RaggedGradePlan{
        row[kTotal],
        row[kWidth],
        row[kRankCount],
        row[kDenseWidth],
        row[kPrimeSource],
        row[kDoubleSource],
        row[kDoubleRankCount],
        row[kDoubleSourceRankCount],
        row[kSelectedOffset],
        row[kSelectedCount],
        row[kCollisionOffset],
        row[kCollisionCount],
        row[kTailPrimaryCount],
    });
    if (grade == 0) continue;

    int64_t grade_work = row[kWidth];
    if (row[kDoubleSource] >= 0) {
      int64_t edge_count;
      int64_t double_work;
      if (!CheckedMultiply(state->double_alphabet_size,
                           row[kDoubleSourceRankCount], &edge_count) ||
          !CheckedMultiply(edge_count, row[kDenseWidth], &double_work) ||
          !CheckedAdd(grade_work, double_work, &grade_work)) {
        return ffi::Unexpected(ffi::Error::InvalidArgument(
            "ragged Horner work estimate exceeds int64"));
      }
      for (int64_t index = 0; index < row[kSelectedCount]; ++index) {
        const int64_t selected_index = row[kSelectedOffset] + index;
        const int64_t edge = selected_edges[selected_index];
        if (edge < 0 || edge >= edge_count) {
          return ffi::Unexpected(ffi::Error::InvalidArgument(
              "ragged Horner selected edge is out of range"));
        }
        const int64_t letter = edge / row[kDoubleSourceRankCount];
        if (letter >= state->double_alphabet_size - 1) {
          return ffi::Unexpected(ffi::Error::InvalidArgument(
              "ragged Horner selected edge overlaps its primary head"));
        }
      }
      for (int64_t index = 0; index < row[kCollisionCount]; ++index) {
        const int64_t target =
            collision_targets[row[kCollisionOffset] + index];
        if (target < 0 || target >= row[kDoubleRankCount]) {
          return ffi::Unexpected(ffi::Error::InvalidArgument(
              "ragged Horner collision target is out of range"));
        }
      }
    }
    if (row[kPrimeSource] >= 0) {
      const int32_t* source_row =
          metadata.begin() + row[kPrimeSource] * kRaggedMetadataFields;
      int64_t prime_work;
      if (!CheckedMultiply(source_row[kWidth], state->prime_alphabet_size,
                           &prime_work) ||
          !CheckedAdd(grade_work, prime_work, &grade_work)) {
        return ffi::Unexpected(ffi::Error::InvalidArgument(
            "ragged Horner work estimate exceeds int64"));
      }
    }
    const int64_t repetitions = state->max_order - row[kTotal] + 1;
    int64_t repeated_work;
    if (!CheckedMultiply(repetitions, grade_work, &repeated_work) ||
        !CheckedAdd(scalar_work_per_batch, repeated_work,
                    &scalar_work_per_batch)) {
      return ffi::Unexpected(ffi::Error::InvalidArgument(
          "ragged Horner work estimate exceeds int64"));
    }
  }
  state->scalar_work_per_batch = scalar_work_per_batch;
  return state;
}

struct IdenticalRaggedBatchMap {
  int64_t GeneratorIndex(int64_t output_index) const {
    return output_index;
  }
  int64_t BaseIndex(int64_t, int64_t output_index) const {
    return output_index;
  }
};

struct BroadcastRaggedBatchMap {
  const RaggedBatchBroadcastPlan* plan;

  int64_t GeneratorIndex(int64_t output_index) const {
    return plan->GeneratorIndex(output_index);
  }
  int64_t BaseIndex(int64_t grade, int64_t output_index) const {
    return plan->BaseIndex(grade, output_index);
  }
};

template <typename T, int64_t alphabet_size>
inline void AddDoubleprimeSpecialized(
    const T* source, const T* generator, const int32_t* selected_edges,
    const int32_t* collision_targets, T inverse_denominator,
    int64_t source_rank_count, int64_t dense_width,
    int64_t runtime_alphabet_size, int64_t collision_count,
    int64_t tail_count, T* output) {
  const int64_t actual_alphabet_size =
      alphabet_size == 0 ? runtime_alphabet_size : alphabet_size;
  const T head_scale =
      generator[actual_alphabet_size - 1] * inverse_denominator;
  for (int64_t source_rank = 0; source_rank < source_rank_count;
       ++source_rank) {
    const T* source_row = source + source_rank * dense_width;
    T* output_row = output + source_rank * dense_width;
    for (int64_t dense = 0; dense < dense_width; ++dense) {
      output_row[dense] += source_row[dense] * head_scale;
    }
  }
  for (int64_t tail = 0; tail < tail_count; ++tail) {
    const int64_t edge = selected_edges[tail];
    int64_t letter;
    int64_t source_rank;
    DecodeSelectedEdge<alphabet_size>(edge, source_rank_count, &letter,
                                      &source_rank);
    const T scale = generator[letter] * inverse_denominator;
    const T* source_row = source + source_rank * dense_width;
    T* output_row =
        output + (source_rank_count + tail) * dense_width;
    for (int64_t dense = 0; dense < dense_width; ++dense) {
      output_row[dense] += source_row[dense] * scale;
    }
  }
  for (int64_t collision = 0; collision < collision_count; ++collision) {
    const int64_t edge = selected_edges[tail_count + collision];
    const int64_t target = collision_targets[collision];
    int64_t letter;
    int64_t source_rank;
    DecodeSelectedEdge<alphabet_size>(edge, source_rank_count, &letter,
                                      &source_rank);
    const T scale = generator[letter] * inverse_denominator;
    const T* source_row = source + source_rank * dense_width;
    T* output_row = output + target * dense_width;
    for (int64_t dense = 0; dense < dense_width; ++dense) {
      output_row[dense] += source_row[dense] * scale;
    }
  }
}

template <typename T>
inline void AddDoubleprime(
    const T* source, const T* generator, const int32_t* selected_edges,
    const int32_t* collision_targets, T inverse_denominator,
    int64_t source_rank_count, int64_t dense_width, int64_t alphabet_size,
    int64_t collision_count, int64_t tail_count, T* output) {
  switch (alphabet_size) {
    case 1:
      return AddDoubleprimeSpecialized<T, 1>(
          source, generator, selected_edges, collision_targets,
          inverse_denominator, source_rank_count, dense_width, alphabet_size,
          collision_count, tail_count, output);
    case 2:
      return AddDoubleprimeSpecialized<T, 2>(
          source, generator, selected_edges, collision_targets,
          inverse_denominator, source_rank_count, dense_width, alphabet_size,
          collision_count, tail_count, output);
    case 3:
      return AddDoubleprimeSpecialized<T, 3>(
          source, generator, selected_edges, collision_targets,
          inverse_denominator, source_rank_count, dense_width, alphabet_size,
          collision_count, tail_count, output);
    case 4:
      return AddDoubleprimeSpecialized<T, 4>(
          source, generator, selected_edges, collision_targets,
          inverse_denominator, source_rank_count, dense_width, alphabet_size,
          collision_count, tail_count, output);
    default:
      return AddDoubleprimeSpecialized<T, 0>(
          source, generator, selected_edges, collision_targets,
          inverse_denominator, source_rank_count, dense_width, alphabet_size,
          collision_count, tail_count, output);
  }
}

template <typename T>
inline void AddPrime(const T* source, const T* generator,
                     T inverse_denominator, int64_t source_rank_count,
                     int64_t source_dense_width,
                     int64_t prime_alphabet_size,
                     int64_t output_rank_offset, T* output) {
  const int64_t output_dense_width =
      source_dense_width * prime_alphabet_size;
  for (int64_t rank = 0; rank < source_rank_count; ++rank) {
    const T* source_row = source + rank * source_dense_width;
    T* output_row =
        output + (output_rank_offset + rank) * output_dense_width;
    for (int64_t dense = 0; dense < source_dense_width; ++dense) {
      const T value = source_row[dense] * inverse_denominator;
      T* output_letters = output_row + dense * prime_alphabet_size;
      for (int64_t letter = 0; letter < prime_alphabet_size; ++letter) {
        output_letters[letter] += value * generator[letter];
      }
    }
  }
}

template <typename T, typename BatchMap>
void ComputeRaggedHornerBatchRange(
    const T* generator, const RaggedHornerState& state,
    const std::vector<const T*>& bases, const std::vector<T*>& outputs,
    int64_t batch_begin, int64_t batch_end, BatchMap batch_map) {
  const int64_t grade_count = static_cast<int64_t>(state.grades.size());
  for (int64_t batch = batch_begin; batch < batch_end; ++batch) {
    const int64_t scalar_base_batch = batch_map.BaseIndex(0, batch);
    std::memcpy(outputs[0] + batch, bases[0] + scalar_base_batch,
                sizeof(T));
    const T* z = generator +
                 batch_map.GeneratorIndex(batch) * state.generator_width;
    const T* z_prime = z;
    const T* z_double = z + state.prime_alphabet_size;
    for (int64_t active_order = 1; active_order <= state.max_order;
         ++active_order) {
      const T inverse_denominator =
          T{1} / static_cast<T>(state.max_order - active_order + 1);
      for (int64_t grade = grade_count - 1; grade >= 1; --grade) {
        const RaggedGradePlan& plan = state.grades[grade];
        if (plan.total > active_order) continue;
        const int64_t width = plan.width;
        T* output = outputs[grade] + batch * width;
        const int64_t base_batch = batch_map.BaseIndex(grade, batch);
        std::memcpy(output, bases[grade] + base_batch * width,
                    static_cast<size_t>(width) * sizeof(T));

        const int64_t double_source_grade = plan.double_source;
        if (double_source_grade >= 0) {
          const RaggedGradePlan& source_plan =
              state.grades[double_source_grade];
          const T* source =
              outputs[double_source_grade] + batch * source_plan.width;
          AddDoubleprime(
              source, z_double,
              state.selected_edges.data() + plan.selected_offset,
              state.collision_targets.data() + plan.collision_offset,
              inverse_denominator, plan.double_source_rank_count,
              plan.dense_width, state.double_alphabet_size,
              plan.collision_count, plan.tail_primary_count, output);
        }
        const int64_t prime_source_grade = plan.prime_source;
        if (prime_source_grade >= 0) {
          const RaggedGradePlan& source_plan =
              state.grades[prime_source_grade];
          const T* source =
              outputs[prime_source_grade] + batch * source_plan.width;
          const int64_t rank_offset =
              double_source_grade >= 0 ? plan.double_rank_count : 0;
          AddPrime(source, z_prime, inverse_denominator,
                   source_plan.rank_count, source_plan.dense_width,
                   state.prime_alphabet_size, rank_offset, output);
        }
      }
    }
  }
}

template <typename T>
void DispatchRaggedHornerBatchRange(
    const T* generator, const RaggedHornerState& state,
    const std::vector<const T*>& bases, const std::vector<T*>& outputs,
    int64_t batch_begin, int64_t batch_end,
    const RaggedBatchBroadcastPlan* broadcast_plan) {
  if (broadcast_plan == nullptr) {
    ComputeRaggedHornerBatchRange(
        generator, state, bases, outputs, batch_begin, batch_end,
        IdenticalRaggedBatchMap{});
  } else {
    ComputeRaggedHornerBatchRange(
        generator, state, bases, outputs, batch_begin, batch_end,
        BroadcastRaggedBatchMap{broadcast_plan});
  }
}

template <ffi::DataType dtype>
ffi::Future FusedRaggedHornerDynamicImpl(
    ffi::ThreadPool thread_pool, RaggedHornerState* state,
    ffi::Buffer<dtype> generator, ffi::RemainingArgs args,
    ffi::RemainingRets rets) {
  using T = ffi::NativeType<dtype>;
  const auto generator_dimensions = generator.dimensions();
  if (generator_dimensions.size() == 0) {
    return InvalidArgument("ragged Horner generator needs a trailing axis");
  }
  const int64_t generator_width =
      generator_dimensions[generator_dimensions.size() - 1];
  const int64_t grade_count = static_cast<int64_t>(state->grades.size());
  if (generator_width != state->generator_width ||
      args.size() != static_cast<size_t>(grade_count) ||
      rets.size() != static_cast<size_t>(grade_count)) {
    return InvalidArgument(
        "ragged Horner data arity disagrees with its instantiated plan");
  }

  std::vector<const T*> bases;
  std::vector<T*> outputs;
  std::vector<std::vector<int64_t>> base_shapes;
  bases.reserve(grade_count);
  outputs.reserve(grade_count);
  base_shapes.reserve(grade_count);
  std::vector<int64_t> common_output_shape;
  bool identical_batch = true;

  for (int64_t grade = 0; grade < grade_count; ++grade) {
    auto input_or = args.get<ffi::Buffer<dtype>>(grade);
    auto output_or = rets.get<ffi::Buffer<dtype>>(grade);
    if (!input_or || !output_or) {
      return InvalidArgument("invalid ragged Horner data buffer");
    }
    auto input = *input_or;
    auto output = *output_or;
    const auto input_dimensions = input.dimensions();
    const auto output_dimensions = output->dimensions();
    const int64_t width = state->grades[grade].width;
    if (input_dimensions.size() == 0 || output_dimensions.size() == 0 ||
        input_dimensions[input_dimensions.size() - 1] != width ||
        output_dimensions[output_dimensions.size() - 1] != width) {
      return InvalidArgument("invalid ragged Horner block width");
    }

    if (grade == 0) {
      common_output_shape.assign(output_dimensions.begin(),
                                 output_dimensions.end());
    } else if (!SameLeadingDimensions(output_dimensions, 1,
                                      common_output_shape, 1)) {
      return InvalidArgument(
          "ragged Horner outputs must share their batch shape");
    }
    identical_batch =
        identical_batch &&
        SameLeadingDimensions(input_dimensions, 1, common_output_shape, 1);
    base_shapes.emplace_back(input_dimensions.begin(),
                             input_dimensions.end());
    bases.push_back(input.typed_data());
    outputs.push_back(output->typed_data());
  }

  if (!SameLeadingDimensions(generator_dimensions, 1, common_output_shape,
                             1)) {
    identical_batch = false;
  }
  int64_t batch_count;
  if (!CheckedLeadingProduct(common_output_shape, 1, &batch_count)) {
    return InvalidArgument("invalid ragged Horner output batch shape");
  }

  std::shared_ptr<const RaggedBatchBroadcastPlan> broadcast_plan;
  if (!identical_batch) {
    auto candidate = std::make_shared<RaggedBatchBroadcastPlan>();
    std::string broadcast_error;
    if (!BuildOutputStrides(common_output_shape, 1,
                            &candidate->output_strides,
                            &broadcast_error) ||
        !BuildOperandStrides(generator_dimensions, 1, common_output_shape, 1,
                             &candidate->generator_strides,
                             &broadcast_error)) {
      return InvalidArgument(std::move(broadcast_error));
    }
    candidate->base_strides.reserve(grade_count);
    for (const auto& base_shape : base_shapes) {
      std::vector<int64_t> strides;
      if (!BuildOperandStrides(base_shape, 1, common_output_shape, 1,
                               &strides, &broadcast_error)) {
        return InvalidArgument(std::move(broadcast_error));
      }
      candidate->base_strides.push_back(std::move(strides));
    }
    broadcast_plan = std::move(candidate);
  }

  int64_t scalar_work;
  if (!CheckedMultiply(batch_count, state->scalar_work_per_batch,
                       &scalar_work)) {
    return InvalidArgument("ragged Horner scalar work exceeds int64");
  }

  const T* generator_data = generator.typed_data();
  const RaggedBatchBroadcastPlan* broadcast_data = broadcast_plan.get();
  const int64_t available_threads = std::max<int64_t>(
      1, static_cast<int64_t>(thread_pool.num_threads()));
  const int64_t desired_workers = std::max<int64_t>(
      1, scalar_work / kTargetScalarWorkPerWorker +
             (scalar_work % kTargetScalarWorkPerWorker != 0));
  const int64_t worker_count = std::min(
      std::min(batch_count, available_threads), desired_workers);
  if (worker_count <= 1 || scalar_work < kMinParallelScalarWork) {
    DispatchRaggedHornerBatchRange(
        generator_data, *state, bases, outputs, 0, batch_count,
        broadcast_data);
    return CompletedFuture();
  }

  ffi::CountDownPromise promise(worker_count);
  ffi::Future future(promise);
  for (int64_t worker = 0; worker < worker_count; ++worker) {
    const int64_t begin = batch_count * worker / worker_count;
    const int64_t end = batch_count * (worker + 1) / worker_count;
    thread_pool.Schedule(
        [=, retained_broadcast_plan = broadcast_plan]() mutable {
          DispatchRaggedHornerBatchRange(
              generator_data, *state, bases, outputs, begin, end,
              retained_broadcast_plan.get());
          promise.CountDown();
        });
  }
  return future;
}

ffi::Future FusedRaggedHornerF32DynamicImpl(
    ffi::ThreadPool thread_pool, RaggedHornerState* state,
    ffi::Buffer<ffi::F32> generator, ffi::RemainingArgs args,
    ffi::RemainingRets rets) {
  return FusedRaggedHornerDynamicImpl<ffi::F32>(
      thread_pool, state, generator, args, rets);
}

ffi::Future FusedRaggedHornerF64DynamicImpl(
    ffi::ThreadPool thread_pool, RaggedHornerState* state,
    ffi::Buffer<ffi::F64> generator, ffi::RemainingArgs args,
    ffi::RemainingRets rets) {
  return FusedRaggedHornerDynamicImpl<ffi::F64>(
      thread_pool, state, generator, args, rets);
}

#define TENSORDEV_RAGGED_HORNER_INSTANTIATE_BINDING       \
  ffi::Ffi::BindInstantiate()                             \
      .Attr<ffi::Span<const int32_t>>("metadata")         \
      .Attr<ffi::Span<const int32_t>>("selected_edges")   \
      .Attr<ffi::Span<const int32_t>>("collision_targets")

#define TENSORDEV_RAGGED_HORNER_EXECUTE_BINDING(dtype) \
  ffi::Ffi::Bind()                                     \
      .Ctx<ffi::ThreadPool>()                          \
      .Ctx<ffi::State<                                  \
          tensordev_native_cpu::RaggedHornerState>>()   \
      .Arg<ffi::Buffer<dtype>>()                       \
      .RemainingArgs()                                 \
      .RemainingRets()

}  // namespace tensordev_native_cpu

extern "C" XLA_FFI_TypeId* TensordevRaggedHornerStateTypeId() {
  return &tensordev_native_cpu::RaggedHornerState::id;
}

extern "C" const XLA_FFI_TypeInfo* TensordevRaggedHornerStateTypeInfo() {
  return &tensordev_native_cpu::kRaggedHornerStateTypeInfo;
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TensordevFusedRaggedHornerInstantiate,
    tensordev_native_cpu::InstantiateRaggedHorner,
    TENSORDEV_RAGGED_HORNER_INSTANTIATE_BINDING);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TensordevFusedRaggedHornerF32Dynamic,
    tensordev_native_cpu::FusedRaggedHornerF32DynamicImpl,
    TENSORDEV_RAGGED_HORNER_EXECUTE_BINDING(ffi::F32));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TensordevFusedRaggedHornerF64Dynamic,
    tensordev_native_cpu::FusedRaggedHornerF64DynamicImpl,
    TENSORDEV_RAGGED_HORNER_EXECUTE_BINDING(ffi::F64));
