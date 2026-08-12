// Deterministic bounded-memory voxel reduction for Boreas Stage-2 replay data.
//
// The Python orchestrator authenticates the replay ledger and writes a compact
// range descriptor.  This helper independently authenticates that descriptor,
// re-hashes every used replay range while holding a shared file lock, performs
// float64 accumulation in exact descriptor/point order, and emits lexicographic
// voxel centroids to stdout.  In verification mode it instead compares the
// canonical NPY header and every centroid byte directly against one frozen
// target file while holding a shared lock.  No second target copy is written.
// It does not contain download or registration code.

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <openssl/sha.h>

namespace {

constexpr std::size_t kPointBytes = 3U * sizeof(double);
constexpr std::uint64_t kMaximumLoadNumerator = 7U;
constexpr std::uint64_t kMaximumLoadDenominator = 10U;
constexpr std::size_t kInitialCapacity = 1024U;
constexpr std::size_t kConstructorCapacityLimit = 1U << 20U;
constexpr std::size_t kPointsPerChunk = 1U << 18U;
constexpr std::size_t kOutputRows = 4096U;
constexpr std::uint64_t kCapacityFixedOverheadBytes = 64U * 1024U * 1024U;

struct Key {
  std::int32_t x;
  std::int32_t y;
  std::int32_t z;
};

struct Slot {
  Key key{};
  double sums[3]{0.0, 0.0, 0.0};
  std::uint64_t count{0};
  bool occupied{false};
};

struct Range {
  std::uint64_t byte_offset;
  std::uint64_t point_count;
  std::string sha256;
};

[[noreturn]] void fail(const std::string &message);

std::uint64_t usable_capacity(std::size_t capacity) {
  return (static_cast<std::uint64_t>(capacity) * kMaximumLoadNumerator) /
         kMaximumLoadDenominator;
}

std::size_t final_capacity_for(std::uint64_t max_voxels) {
  std::size_t capacity = kInitialCapacity;
  while (usable_capacity(capacity) < max_voxels) {
    if (capacity > std::numeric_limits<std::size_t>::max() / 2U) {
      fail("capacity layout overflows size_t");
    }
    capacity *= 2U;
  }
  return capacity;
}

std::size_t constructor_capacity_for(std::uint64_t max_voxels) {
  std::size_t capacity = kInitialCapacity;
  while (usable_capacity(capacity) < max_voxels &&
         capacity < kConstructorCapacityLimit) {
    capacity *= 2U;
  }
  return capacity;
}

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

std::uint64_t parse_u64(const std::string &text, const char *field) {
  if (text.empty() || text[0] == '-') {
    fail(std::string(field) + " must be a nonnegative integer");
  }
  std::size_t consumed = 0;
  const auto value = std::stoull(text, &consumed, 10);
  if (consumed != text.size()) {
    fail(std::string(field) + " is not an exact integer");
  }
  return value;
}

double parse_finite_double(const std::string &text, const char *field) {
  std::size_t consumed = 0;
  const double value = std::stod(text, &consumed);
  if (consumed != text.size() || !std::isfinite(value)) {
    fail(std::string(field) + " must be finite");
  }
  return value;
}

bool is_lower_hex_sha256(const std::string &value) {
  if (value.size() != 64U)
    return false;
  for (const char ch : value) {
    if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')))
      return false;
  }
  return true;
}

std::string hex_digest(const unsigned char *digest, std::size_t size) {
  static constexpr char alphabet[] = "0123456789abcdef";
  std::string result(size * 2U, '0');
  for (std::size_t index = 0; index < size; ++index) {
    result[index * 2U] = alphabet[digest[index] >> 4U];
    result[index * 2U + 1U] = alphabet[digest[index] & 0x0fU];
  }
  return result;
}

std::string sha256_file(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail("cannot open descriptor for SHA-256");
  SHA256_CTX context;
  SHA256_Init(&context);
  std::array<char, 1U << 20U> buffer{};
  while (stream) {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = stream.gcount();
    if (count > 0)
      SHA256_Update(&context, buffer.data(), static_cast<std::size_t>(count));
  }
  if (!stream.eof())
    fail("descriptor read failed");
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256_Final(digest, &context);
  return hex_digest(digest, SHA256_DIGEST_LENGTH);
}

std::string sha256_final_hex(SHA256_CTX *context) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256_Final(digest, context);
  return hex_digest(digest, SHA256_DIGEST_LENGTH);
}

std::vector<Range> read_descriptor(const std::string &path) {
  std::ifstream stream(path);
  if (!stream)
    fail("cannot open range descriptor");
  std::string line;
  if (!std::getline(stream, line) ||
      line != "byte_offset\tpoint_count\tsha256") {
    fail("range descriptor header differs");
  }
  std::vector<Range> result;
  std::uint64_t previous_offset = 0;
  std::uint64_t previous_end = 0;
  while (std::getline(stream, line)) {
    if (line.empty())
      fail("range descriptor contains an empty row");
    std::istringstream row(line);
    std::string offset_text, count_text, digest, extra;
    if (!std::getline(row, offset_text, '\t') ||
        !std::getline(row, count_text, '\t') ||
        !std::getline(row, digest, '\t') || std::getline(row, extra, '\t')) {
      fail("range descriptor row schema differs");
    }
    Range range{parse_u64(offset_text, "byte_offset"),
                parse_u64(count_text, "point_count"), digest};
    if (!is_lower_hex_sha256(range.sha256))
      fail("range SHA-256 is invalid");
    if (!result.empty() && range.byte_offset <= previous_offset) {
      fail("range descriptor offsets must be strictly increasing");
    }
    if (range.point_count >
        std::numeric_limits<std::uint64_t>::max() / kPointBytes) {
      fail("range byte count overflows");
    }
    const auto byte_count = range.point_count * kPointBytes;
    if (range.byte_offset >
        std::numeric_limits<std::uint64_t>::max() - byte_count) {
      fail("range endpoint overflows");
    }
    if (!result.empty() && range.byte_offset < previous_end) {
      fail("range descriptor active ranges overlap");
    }
    previous_offset = range.byte_offset;
    previous_end = range.byte_offset + byte_count;
    result.push_back(range);
  }
  if (!stream.eof())
    fail("range descriptor read failed");
  if (result.empty())
    fail("range descriptor must contain at least one range");
  return result;
}

std::uint64_t mix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::uint64_t key_hash(const Key &key) {
  const auto x = static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.x));
  const auto y = static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.y));
  const auto z = static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.z));
  return mix64(x) ^ (mix64(y + 0x517cc1b727220a95ULL) << 1U) ^
         (mix64(z + 0x6eed0e9da4d94a4fULL) << 2U);
}

bool same_key(const Key &left, const Key &right) {
  return left.x == right.x && left.y == right.y && left.z == right.z;
}

class VoxelTable {
public:
  explicit VoxelTable(std::uint64_t max_voxels) : max_voxels_(max_voxels) {
    if (max_voxels_ == 0U)
      fail("max_voxels must be positive");
    slots_.resize(constructor_capacity_for(max_voxels_));
  }

  void add(const Key &key, const double *point) {
    if ((size_ + 1U) > usable_capacity(slots_.size())) {
      grow();
    }
    const std::size_t mask = slots_.size() - 1U;
    std::size_t index = static_cast<std::size_t>(key_hash(key)) & mask;
    while (true) {
      Slot &slot = slots_[index];
      if (!slot.occupied) {
        if (size_ >= max_voxels_)
          fail("voxel count exceeds fail-closed max_voxels");
        slot.occupied = true;
        slot.key = key;
        slot.sums[0] = point[0];
        slot.sums[1] = point[1];
        slot.sums[2] = point[2];
        slot.count = 1U;
        ++size_;
        return;
      }
      if (same_key(slot.key, key)) {
        // Exact encounter-order float64 reduction; do not
        // parallelize/reassociate.
        slot.sums[0] += point[0];
        slot.sums[1] += point[1];
        slot.sums[2] += point[2];
        ++slot.count;
        return;
      }
      index = (index + 1U) & mask;
    }
  }

  std::uint64_t size() const { return size_; }

  std::vector<std::size_t> sorted_indices() const {
    std::vector<std::size_t> result;
    result.reserve(static_cast<std::size_t>(size_));
    for (std::size_t index = 0; index < slots_.size(); ++index) {
      if (slots_[index].occupied)
        result.push_back(index);
    }
    std::sort(result.begin(), result.end(),
              [this](std::size_t a, std::size_t b) {
                const Key &left = slots_[a].key;
                const Key &right = slots_[b].key;
                if (left.x != right.x)
                  return left.x < right.x;
                if (left.y != right.y)
                  return left.y < right.y;
                return left.z < right.z;
              });
    return result;
  }

  const Slot &slot(std::size_t index) const { return slots_[index]; }

private:
  void grow() {
    if (slots_.size() > std::numeric_limits<std::size_t>::max() / 2U) {
      fail("voxel table capacity overflows");
    }
    const std::uint64_t next_usable = usable_capacity(slots_.size() * 2U);
    if (next_usable / 2U >= max_voxels_ && size_ < max_voxels_) {
      // The current table may still fill up to max_voxels without another grow.
    }
    std::vector<Slot> replacement(slots_.size() * 2U);
    const std::size_t mask = replacement.size() - 1U;
    for (const Slot &old : slots_) {
      if (!old.occupied)
        continue;
      std::size_t index = static_cast<std::size_t>(key_hash(old.key)) & mask;
      while (replacement[index].occupied)
        index = (index + 1U) & mask;
      replacement[index] = old;
    }
    slots_.swap(replacement);
  }

  std::vector<Slot> slots_;
  std::uint64_t size_{0};
  std::uint64_t max_voxels_;
};

void print_capacity_layout(std::uint64_t max_voxels) {
  if (max_voxels == 0U)
    fail("max_voxels must be positive");
  const auto constructor_capacity = constructor_capacity_for(max_voxels);
  const auto final_capacity = final_capacity_for(max_voxels);
  const auto checked_product = [](std::uint64_t left, std::uint64_t right,
                                  const char *field) {
    if (right != 0U &&
        left > std::numeric_limits<std::uint64_t>::max() / right) {
      fail(std::string(field) + " overflows uint64");
    }
    return left * right;
  };
  const auto slot_bytes = static_cast<std::uint64_t>(sizeof(Slot));
  const auto final_table_bytes =
      checked_product(final_capacity, slot_bytes, "final_table_bytes");
  const auto largest_old_capacity =
      final_capacity > constructor_capacity ? final_capacity / 2U : 0U;
  const auto peak_growth_table_bytes =
      checked_product(final_capacity + largest_old_capacity, slot_bytes,
                      "peak_growth_table_bytes");
  const auto sorted_index_upper_bytes = checked_product(
      max_voxels, sizeof(std::size_t), "sorted_index_upper_bytes");
  const auto input_buffer_bytes =
      static_cast<std::uint64_t>(kPointsPerChunk) * kPointBytes;
  const auto output_buffer_bytes =
      static_cast<std::uint64_t>(kOutputRows) * kPointBytes;
  const auto target_npy_upper_bytes =
      checked_product(max_voxels, kPointBytes, "target_npy_upper_bytes") +
      4096U;
  const auto growth_peak = peak_growth_table_bytes + input_buffer_bytes +
                           kCapacityFixedOverheadBytes;
  const auto output_peak = final_table_bytes + sorted_index_upper_bytes +
                           input_buffer_bytes + output_buffer_bytes +
                           target_npy_upper_bytes + kCapacityFixedOverheadBytes;
  const auto total_peak = std::max(growth_peak, output_peak);
  std::cout << "{\"constructor_capacity\":" << constructor_capacity
            << ",\"final_table_bytes\":" << final_table_bytes
            << ",\"fixed_overhead_bytes\":" << kCapacityFixedOverheadBytes
            << ",\"input_buffer_bytes\":" << input_buffer_bytes
            << ",\"key_bytes\":" << sizeof(Key)
            << ",\"max_voxels\":" << max_voxels
            << ",\"maximum_load_denominator\":" << kMaximumLoadDenominator
            << ",\"maximum_load_numerator\":" << kMaximumLoadNumerator
            << ",\"output_buffer_bytes\":" << output_buffer_bytes
            << ",\"peak_growth_table_bytes\":" << peak_growth_table_bytes
            << ",\"schema\":\"zprm-boreas-stage2-reducer-capacity-layout-v1\""
            << ",\"size_t_bytes\":" << sizeof(std::size_t)
            << ",\"slot_bytes\":" << sizeof(Slot)
            << ",\"sorted_index_upper_bytes\":" << sorted_index_upper_bytes
            << ",\"table_capacity\":" << final_capacity
            << ",\"target_npy_upper_bytes\":" << target_npy_upper_bytes
            << ",\"total_peak_upper_bound_bytes\":" << total_peak << "}\n";
}

void read_exact(int descriptor, void *destination, std::size_t count,
                std::uint64_t offset) {
  auto *cursor = static_cast<unsigned char *>(destination);
  std::size_t remaining = count;
  while (remaining > 0U) {
    const auto chunk =
        ::pread(descriptor, cursor, remaining, static_cast<off_t>(offset));
    if (chunk < 0 && errno == EINTR)
      continue;
    if (chunk <= 0)
      fail("replay range is truncated or unreadable");
    cursor += chunk;
    offset += static_cast<std::uint64_t>(chunk);
    remaining -= static_cast<std::size_t>(chunk);
  }
}

Key voxel_key(const double *point, double size,
              const std::array<double, 3> &origin) {
  std::int32_t values[3];
  for (std::size_t axis = 0; axis < 3U; ++axis) {
    if (!std::isfinite(point[axis]))
      fail("replay contains a nonfinite point");
    const double scaled = std::floor((point[axis] - origin[axis]) / size);
    if (!std::isfinite(scaled) ||
        scaled < std::numeric_limits<std::int32_t>::min() ||
        scaled > std::numeric_limits<std::int32_t>::max()) {
      fail("voxel key exceeds int32 range");
    }
    values[axis] = static_cast<std::int32_t>(scaled);
  }
  return Key{values[0], values[1], values[2]};
}

void write_all_stdout(const void *source, std::size_t count) {
  const auto *cursor = static_cast<const unsigned char *>(source);
  while (count > 0U) {
    const auto written = ::write(STDOUT_FILENO, cursor, count);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0)
      fail("stdout write failed");
    cursor += written;
    count -= static_cast<std::size_t>(written);
  }
}

std::string canonical_npy_header(std::uint64_t row_count) {
  // This is NumPy's v1.0 canonical header for a C-contiguous <f8 (N, 3)
  // array: sorted dictionary keys, trailing comma/space, newline, and padding
  // to the 64-byte ARRAY_ALIGN boundary.
  const std::string dictionary =
      "{'descr': '<f8', 'fortran_order': False, 'shape': (" +
      std::to_string(row_count) + ", 3), }";
  constexpr std::size_t preamble_bytes =
      10U; // magic/version + uint16 header length
  constexpr std::size_t alignment = 64U;
  const std::size_t padding =
      alignment - ((preamble_bytes + dictionary.size() + 1U) % alignment);
  const std::size_t header_length = dictionary.size() + padding + 1U;
  if (header_length > std::numeric_limits<std::uint16_t>::max()) {
    fail("canonical NPY header exceeds v1.0 uint16 limit");
  }
  std::string result;
  result.reserve(preamble_bytes + header_length);
  result.append("\x93NUMPY", 6U);
  result.push_back(static_cast<char>(1));
  result.push_back(static_cast<char>(0));
  result.push_back(static_cast<char>(header_length & 0xffU));
  result.push_back(static_cast<char>((header_length >> 8U) & 0xffU));
  result += dictionary;
  result.append(padding, ' ');
  result.push_back('\n');
  return result;
}

int open_locked_regular(const std::string &path, const char *label) {
  int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif
  const int descriptor = ::open(path.c_str(), flags);
  if (descriptor < 0)
    fail(std::string("cannot open ") + label);
  if (::flock(descriptor, LOCK_SH) != 0) {
    ::close(descriptor);
    fail(std::string("cannot lock ") + label);
  }
  struct stat status {};
  if (::fstat(descriptor, &status) != 0 || !S_ISREG(status.st_mode)) {
    ::flock(descriptor, LOCK_UN);
    ::close(descriptor);
    fail(std::string(label) + " is not a regular file");
  }
  return descriptor;
}

void close_locked(int descriptor, const char *label) {
  if (::flock(descriptor, LOCK_UN) != 0) {
    ::close(descriptor);
    fail(std::string("cannot unlock ") + label);
  }
  if (::close(descriptor) != 0)
    fail(std::string("cannot close ") + label);
}

void compare_exact(int descriptor, const void *expected, std::size_t count,
                   std::uint64_t offset, SHA256_CTX *target_sha) {
  std::vector<unsigned char> actual(count);
  read_exact(descriptor, actual.data(), count, offset);
  if (std::memcmp(actual.data(), expected, count) != 0) {
    fail("target NPY differs from deterministic replay reduction");
  }
  SHA256_Update(target_sha, actual.data(), count);
}

struct Arguments {
  std::string replay;
  std::string descriptor;
  std::string descriptor_sha256;
  double voxel_size{0.0};
  std::array<double, 3> origin{};
  std::uint64_t max_voxels{0};
  std::string verify_target_npy;
};

Arguments parse_arguments(int argc, char **argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string name(argv[index]);
    if (index + 1 >= argc)
      fail("missing CLI option value");
    const std::string value(argv[++index]);
    if (name == "--replay")
      result.replay = value;
    else if (name == "--descriptor")
      result.descriptor = value;
    else if (name == "--descriptor-sha256")
      result.descriptor_sha256 = value;
    else if (name == "--voxel-size-m")
      result.voxel_size = parse_finite_double(value, "voxel_size_m");
    else if (name == "--origin-x-m")
      result.origin[0] = parse_finite_double(value, "origin_x_m");
    else if (name == "--origin-y-m")
      result.origin[1] = parse_finite_double(value, "origin_y_m");
    else if (name == "--origin-z-m")
      result.origin[2] = parse_finite_double(value, "origin_z_m");
    else if (name == "--max-voxels")
      result.max_voxels = parse_u64(value, "max_voxels");
    else if (name == "--verify-target-npy")
      result.verify_target_npy = value;
    else
      fail("unknown CLI option: " + name);
  }
  if (result.replay.empty() || result.descriptor.empty() ||
      !is_lower_hex_sha256(result.descriptor_sha256) ||
      !(result.voxel_size > 0.0) || result.max_voxels == 0U) {
    fail("required CLI contract is incomplete");
  }
  return result;
}

void execute(const Arguments &arguments) {
  if (sha256_file(arguments.descriptor) != arguments.descriptor_sha256) {
    fail("range descriptor SHA-256 differs");
  }
  const auto ranges = read_descriptor(arguments.descriptor);
  const int replay = open_locked_regular(arguments.replay, "replay array");
  int target = -1;
  try {
    if (!arguments.verify_target_npy.empty()) {
      target = open_locked_regular(arguments.verify_target_npy, "target NPY");
    }
    VoxelTable table(arguments.max_voxels);
    constexpr std::size_t points_per_chunk = 1U << 18U;
    std::vector<double> buffer(points_per_chunk * 3U);
    for (std::size_t ordinal = 0; ordinal < ranges.size(); ++ordinal) {
      const Range &range = ranges[ordinal];
      SHA256_CTX context;
      SHA256_Init(&context);
      std::uint64_t remaining = range.point_count;
      std::uint64_t offset = range.byte_offset;
      while (remaining > 0U) {
        const auto count = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining, points_per_chunk));
        const auto byte_count = count * kPointBytes;
        read_exact(replay, buffer.data(), byte_count, offset);
        SHA256_Update(&context, buffer.data(), byte_count);
        for (std::size_t point_index = 0; point_index < count; ++point_index) {
          const double *point = buffer.data() + point_index * 3U;
          table.add(voxel_key(point, arguments.voxel_size, arguments.origin),
                    point);
        }
        remaining -= count;
        offset += byte_count;
      }
      unsigned char digest[SHA256_DIGEST_LENGTH];
      SHA256_Final(digest, &context);
      if (hex_digest(digest, SHA256_DIGEST_LENGTH) != range.sha256) {
        fail("authenticated replay range SHA-256 differs at ordinal " +
             std::to_string(ordinal));
      }
      if ((ordinal + 1U) % 100U == 0U || ordinal + 1U == ranges.size()) {
        std::cerr << "processed_ranges=" << (ordinal + 1U)
                  << " voxels=" << table.size() << '\n';
      }
    }
    const auto indices = table.sorted_indices();
    const std::uint64_t count = table.size();
    SHA256_CTX target_sha_context;
    std::uint64_t target_offset = 0U;
    std::string target_header;
    if (target >= 0) {
      SHA256_Init(&target_sha_context);
      target_header = canonical_npy_header(count);
      compare_exact(target, target_header.data(), target_header.size(), 0U,
                    &target_sha_context);
      target_offset = static_cast<std::uint64_t>(target_header.size());
    } else {
      write_all_stdout(&count, sizeof(count));
    }
    std::array<double, 3U * 4096U> output{};
    std::size_t buffered = 0U;
    for (const std::size_t index : indices) {
      const Slot &slot = table.slot(index);
      for (std::size_t axis = 0; axis < 3U; ++axis) {
        double value = slot.sums[axis] / static_cast<double>(slot.count);
        // Match the executable Python specification's signed-zero
        // canonicalization.
        if (value == 0.0)
          value = 0.0;
        output[buffered * 3U + axis] = value;
      }
      ++buffered;
      if (buffered == 4096U) {
        const auto byte_count = buffered * kPointBytes;
        if (target >= 0) {
          compare_exact(target, output.data(), byte_count, target_offset,
                        &target_sha_context);
          target_offset += byte_count;
        } else {
          write_all_stdout(output.data(), byte_count);
        }
        buffered = 0U;
      }
    }
    if (buffered) {
      const auto byte_count = buffered * kPointBytes;
      if (target >= 0) {
        compare_exact(target, output.data(), byte_count, target_offset,
                      &target_sha_context);
        target_offset += byte_count;
      } else {
        write_all_stdout(output.data(), byte_count);
      }
    }
    if (target >= 0) {
      struct stat target_status {};
      if (::fstat(target, &target_status) != 0 || target_status.st_size < 0 ||
          static_cast<std::uint64_t>(target_status.st_size) != target_offset) {
        fail(
            "target NPY byte size differs from deterministic replay reduction");
      }
      const std::string target_sha = sha256_final_hex(&target_sha_context);
      close_locked(target, "target NPY");
      target = -1;
      // Keys are emitted in lexical order so the line is canonical JSON.
      std::cout
          << "{\"compared_npy_bytes\":" << target_offset
          << ",\"descriptor_sha256\":\"" << arguments.descriptor_sha256
          << "\",\"replay_range_count\":" << ranges.size()
          << ",\"schema\":\"zprm.boreas.stage2.target_replay_verification.v1\""
          << ",\"status\":\"PASS_EXACT_TARGET_NPY_REPLAY\""
          << ",\"target_npy_sha256\":\"" << target_sha
          << "\",\"target_point_count\":" << count << "}\n";
    }
    close_locked(replay, "replay array");
  } catch (...) {
    if (target >= 0) {
      ::flock(target, LOCK_UN);
      ::close(target);
    }
    ::flock(replay, LOCK_UN);
    ::close(replay);
    throw;
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "--print-capacity-layout" &&
        std::string(argv[2]) == "--max-voxels") {
      print_capacity_layout(parse_u64(argv[3], "max_voxels"));
      return 0;
    }
    execute(parse_arguments(argc, argv));
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "boreas_stage2_voxel_reduce: " << error.what() << '\n';
    return 2;
  }
}
