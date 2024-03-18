#include <fcntl.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#define real_inline inline

constexpr uint MAX_UNIQUE_STATION_NAME = 10'000;
constexpr int NUM_OF_THREADS = 8;

inline double betterRound(double val) { return std::round(val * 10) / 10; }

inline int _mm256i_equal(__m256i a, __m256i b) {
  __m256i neq = _mm256_xor_si256(a, b);
  return _mm256_testz_si256(neq, neq);
}

struct MemoryMappingFile {
 private:
  int fd_ = -1;
  long long int len_ = 0;
  char* buffer;

 public:
  explicit MemoryMappingFile(const std::string& fileName) {
    fd_ = open(fileName.c_str(), O_RDONLY);
    len_ = lseek(fd_, 0, SEEK_END);
    buffer =
        static_cast<char*>(mmap(nullptr, len_, PROT_READ, MAP_PRIVATE, fd_, 0));
  }

  ~MemoryMappingFile() {
    close(fd_);
    munmap(buffer, len_);
  }

  char* getData() const { return buffer; }
  uint64_t getSize() const { return len_; }
};

struct StationResult {
  char key[32]{};
  int len = 0;
  int max = -999;
  int min = 999;
  int total = 0;
  int n = 0;

  explicit StationResult() = default;

  double getMean() const { return betterRound(betterRound(total / 10.f) / n); }
  double getMax() const { return max / 10.f; }
  double getMin() const { return min / 10.f; }
};

struct LinearProbeHash {
 private:
  using HashInfo = StationResult;

  static inline constexpr uint32_t MAX_BUCKET = 1 << 14;
  alignas(4096) static inline constexpr std::array<uint8_t, 64> idxMask = []() {
    std::array<uint8_t, 64> res{};
    for (int i = 0; i < 32; ++i) {
      res[i] = 255;
    }
    return res;
  }();

  alignas(4096) std::array<HashInfo, MAX_BUCKET> data;
  std::vector<StationResult*> resultInfo;

  static inline uint32_t hash(std::string_view key) {
    return std::hash<std::string_view>{}(key) % MAX_BUCKET;
  }

 public:
  explicit LinearProbeHash() { resultInfo.reserve(MAX_UNIQUE_STATION_NAME); }

  const std::vector<StationResult*>& getResultInfo() const {
    return resultInfo;
  }
  const std::array<HashInfo, MAX_BUCKET>& getRawData() const { return data; };

  StationResult& get(const std::string_view key) {
    uint32_t hashKey = hash(key);
    while (data[hashKey].len > 0) {
      assert(key.length() <= 32);
      // clang-format off
      __m256i keyBytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(key.data()));
      __m256i keyBytesPd = _mm256_and_si256(
          keyBytes,
          _mm256_load_si256( reinterpret_cast<const __m256i*>(idxMask.data() + 32 - key.size())));
      __m256i bucketKeyBytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data[hashKey].key));
      // clang-format on
      if (_mm256i_equal(keyBytesPd, bucketKeyBytes)) {
        return data[hashKey];
      }
      hashKey = (hashKey + 1) % MAX_BUCKET;
    }
    auto& bucket = data[hashKey];
    std::memcpy(bucket.key, key.data(), key.size());
    bucket.len = key.size();
    resultInfo.push_back(&bucket);
    return bucket;
  }
};

void debug(const std::string& name,
           std::unordered_map<std::string, StationResult>& result) {
  auto v = &result[name];
  std::cout << "total=" << v->total << "\n";
  std::cout << "min=" << v->min << "\n";
  std::cout << "max=" << v->max << "\n";
  std::cout << "n=" << v->n << "\n";
  std::cout << "mean=" << v->getMean() << "\n";
}

void outputResult(const std::vector<StationResult*>& result,
                  const std::string& outputPath) {
  std::fstream ofs(outputPath, std::ios::in | std::ios::out | std::ios::trunc);

  std::string formatPrefix = "{";
  ofs << std::setprecision(1) << std::fixed;
  for (auto& v : result) {
    ofs << formatPrefix << std::string(v->key, v->len) << "=" << v->getMin()
        << "/" << v->getMean() << "/" << v->getMax();
    formatPrefix = ", ";
  }
  ofs << "}\n";
}

real_inline int parseInt(std::string_view str) {
  int res = 0;
  res += (str[str.size() - 1] - '0') + (str[str.size() - 3] - '0') * 10 +
         (str[str.size() - 4] - '0') * 100 * (str.size() == 4);
  return res;
}

real_inline void aggregation(LinearProbeHash& count,
                             std::string_view stationName, int temperature) {
  StationResult& result = count.get(stationName);
  if (result.max < temperature) [[unlikely]] {
    result.max = temperature;
  }

  if (result.min > temperature) [[unlikely]] {
    result.min = temperature;
  }
  result.total += temperature;
  ++result.n;
}

real_inline uint64_t findNext(char* buf, uint64_t startIdx, uint64_t endIdx,
                              char ch) {
  while (startIdx < endIdx && buf[startIdx] != ch) [[likely]] {
    ++startIdx;
  }
  return startIdx;
}

inline uint64_t getDelimiterMaskWithSimd(char* buf, uint64_t startIdx) {
  // clang-format off
  __m256i dataMask1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buf + startIdx));
  __m256i dataMask2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buf + startIdx + 32));
  __m256i semiColonMask = _mm256_set1_epi8(';');
  uint32_t cmpMask1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(dataMask1, semiColonMask));
  uint32_t cmpMask2 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(dataMask2, semiColonMask));
  uint64_t cmpMask = (static_cast<uint64_t>(cmpMask2) << 32U) | cmpMask1;
  // clang-format on
  return cmpMask;
}

inline void processChunk(LinearProbeHash& count, char* buf, uint64_t startIdx,
                         uint64_t endIdx) {
  while (startIdx < endIdx) [[likely]] {
    uint64_t cmpMask = getDelimiterMaskWithSimd(buf, startIdx);

    uint64_t semiColonIdx = __builtin_ctzll(cmpMask) + startIdx;
    std::string_view stationName(buf + startIdx, semiColonIdx - (startIdx));

    bool sign = buf[semiColonIdx + 1] == '-';
    startIdx = semiColonIdx + (sign);

    uint64_t newLineIdx = startIdx + 4 + (buf[startIdx + 3] == '.');
    std::string_view temperature(buf + startIdx + 1, newLineIdx - startIdx - 1);
    startIdx = newLineIdx + 1;

    aggregation(count, stationName, parseInt(temperature) * (sign ? -1 : 1));
  }
}

inline void processChunkUnroll(LinearProbeHash& count, char* buf, uint64_t sidx,
                               uint64_t eidx) {
  const uint64_t subChunkSize = (eidx - sidx) / 3;
  uint64_t startIdx0 = sidx;
  uint64_t endIdx0 = startIdx0 + subChunkSize;
  uint64_t startIdx1 = findNext(buf, endIdx0, eidx, '\n') + 1;
  uint64_t endIdx1 = eidx;

  while (true) {
    if (startIdx0 >= endIdx0) [[unlikely]]
      break;
    if (startIdx1 >= endIdx1) [[unlikely]]
      break;

    uint64_t nameStartIdx0 = startIdx0;
    uint64_t nameStartIdx1 = startIdx1;

    uint64_t cmpMask0 = getDelimiterMaskWithSimd(buf, startIdx0);
    uint64_t cmpMask1 = getDelimiterMaskWithSimd(buf, startIdx1);

    uint64_t semiColonIdx0 = __builtin_ctzll(cmpMask0) + startIdx0;
    uint64_t semiColonIdx1 = __builtin_ctzll(cmpMask1) + startIdx1;

    bool sign0 = buf[semiColonIdx0 + 1] == '-';
    bool sign1 = buf[semiColonIdx1 + 1] == '-';

    startIdx0 = semiColonIdx0 + (sign0);
    startIdx1 = semiColonIdx1 + (sign1);

    uint64_t newLineIdx0 = startIdx0 + 4 + (buf[startIdx0 + 3] == '.');
    uint64_t newLineIdx1 = startIdx1 + 4 + (buf[startIdx1 + 3] == '.');

    std::string_view stationName0(buf + nameStartIdx0,
                                  semiColonIdx0 - nameStartIdx0);
    std::string_view stationName1(buf + nameStartIdx1,
                                  semiColonIdx1 - nameStartIdx1);

    std::string_view temperature0(buf + startIdx0 + 1,
                                  newLineIdx0 - startIdx0 - 1);
    std::string_view temperature1(buf + startIdx1 + 1,
                                  newLineIdx1 - startIdx1 - 1);

    startIdx0 = newLineIdx0 + 1;
    startIdx1 = newLineIdx1 + 1;

    aggregation(count, stationName0, parseInt(temperature0) * (sign0 ? -1 : 1));
    aggregation(count, stationName1, parseInt(temperature1) * (sign1 ? -1 : 1));
  }

  while (startIdx0 < endIdx0) [[likely]] {
    uint64_t nameStartIdx0 = startIdx0;
    uint64_t cmpMask0 = getDelimiterMaskWithSimd(buf, startIdx0);
    uint64_t semiColonIdx0 = __builtin_ctzll(cmpMask0) + startIdx0;

    bool sign0 = buf[semiColonIdx0 + 1] == '-';
    startIdx0 = semiColonIdx0 + (sign0);

    uint64_t newLineIdx0 = startIdx0 + 4 + (buf[startIdx0 + 3] == '.');
    std::string_view stationName0(buf + nameStartIdx0,
                                  semiColonIdx0 - nameStartIdx0);
    std::string_view temperature0(buf + startIdx0 + 1,
                                  newLineIdx0 - startIdx0 - 1);
    startIdx0 = newLineIdx0 + 1;
    aggregation(count, stationName0, parseInt(temperature0) * (sign0 ? -1 : 1));
  }

  while (startIdx1 < endIdx1) [[likely]] {
    uint64_t nameStartIdx1 = startIdx1;
    uint64_t cmpMask1 = getDelimiterMaskWithSimd(buf, startIdx1);
    uint64_t semiColonIdx1 = __builtin_ctzll(cmpMask1) + startIdx1;

    bool sign1 = buf[semiColonIdx1 + 1] == '-';
    startIdx1 = semiColonIdx1 + (sign1);

    uint64_t newLineIdx1 = startIdx1 + 4 + (buf[startIdx1 + 3] == '.');
    std::string_view stationName1(buf + nameStartIdx1,
                                  semiColonIdx1 - nameStartIdx1);
    std::string_view temperature1(buf + startIdx1 + 1,
                                  newLineIdx1 - startIdx1 - 1);
    startIdx1 = newLineIdx1 + 1;
    aggregation(count, stationName1, parseInt(temperature1) * (sign1 ? -1 : 1));
  }
}

void parallelTask(LinearProbeHash& count, char* buf, uint64_t startIdx,
                  uint64_t endIdx) {
  // processChunk(count, buf, startIdx, endIdx);
  processChunkUnroll(count, buf, startIdx, endIdx);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <measurement_file> <result_file>\n";
    return 1;
  }
  const std::string measurementFile = argv[1];
  const std::string resultFile = argv[2];

  MemoryMappingFile file(measurementFile);
  char* buf = file.getData();

  std::vector<std::thread> threads;
  std::vector<LinearProbeHash> partial(NUM_OF_THREADS);
  uint64_t chunkSize = file.getSize() / NUM_OF_THREADS;
  uint64_t nextStartIdx = 0;
  for (int i = 0; i < NUM_OF_THREADS; ++i) {
    uint64_t endIdx = std::min(
        findNext(buf, nextStartIdx + chunkSize, file.getSize(), '\n') + 1,
        file.getSize());
    threads.emplace_back([&partial, i, buf, nextStartIdx, endIdx]() {
      parallelTask(partial[i], buf, nextStartIdx, endIdx);
    });
    nextStartIdx = endIdx;
  }

  for (auto& t : threads) {
    t.join();
  }

  std::unordered_map<std::string, StationResult> fullRes;
  for (auto& partialRes : partial) {
    for (auto& result : partialRes.getResultInfo()) {
      const std::string k(result->key, result->len);
      StationResult& v = *result;
      if (!fullRes.contains(k)) {
        fullRes[k] = std::move(v);
      } else {
        StationResult& result = fullRes[k];
        result.total += v.total;
        result.n += v.n;
        result.min = std::min(result.min, v.min);
        result.max = std::max(result.max, v.max);
      }
    }
  }

  std::vector<StationResult*> res;
  for (auto& [k, v] : fullRes) {
    res.push_back(&v);
  }
  std::ranges::sort(res, [](auto& a, auto& b) {
    return std::string(a->key, a->len) < std::string(b->key, b->len);
  });

  outputResult(res, resultFile);

  return 0;
}
