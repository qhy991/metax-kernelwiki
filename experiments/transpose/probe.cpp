// Exact FP32 row-major transpose: direct and two shared-memory variants.
#include <mcr/mc_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint64_t kInputElements = 1ULL << 24;
constexpr uint64_t kGuardElements = 32;

void check_mc(mcError_t status, const char* expression) {
    if (status != mcSuccess)
        throw std::runtime_error(std::string(expression) + ": " + mcGetErrorString(status));
}
#define MC_CHECK(expression) check_mc((expression), #expression)

std::string json_string(const std::string& value) {
    std::ostringstream result;
    result << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') result << '\\' << c;
        else if (c < 32) result << "\\u" << std::hex << std::setw(4)
                                << std::setfill('0') << static_cast<int>(c) << std::dec;
        else result << c;
    }
    result << '"';
    return result.str();
}

std::string json_environment(const char* key) {
    const char* value = std::getenv(key);
    return value ? json_string(value) : "null";
}

struct Case {
    std::string id, variant;
    uint64_t rows = 0, cols = 0, n = 0;
    unsigned warmups = 0, samples = 0, launches = 0;
};

struct DynamicShared {
    unsigned pitch = 0, bytes = 0;
};

DynamicShared dynamic_shared(const Case& c) {
    if (c.variant == "dynamic_pitch64_bytes16384") return {64, 16384};
    if (c.variant == "dynamic_pitch64_bytes16640") return {64, 16640};
    if (c.variant == "dynamic_pitch65_bytes16640") return {65, 16640};
    return {};
}

unsigned runtime_pitch(const Case& c) {
    if (c.variant == "runtime_pitch64") return 64;
    if (c.variant == "runtime_pitch65") return 65;
    return dynamic_shared(c).pitch;
}

bool admitted_shape(uint64_t rows, uint64_t cols) {
    return (rows == 1 && (cols == 1 || cols == 65)) || (rows == 65 && cols == 1)
        || (rows == 31 && cols == 33) || (rows == 33 && cols == 31)
        || (rows >= 63 && rows <= 65 && cols >= 63 && cols <= 65)
        || (rows == 262144 && cols == 64) || (rows == 65536 && cols == 256)
        || (rows == 4096 && cols == 4096) || (rows == 262143 && cols == 63)
        || (rows == 4095 && cols == 4097);
}

dim3 block_shape(const Case& c) {
    return c.variant == "direct" ? dim3(256, 1, 1) : dim3(64, 4, 1);
}

dim3 grid_shape(const Case& c) {
    return c.variant == "direct" ? dim3(static_cast<unsigned>((c.n + 255) / 256), 1, 1)
                                : dim3(static_cast<unsigned>((c.cols + 63) / 64),
                                       static_cast<unsigned>((c.rows + 63) / 64), 1);
}

std::vector<Case> read_plan(const std::string& directory) {
    std::ifstream file(directory + "/cases.tsv");
    if (!file) throw std::runtime_error("Cannot open cases.tsv");
    std::string line;
    if (!std::getline(file, line) || line != "id\trows\tcols\tvariant\twarmups\tsamples\tlaunches")
        throw std::runtime_error("Unsupported cases.tsv header");
    std::vector<Case> cases;
    while (std::getline(file, line)) {
        std::istringstream row(line);
        Case c;
        std::string trailing;
        if (!(row >> c.id >> c.rows >> c.cols >> c.variant >> c.warmups >> c.samples >> c.launches)
            || (row >> trailing)) throw std::runtime_error("Invalid case row");
        if (c.id.empty() || c.id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos)
            throw std::runtime_error("Unsafe case id");
        for (const auto& old : cases)
            if (old.id == c.id) throw std::runtime_error("Duplicate case id");
        if (c.rows == 0 || c.cols == 0 || c.cols > kInputElements || c.rows > kInputElements / c.cols
            || !admitted_shape(c.rows, c.cols))
            throw std::runtime_error("Shape outside the fixed experiment plan or input capacity");
        c.n = c.rows * c.cols;
        if (c.variant != "direct" && c.variant != "tile64" && c.variant != "tile64_pad1" && runtime_pitch(c) == 0)
            throw std::runtime_error("Unknown transpose variant");
        const auto dynamic = dynamic_shared(c);
        if ((dynamic.pitch || dynamic.bytes) && ((dynamic.pitch != 64 && dynamic.pitch != 65)
                                                || dynamic.bytes % sizeof(float)
                                                || dynamic.bytes < 64 * dynamic.pitch * sizeof(float)))
            throw std::runtime_error("Insufficient or invalid dynamic shared capacity for 64 pitched rows");
        if (c.warmups != 10 || c.samples != 10 || c.launches != 10)
            throw std::runtime_error("Probe fixes 10 warmups and 10 samples of 10 launches");
        cases.push_back(c);
        if (cases.size() > 57) throw std::runtime_error("Maximum 57 cases");
    }
    if (!file.eof() || cases.empty()) throw std::runtime_error("Invalid or empty plan");
    return cases;
}

__global__ void transpose_direct_kernel(const float* __restrict__ input, float* __restrict__ output,
                                        uint64_t rows, uint64_t cols) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < rows * cols) output[i] = input[(i % rows) * cols + i / rows];
}

template <unsigned kSharedColumns>
__global__ void transpose_tile_kernel(const float* __restrict__ input, float* __restrict__ output,
                                      uint64_t rows, uint64_t cols) {
    __shared__ float tile[64][kSharedColumns];
    const uint64_t input_col = static_cast<uint64_t>(blockIdx.x) * 64 + threadIdx.x;
    const uint64_t input_row = static_cast<uint64_t>(blockIdx.y) * 64 + threadIdx.y;
    for (unsigned offset = 0; offset < 64; offset += 4) {
        if (input_col < cols && input_row + offset < rows)
            tile[threadIdx.y + offset][threadIdx.x] = input[(input_row + offset) * cols + input_col];
    }
    // Every thread reaches this barrier, including threads outside either matrix boundary.
    __syncthreads();
    const uint64_t output_col = static_cast<uint64_t>(blockIdx.y) * 64 + threadIdx.x;
    const uint64_t output_row = static_cast<uint64_t>(blockIdx.x) * 64 + threadIdx.y;
    for (unsigned offset = 0; offset < 64; offset += 4) {
        // This guard is also exactly the validity condition for tile[x][y+offset]'s load.
        if (output_col < rows && output_row + offset < cols)
            output[(output_row + offset) * rows + output_col] = tile[threadIdx.x][threadIdx.y + offset];
    }
}

// Both controls call this single non-template kernel with the same source storage capacity.
__global__ void transpose_runtime_pitch_kernel(const float* __restrict__ input, float* __restrict__ output,
                                               uint64_t rows, uint64_t cols, unsigned pitch) {
    __shared__ float storage[64 * 65];
    const uint64_t input_col = static_cast<uint64_t>(blockIdx.x) * 64 + threadIdx.x;
    const uint64_t input_row = static_cast<uint64_t>(blockIdx.y) * 64 + threadIdx.y;
    for (unsigned offset = 0; offset < 64; offset += 4) {
        if (input_col < cols && input_row + offset < rows)
            storage[(threadIdx.y + offset) * pitch + threadIdx.x] = input[(input_row + offset) * cols + input_col];
    }
    __syncthreads();
    const uint64_t output_col = static_cast<uint64_t>(blockIdx.y) * 64 + threadIdx.x;
    const uint64_t output_row = static_cast<uint64_t>(blockIdx.x) * 64 + threadIdx.y;
    for (unsigned offset = 0; offset < 64; offset += 4) {
        if (output_col < rows && output_row + offset < cols)
            output[(output_row + offset) * rows + output_col] = storage[threadIdx.x * pitch + threadIdx.y + offset];
    }
}

// The three dynamic controls differ only in runtime pitch and requested launch bytes.
__global__ void transpose_dynamic_shared_kernel(const float* __restrict__ input, float* __restrict__ output,
                                                uint64_t rows, uint64_t cols, unsigned pitch) {
    extern __shared__ float storage[];
    const uint64_t input_col = static_cast<uint64_t>(blockIdx.x) * 64 + threadIdx.x;
    const uint64_t input_row = static_cast<uint64_t>(blockIdx.y) * 64 + threadIdx.y;
    for (unsigned offset = 0; offset < 64; offset += 4) {
        if (input_col < cols && input_row + offset < rows)
            storage[(threadIdx.y + offset) * pitch + threadIdx.x] = input[(input_row + offset) * cols + input_col];
    }
    __syncthreads();
    const uint64_t output_col = static_cast<uint64_t>(blockIdx.y) * 64 + threadIdx.x;
    const uint64_t output_row = static_cast<uint64_t>(blockIdx.x) * 64 + threadIdx.y;
    for (unsigned offset = 0; offset < 64; offset += 4) {
        if (output_col < rows && output_row + offset < cols)
            output[(output_row + offset) * rows + output_col] = storage[threadIdx.x * pitch + threadIdx.y + offset];
    }
}

mcFuncAttributes function_attributes(const Case& c) {
    const void* function = c.variant == "direct" ? reinterpret_cast<const void*>(transpose_direct_kernel)
                         : c.variant == "tile64" ? reinterpret_cast<const void*>(transpose_tile_kernel<64>)
                         : c.variant == "tile64_pad1" ? reinterpret_cast<const void*>(transpose_tile_kernel<65>)
                         : dynamic_shared(c).bytes ? reinterpret_cast<const void*>(transpose_dynamic_shared_kernel)
                         : reinterpret_cast<const void*>(transpose_runtime_pitch_kernel);
    mcFuncAttributes attributes{};
    MC_CHECK(mcFuncGetAttributes(&attributes, function));
    return attributes;
}

void launch(const Case& c, const float* input, float* output) {
    const dim3 grid = grid_shape(c), block = block_shape(c);
    const auto dynamic = dynamic_shared(c);
    if (c.variant == "direct")
        transpose_direct_kernel<<<grid, block>>>(input, output, c.rows, c.cols);
    else if (c.variant == "tile64")
        transpose_tile_kernel<64><<<grid, block>>>(input, output, c.rows, c.cols);
    else if (c.variant == "tile64_pad1")
        transpose_tile_kernel<65><<<grid, block>>>(input, output, c.rows, c.cols);
    else if (dynamic.bytes)
        transpose_dynamic_shared_kernel<<<grid, block, dynamic.bytes>>>(input, output, c.rows, c.cols, dynamic.pitch);
    else
        transpose_runtime_pitch_kernel<<<grid, block>>>(input, output, c.rows, c.cols, runtime_pitch(c));
    MC_CHECK(mcGetLastError());
}

void run(const std::string& input_directory, const std::string& output_directory) {
    const uint16_t endian = 1;
    if (*reinterpret_cast<const uint8_t*>(&endian) != 1 || sizeof(float) != 4)
        throw std::runtime_error("Probe requires little-endian binary32 host");
    // Admit the complete plan and all file paths before any device API.
    const auto cases = read_plan(input_directory);
    std::vector<float> input(kInputElements);
    std::ifstream input_file(input_directory + "/input.f32", std::ios::binary);
    if (!input_file.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float)))
        throw std::runtime_error("Missing or truncated input.f32");
    if (input_file.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Extra input bytes");
    if (!std::filesystem::is_directory(output_directory) || !std::filesystem::is_empty(output_directory))
        throw std::runtime_error("Output directory must already exist and be empty");
    std::ofstream records(output_directory + "/raw.jsonl");
    if (!records) throw std::runtime_error("Cannot create raw.jsonl");
    records << std::setprecision(12);

    int device_count = 0;
    MC_CHECK(mcGetDeviceCount(&device_count));
    if (device_count != 1) throw std::runtime_error("Exactly one leased MACA device must be visible");
    MC_CHECK(mcSetDevice(0));
    mcDeviceProp_t property{};
    MC_CHECK(mcGetDeviceProperties(&property, 0));
    if (std::string(property.name) != "MetaX C550")
        throw std::runtime_error("Expected exact device name MetaX C550; observed " + std::string(property.name));
    if (property.maxThreadsPerBlock < 256) throw std::runtime_error("Runtime device limit below block 256");
    char pci_bus_id[64]{};
    MC_CHECK(mcDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0));
    int runtime_version = 0, driver_version = 0;
    MC_CHECK(mcRuntimeGetVersion(&runtime_version));
    MC_CHECK(mcDriverGetVersion(&driver_version));
    records << "{\"type\":\"device\",\"logical_device\":0,\"visible_device_count\":" << device_count
            << ",\"name\":" << json_string(property.name) << ",\"pci_bus_id\":" << json_string(pci_bus_id)
            << ",\"runtime_version_api\":" << runtime_version << ",\"driver_version_api\":" << driver_version
            << ",\"api_major\":" << property.major << ",\"api_minor\":" << property.minor
            << ",\"total_global_mem_bytes\":" << property.totalGlobalMem
            << ",\"wave_size_api\":" << property.waveSize
            << ",\"warp_size_compat_api\":" << property.warpSize
            << ",\"multiprocessor_count\":" << property.multiProcessorCount
            << ",\"l2_cache_bytes\":" << property.l2CacheSize
            << ",\"shared_mem_per_block_bytes\":" << property.sharedMemPerBlock
            << ",\"shared_mem_per_multiprocessor_bytes\":" << property.sharedMemPerMultiprocessor
            << ",\"max_threads_per_multiprocessor\":" << property.maxThreadsPerMultiProcessor
            << ",\"max_threads_per_block\":" << property.maxThreadsPerBlock << "}\n";
    records << "{\"type\":\"protocol\",\"schema_version\":1,\"experiment\":\"fp32-row-major-transpose\","
               "\"input_elements\":" << kInputElements << ",\"guard_elements_each_side\":" << kGuardElements
            << ",\"dtype\":\"float32\",\"timer\":\"mcEventElapsedTime\","
               "\"timed_scope\":\"default-stream events around 10 launches; may include device idle gaps during host submission\","
               "\"cache_policy\":\"repeated addresses; no explicit application cache reset; runtime cache policy unknown\","
               "\"transfer_scope\":\"allocation, initialization, transfers and file writes outside event intervals\","
               "\"throughput_scope\":\"logical read+write bytes; not measured DRAM traffic\","
               "\"function_attribute_query\":\"before each case's warmups and timing; runtime observations, not occupancy measurements\","
               "\"wave_size_note\":\"MACA 3.5.3 headers alias waveSize to warpSize; one API observation\","
               "\"MACA_LAUNCH_MODE\":" << json_environment("MACA_LAUNCH_MODE")
            << ",\"MACA_LAUNCH_BLOCKING\":" << json_environment("MACA_LAUNCH_BLOCKING")
            << ",\"MACA_DIRECT_DISPATCH\":" << json_environment("MACA_DIRECT_DISPATCH")
            << ",\"MACA_CACHE_PATH\":" << json_environment("MACA_CACHE_PATH")
            << ",\"MACA_CACHE_DISABLE\":" << json_environment("MACA_CACHE_DISABLE") << "}\n";
    records.flush();
    float* device_input = nullptr;
    MC_CHECK(mcMalloc(reinterpret_cast<void**>(&device_input), input.size() * sizeof(float)));
    MC_CHECK(mcMemcpy(device_input, input.data(), input.size() * sizeof(float), mcMemcpyHostToDevice));
    std::vector<float>().swap(input);
    // Reuse both device allocations so case order does not change base addresses.
    float* device_output = nullptr;
    MC_CHECK(mcMalloc(reinterpret_cast<void**>(&device_output),
                      (kInputElements + 2 * kGuardElements) * sizeof(float)));
    float* payload = device_output + kGuardElements;
    mcEvent_t start, stop;
    MC_CHECK(mcEventCreate(&start));
    MC_CHECK(mcEventCreate(&stop));
    for (const auto& c : cases) {
        const uint64_t words = c.n + 2 * kGuardElements;
        MC_CHECK(mcMemset(device_output, 0xff, words * sizeof(float)));
        const auto attributes = function_attributes(c);
        const dim3 grid = grid_shape(c), block = block_shape(c);
        const unsigned tile_size = c.variant == "direct" ? 0 : 64;
        const auto dynamic = dynamic_shared(c);
        const unsigned pitch = runtime_pitch(c);
        const unsigned padding = c.variant == "tile64_pad1" || pitch == 65 ? 1 : 0;
        const unsigned allocated_shared_elements = dynamic.bytes ? 0 : pitch ? 64 * 65 : tile_size * (tile_size + padding);
        records << "{\"type\":\"case\",\"id\":" << json_string(c.id)
                << ",\"rows\":" << c.rows << ",\"cols\":" << c.cols << ",\"n\":" << c.n
                << ",\"variant\":" << json_string(c.variant)
                << ",\"block_x\":" << block.x << ",\"block_y\":" << block.y << ",\"block_z\":" << block.z
                << ",\"grid_x\":" << grid.x << ",\"grid_y\":" << grid.y << ",\"grid_z\":" << grid.z
                << ",\"tile_rows\":" << tile_size << ",\"tile_cols\":" << tile_size << ",\"padding\":" << padding
                << ",\"intended_static_shared_bytes\":" << allocated_shared_elements * sizeof(float);
        if (dynamic.bytes)
            records << ",\"shared_pitch_elements\":" << dynamic.pitch
                    << ",\"requested_dynamic_shared_bytes\":" << dynamic.bytes
                    << ",\"requested_shared_elements\":" << dynamic.bytes / sizeof(float);
        else if (pitch)
            records << ",\"shared_pitch_elements\":" << pitch
                    << ",\"allocated_shared_elements\":" << allocated_shared_elements;
        records << ",\"function_attributes_before_timing\":{\"maxThreadsPerBlock\":" << attributes.maxThreadsPerBlock
                << ",\"numRegs\":" << attributes.numRegs << ",\"sharedSizeBytes\":" << attributes.sharedSizeBytes
                << ",\"localSizeBytes\":" << attributes.localSizeBytes;
        if (dynamic.bytes)
            records << ",\"maxDynamicSharedSizeBytes\":" << attributes.maxDynamicSharedSizeBytes;
        records << "},\"warmups\":" << c.warmups << ",\"samples\":" << c.samples
                << ",\"launches_per_sample\":" << c.launches
                << ",\"total_launches\":" << c.warmups + c.samples * c.launches
                << ",\"logical_bytes_per_launch\":" << c.n * 8
                << ",\"unique_input_elements\":" << c.n << ",\"input_span_bytes\":" << c.n * 4
                << ",\"output_file\":" << json_string(c.id + ".f32") << "}\n";
        records.flush();
        for (unsigned i = 0; i < c.warmups; ++i) launch(c, device_input, payload);
        MC_CHECK(mcDeviceSynchronize());
        for (unsigned sample = 0; sample < c.samples; ++sample) {
            MC_CHECK(mcEventRecord(start, 0));
            const auto host_start = std::chrono::steady_clock::now();
            for (unsigned i = 0; i < c.launches; ++i) launch(c, device_input, payload);
            const auto host_stop = std::chrono::steady_clock::now();
            MC_CHECK(mcEventRecord(stop, 0));
            MC_CHECK(mcEventSynchronize(stop));
            float elapsed_ms = 0;
            MC_CHECK(mcEventElapsedTime(&elapsed_ms, start, stop));
            const double enqueue_us = std::chrono::duration<double, std::micro>(host_stop - host_start).count();
            if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0 || !std::isfinite(enqueue_us) || enqueue_us <= 0)
                throw std::runtime_error("Nonpositive or nonfinite timing");
            records << "{\"type\":\"sample\",\"id\":" << json_string(c.id) << ",\"sample\":" << sample
                    << ",\"event_batch_ms\":" << elapsed_ms << ",\"host_enqueue_batch_us\":" << enqueue_us << "}\n";
        }
        std::vector<float> output(words);
        MC_CHECK(mcMemcpy(output.data(), device_output, words * sizeof(float), mcMemcpyDeviceToHost));
        std::ofstream output_file(output_directory + "/" + c.id + ".f32", std::ios::binary);
        output_file.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
        output_file.close();
        if (!output_file) throw std::runtime_error("Could not save device output");
        records.flush();
        if (!records) throw std::runtime_error("Could not save timing records");
    }
    MC_CHECK(mcEventDestroy(stop));
    MC_CHECK(mcEventDestroy(start));
    MC_CHECK(mcFree(device_output));
    MC_CHECK(mcFree(device_input));
    MC_CHECK(mcDeviceSynchronize());
    records << "{\"type\":\"complete\",\"cases\":" << cases.size() << ",\"cpu_correctness_checked\":false}\n";
    records.close();
    if (!records) throw std::runtime_error("Could not finish timing records");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 || std::string(argv[1]) != "--run") {
            std::cerr << "Usage: probe --run INPUT_DIRECTORY FRESH_EMPTY_OUTPUT_DIRECTORY\n";
            return 2;
        }
        run(argv[2], argv[3]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "{\"type\":\"error\",\"message\":" << json_string(error.what()) << "}\n";
        // Process exit releases allocations remaining after any checked failure.
        return 1;
    }
}
