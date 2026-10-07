// Standalone MACA runtime probe. See README.md for preparation and timing scope.
#include <mcr/mc_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef C550_COPY_LAUNCH_BOUND
#define C550_COPY_LAUNCH_BOUND 0
#endif
#if C550_COPY_LAUNCH_BOUND != 0 && C550_COPY_LAUNCH_BOUND != 1024
#error "C550_COPY_LAUNCH_BOUND must be 0 or 1024"
#endif

namespace {
constexpr uint64_t kInputElements = 1ULL << 24;  // 64 MiB, exact float indices.
constexpr uint64_t kGuardElements = 32;
constexpr uint64_t kMaxOutputElements = (1ULL << 24) - 64;

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
    std::string id, kind;
    uint64_t n = 0, stride = 0;
    unsigned block = 0, warmups = 0, samples = 0, launches = 0;
};

std::vector<Case> read_plan(const std::string& directory) {
    std::ifstream file(directory + "/cases.tsv");
    if (!file) throw std::runtime_error("Cannot open cases.tsv");
    std::string line;
    if (!std::getline(file, line) || line != "id\tkind\tn\tstride\tblock\twarmups\tsamples\tlaunches")
        throw std::runtime_error("Unsupported cases.tsv header");
    std::vector<Case> cases;
    while (std::getline(file, line)) {
        std::istringstream row(line);
        Case c;
        std::string trailing;
        if (!(row >> c.id >> c.kind >> c.n >> c.stride >> c.block >> c.warmups >> c.samples >> c.launches)
            || (row >> trailing)) throw std::runtime_error("Invalid case row");
        if (c.id.empty() || c.id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos)
            throw std::runtime_error("Unsafe case id");
        for (const auto& old : cases)
            if (old.id == c.id) throw std::runtime_error("Duplicate case id");
        if (c.block != 64 && c.block != 128 && c.block != 256 && c.block != 512 && c.block != 1024)
            throw std::runtime_error("Block must be 64, 128, 256, 512 or 1024");
        if (c.warmups != 20 || c.samples != 10 || c.launches != 100)
            throw std::runtime_error("Probe fixes 20 warmups and 10 samples of 100 launches");
        if (c.kind == "empty") {
            if (c.n != 0 || c.stride != 0) throw std::runtime_error("Invalid empty case");
        } else {
            if (c.kind != "copy" && c.kind != "gather") throw std::runtime_error("Unknown case kind");
            if (c.n == 0 || c.n > kMaxOutputElements || c.stride == 0 || c.stride > 16
                || (c.n - 1) * c.stride >= kInputElements)
                throw std::runtime_error("Case exceeds input/output limits");
            if (c.kind == "copy" && c.stride != 1) throw std::runtime_error("Copy stride must be one");
        }
        cases.push_back(c);
        if (cases.size() > 64) throw std::runtime_error("Maximum 64 cases");
    }
    if (!file.eof() || cases.empty()) throw std::runtime_error("Invalid or empty plan");
    return cases;
}

#if C550_COPY_LAUNCH_BOUND == 1024
__global__ __launch_bounds__(1024)
#else
__global__
#endif
void copy_kernel(const float* __restrict__ input, float* __restrict__ output, uint64_t n) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i];
}

__global__ void gather_kernel(const float* __restrict__ input, float* __restrict__ output,
                              uint64_t n, uint64_t stride) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i * stride];
}

__global__ void empty_kernel() {}

void launch(const Case& c, const float* input, float* output) {
    const unsigned grid = c.kind == "empty" ? 1 : static_cast<unsigned>((c.n + c.block - 1) / c.block);
    if (c.kind == "copy") copy_kernel<<<grid, c.block>>>(input, output, c.n);
    else if (c.kind == "gather") gather_kernel<<<grid, c.block>>>(input, output, c.n, c.stride);
    else empty_kernel<<<grid, c.block>>>();
    MC_CHECK(mcGetLastError());
}

void run(const std::string& input_directory, const std::string& output_directory,
         bool record_first_launch) {
    const uint16_t endian = 1;
    if (*reinterpret_cast<const uint8_t*>(&endian) != 1 || sizeof(float) != 4)
        throw std::runtime_error("Probe requires little-endian binary32 host");
    // All file admission precedes the first device API call.
    const auto cases = read_plan(input_directory);
    std::vector<float> input(kInputElements);
    std::ifstream input_file(input_directory + "/input.f32", std::ios::binary);
    if (!input_file.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float)))
        throw std::runtime_error("Missing or truncated input.f32");
    if (input_file.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Extra input bytes");
    const std::string result_path = output_directory + "/raw.jsonl";
    if (std::ifstream(result_path)) throw std::runtime_error("Output already exists; choose a fresh directory");
    std::ofstream records(result_path);
    if (!records) throw std::runtime_error("Cannot create raw.jsonl; create output directory first");
    records << std::setprecision(12);

    int device_count = 0;
    MC_CHECK(mcGetDeviceCount(&device_count));
    if (device_count != 1) throw std::runtime_error("Exactly one leased MACA device must be visible");
    MC_CHECK(mcSetDevice(0));
    mcDeviceProp_t property{};
    MC_CHECK(mcGetDeviceProperties(&property, 0));
    if (std::string(property.name) != "MetaX C550")
        throw std::runtime_error("Expected exact device name MetaX C550; observed " + std::string(property.name));
    char pci_bus_id[64]{};
    MC_CHECK(mcDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0));
    records << "{\"type\":\"device\",\"logical_device\":0,\"visible_device_count\":" << device_count
            << ",\"name\":" << json_string(property.name)
            << ",\"pci_bus_id\":" << json_string(pci_bus_id)
            << ",\"api_major\":" << property.major << ",\"api_minor\":" << property.minor
            << ",\"total_global_mem_bytes\":" << property.totalGlobalMem
            << ",\"wave_size_api\":" << property.waveSize
            << ",\"warp_size_compat_api\":" << property.warpSize
            << ",\"multiprocessor_count\":" << property.multiProcessorCount
            << ",\"l2_cache_bytes\":" << property.l2CacheSize
            << ",\"shared_mem_per_block_bytes\":" << property.sharedMemPerBlock
            << ",\"max_threads_per_multiprocessor\":" << property.maxThreadsPerMultiProcessor
            << ",\"max_threads_per_block\":" << property.maxThreadsPerBlock << "}\n";
    records << "{\"type\":\"protocol\",\"schema_version\":1,\"input_elements\":" << kInputElements
            << ",\"guard_elements_each_side\":" << kGuardElements
            << ",\"copy_launch_bound\":" << C550_COPY_LAUNCH_BOUND
            << ",\"dtype\":\"float32\",\"timer\":\"mcEventElapsedTime\","
               "\"timed_scope\":\"default-stream event interval around a batch; includes device idle gaps from host submission\","
               "\"cache_policy\":\"repeated addresses; no explicit application cache reset; runtime cache policy unverified\","
               "\"transfer_scope\":\"host-device transfers and allocation outside timed interval\","
               "\"throughput_scope\":\"logical read+write bytes; not measured DRAM traffic\","
               "\"wave_size_note\":\"MACA 3.5.3 headers alias waveSize to warpSize; both fields are one API observation\","
               "\"MACA_LAUNCH_MODE\":" << json_environment("MACA_LAUNCH_MODE")
            << ",\"MACA_LAUNCH_BLOCKING\":" << json_environment("MACA_LAUNCH_BLOCKING")
            << ",\"MACA_DIRECT_DISPATCH\":" << json_environment("MACA_DIRECT_DISPATCH");
    if (record_first_launch) {
        records << ",\"record_first_launch\":true,\"additional_launches_per_case\":1,"
                   "\"first_launch_timing_scope\":\"host monotonic clock around one launch, error check and device synchronization; "
                   "includes any lazy initialization/JIT triggered there; prior device work synchronized before timing; "
                   "before 20 warmups; not pure GPU latency or a fresh process per case\"";
        records << ",\"MACA_CACHE_PATH\":" << json_environment("MACA_CACHE_PATH")
                << ",\"MACA_CACHE_DISABLE\":" << json_environment("MACA_CACHE_DISABLE");
    }
    records << "}\n";
    records.flush();
    float* device_input = nullptr;
    MC_CHECK(mcMalloc(reinterpret_cast<void**>(&device_input), input.size() * sizeof(float)));
    MC_CHECK(mcMemcpy(device_input, input.data(), input.size() * sizeof(float), mcMemcpyHostToDevice));
    std::vector<float>().swap(input);
    mcEvent_t start, stop;
    MC_CHECK(mcEventCreate(&start));
    MC_CHECK(mcEventCreate(&stop));
    for (const auto& c : cases) {
        if (c.block > static_cast<unsigned>(property.maxThreadsPerBlock))
            throw std::runtime_error("Case block exceeds runtime device maximum");
        const uint64_t words = c.n + 2 * kGuardElements;
        float* device_output = nullptr;
        if (c.kind != "empty") {
            MC_CHECK(mcMalloc(reinterpret_cast<void**>(&device_output), words * sizeof(float)));
            MC_CHECK(mcMemset(device_output, 0xff, words * sizeof(float)));
        }
        records << "{\"type\":\"case\",\"id\":" << json_string(c.id)
                << ",\"kind\":" << json_string(c.kind) << ",\"n\":" << c.n
                << ",\"stride\":" << c.stride << ",\"block\":" << c.block
                << ",\"grid\":" << (c.kind == "empty" ? 1 : (c.n + c.block - 1) / c.block)
                << ",\"warmups\":" << c.warmups << ",\"samples\":" << c.samples
                << ",\"launches_per_sample\":" << c.launches
                << ",\"logical_bytes_per_launch\":" << c.n * 2 * sizeof(float)
                << ",\"unique_input_elements\":" << c.n
                << ",\"input_span_bytes\":" << (c.n ? ((c.n - 1) * c.stride + 1) * sizeof(float) : 0)
                << ",\"output_file\":" << (c.kind == "empty" ? "null" : json_string(c.id + ".f32")) << "}\n";
        float* payload = device_output ? device_output + kGuardElements : nullptr;
        if (record_first_launch) {
            // Preserve the active case even if its first launch or synchronization fails.
            records.flush();
            // Drain allocation/memset and prior cases outside this host interval.
            MC_CHECK(mcDeviceSynchronize());
            const auto first_start = std::chrono::steady_clock::now();
            launch(c, device_input, payload);
            MC_CHECK(mcDeviceSynchronize());
            const auto first_stop = std::chrono::steady_clock::now();
            const double first_us = std::chrono::duration<double, std::micro>(first_stop - first_start).count();
            if (!std::isfinite(first_us) || first_us <= 0)
                throw std::runtime_error("Nonpositive or nonfinite first-launch host timing");
            // Query only after timing: the query may itself trigger lazy loading.
            mcFuncAttributes attributes{};
            const void* function = c.kind == "copy" ? reinterpret_cast<const void*>(copy_kernel)
                                 : c.kind == "gather" ? reinterpret_cast<const void*>(gather_kernel)
                                 : reinterpret_cast<const void*>(empty_kernel);
            MC_CHECK(mcFuncGetAttributes(&attributes, function));
            records << "{\"type\":\"first_launch\",\"id\":" << json_string(c.id)
                    << ",\"first_launch_host_complete_us\":" << first_us
                    << ",\"function_attributes_after_first_launch\":{\"maxThreadsPerBlock\":" << attributes.maxThreadsPerBlock
                    << ",\"numRegs\":" << attributes.numRegs
                    << ",\"sharedSizeBytes\":" << attributes.sharedSizeBytes
                    << ",\"localSizeBytes\":" << attributes.localSizeBytes << "}}\n";
            records.flush();
        }
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
            if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0)
                throw std::runtime_error("Nonpositive or nonfinite device timing");
            const double enqueue_us = std::chrono::duration<double, std::micro>(host_stop - host_start).count();
            records << "{\"type\":\"sample\",\"id\":" << json_string(c.id)
                    << ",\"sample\":" << sample << ",\"event_batch_ms\":" << elapsed_ms
                    << ",\"host_enqueue_batch_us\":" << enqueue_us << "}\n";
        }
        if (device_output) {
            std::vector<float> output(words);
            MC_CHECK(mcMemcpy(output.data(), device_output, words * sizeof(float), mcMemcpyDeviceToHost));
            MC_CHECK(mcFree(device_output));
            std::ofstream output_file(output_directory + "/" + c.id + ".f32", std::ios::binary);
            output_file.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
            output_file.close();
            if (!output_file) throw std::runtime_error("Could not save device output");
        }
        records.flush();
        if (!records) throw std::runtime_error("Could not save timing records");
    }
    MC_CHECK(mcEventDestroy(stop));
    MC_CHECK(mcEventDestroy(start));
    MC_CHECK(mcFree(device_input));
    MC_CHECK(mcDeviceSynchronize());
    records << "{\"type\":\"complete\",\"cases\":" << cases.size()
            << ",\"cpu_correctness_checked\":false}\n";
    records.close();
    if (!records) throw std::runtime_error("Could not finish timing records");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if ((argc != 4 && argc != 5) || std::string(argv[1]) != "--run"
            || (argc == 5 && std::string(argv[4]) != "--record-first-launch")) {
            std::cerr << "Usage: probe --run INPUT_DIRECTORY FRESH_OUTPUT_DIRECTORY [--record-first-launch]\n";
            return 2;
        }
        run(argv[2], argv[3], argc == 5);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "{\"type\":\"error\",\"message\":" << json_string(error.what()) << "}\n";
        // Process exit releases any allocations still held after a runtime failure.
        return 1;
    }
}
