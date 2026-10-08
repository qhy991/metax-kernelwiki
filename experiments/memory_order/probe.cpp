// Fixed-footprint read-order experiment; independent of native gather semantics.
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
    std::string id;
    uint64_t n = 0;
    unsigned shift = 0, block = 0, warmups = 0, samples = 0, launches = 0, log2_n = 0;
};

std::vector<Case> read_plan(const std::string& directory) {
    std::ifstream file(directory + "/cases.tsv");
    if (!file) throw std::runtime_error("Cannot open cases.tsv");
    std::string line;
    if (!std::getline(file, line) || line != "id\tn\tshift\tblock\twarmups\tsamples\tlaunches")
        throw std::runtime_error("Unsupported cases.tsv header");
    std::vector<Case> cases;
    while (std::getline(file, line)) {
        std::istringstream row(line);
        Case c;
        std::string trailing;
        if (!(row >> c.id >> c.n >> c.shift >> c.block >> c.warmups >> c.samples >> c.launches)
            || (row >> trailing)) throw std::runtime_error("Invalid case row");
        if (c.id.empty() || c.id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos)
            throw std::runtime_error("Unsafe case id");
        for (const auto& old : cases)
            if (old.id == c.id) throw std::runtime_error("Duplicate case id");
        if (c.n != (1ULL << 16) && c.n != (1ULL << 20) && c.n != kInputElements)
            throw std::runtime_error("Length must be 2^16, 2^20 or 2^24");
        for (uint64_t value = c.n; value > 1; value >>= 1) ++c.log2_n;
        if (c.shift >= c.log2_n || (c.shift != 0 && c.shift != 2 && c.shift != 4
                                  && c.shift != 6 && c.shift != 8 && c.shift != 12))
            throw std::runtime_error("Shift outside the experiment plan or >= log2(n)");
        if (c.block != 256 || c.warmups != 10 || c.samples != 10 || c.launches != 10)
            throw std::runtime_error("Probe fixes block 256, 10 warmups and 10 samples of 10 launches");
        cases.push_back(c);
        if (cases.size() > 18) throw std::runtime_error("Maximum 18 cases");
    }
    if (!file.eof() || cases.empty()) throw std::runtime_error("Invalid or empty plan");
    return cases;
}

// Every case calls this one kernel with runtime n/shift/log2_n arguments.
__global__ void memory_order_kernel(const float* __restrict__ input, float* __restrict__ output,
                                    uint64_t n, unsigned shift, unsigned log2_n) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const uint64_t j = ((i << shift) | (i >> (log2_n - shift))) & (n - 1);
        output[i] = input[j];
    }
}

void launch(const Case& c, const float* input, float* output) {
    memory_order_kernel<<<static_cast<unsigned>(c.n / c.block), c.block>>>(input, output, c.n, c.shift, c.log2_n);
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
            << ",\"max_threads_per_multiprocessor\":" << property.maxThreadsPerMultiProcessor
            << ",\"max_threads_per_block\":" << property.maxThreadsPerBlock << "}\n";
    records << "{\"type\":\"protocol\",\"schema_version\":1,\"experiment\":\"fixed-footprint-memory-order\","
               "\"input_elements\":" << kInputElements << ",\"guard_elements_each_side\":" << kGuardElements
            << ",\"dtype\":\"float32\",\"timer\":\"mcEventElapsedTime\","
               "\"timed_scope\":\"default-stream events around 10 launches; may include device idle gaps during host submission\","
               "\"cache_policy\":\"repeated addresses; no explicit application cache reset; runtime cache policy unknown\","
               "\"transfer_scope\":\"allocation, initialization, transfers and file writes outside event intervals\","
               "\"throughput_scope\":\"logical read+write bytes; not measured DRAM traffic\","
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
        records << "{\"type\":\"case\",\"id\":" << json_string(c.id)
                << ",\"n\":" << c.n << ",\"shift\":" << c.shift << ",\"log2_n\":" << c.log2_n
                << ",\"block\":" << c.block << ",\"grid\":" << c.n / c.block
                << ",\"warmups\":" << c.warmups << ",\"samples\":" << c.samples
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
