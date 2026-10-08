// Version-bound native MACA WMMA fragment probe; no cu-bridge or NVIDIA alias.
#include <mcr/mc_runtime.h>
#include <__clang_maca_mma_functions.h>

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
#include <utility>
#include <vector>

namespace {
constexpr unsigned kOperandHalfwords = 1024;
constexpr unsigned kOutputWords = 256;
constexpr unsigned kGuardWords = 64;
static_assert(sizeof(__half) == 2 && sizeof(float) == 4, "Probe requires FP16 operands and FP32 output");

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
    unsigned m = 0, n = 0, k = 0, warmups = 0, samples = 0, launches = 0;
    std::vector<uint16_t> a, b;
};

bool admitted_shape(const Case& c) {
    const unsigned shapes[][3] = {{16,16,0},{1,1,1},{16,16,16},{15,16,16},{16,15,16},{15,15,15},
                                 {7,9,17},{15,16,31},{16,15,32},{16,16,33},{9,7,63},{16,16,64}};
    for (const auto& shape : shapes)
        if (c.m == shape[0] && c.n == shape[1] && c.k == shape[2]) return true;
    return false;
}

uint16_t sixteenth_bits(int numerator) {
    // Exact normal half encoding for this contract's integer numerators in [-15,15].
    if (!numerator) return 0;
    const unsigned magnitude = numerator < 0 ? -numerator : numerator;
    unsigned exponent = 0;
    while ((1U << (exponent + 1)) <= magnitude) ++exponent;
    return static_cast<uint16_t>((numerator < 0 ? 0x8000 : 0)
           | ((exponent + 11) << 10) | ((magnitude - (1U << exponent)) << (10 - exponent)));
}

std::vector<uint16_t> read_operand(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<uint16_t> words(kOperandHalfwords);
    if (!file.read(reinterpret_cast<char*>(words.data()), words.size() * sizeof(uint16_t))
        || file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Missing or incorrect packed operand file extent");
    return words;
}

void validate_inputs(const Case& c) {
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        for (unsigned outer = 0; outer < 16; ++outer) {
            for (unsigned inner = 0; inner < 16; ++inner) {
                const unsigned k = chunk * 16 + inner;
                const unsigned index = chunk * 256 + outer * 16 + inner;
                const int a_num = outer < c.m && k < c.k ? static_cast<int>((67 * outer + 13 * k) % 31) - 15 : 0;
                const int b_num = outer < c.n && k < c.k ? static_cast<int>((17 * k + 5 * outer + 3) % 29) - 14 : 0;
                if (c.a[index] != sixteenth_bits(a_num) || c.b[index] != sixteenth_bits(b_num))
                    throw std::runtime_error("Packed input words differ from the declared A-row/B-column contract");
            }
        }
    }
}

std::vector<Case> read_plan(const std::string& directory) {
    std::ifstream file(directory + "/cases.tsv");
    std::string line;
    if (!file || !std::getline(file, line) || line != "id\tm\tn\tk\twarmups\tsamples\tlaunches")
        throw std::runtime_error("Unsupported or missing cases.tsv");
    std::vector<Case> cases;
    while (std::getline(file, line)) {
        std::istringstream row(line);
        Case c;
        std::string trailing;
        if (!(row >> c.id >> c.m >> c.n >> c.k >> c.warmups >> c.samples >> c.launches) || (row >> trailing))
            throw std::runtime_error("Invalid case row");
        if (c.id.empty() || c.id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos)
            throw std::runtime_error("Unsafe case id");
        for (const auto& old : cases)
            if (old.id == c.id) throw std::runtime_error("Duplicate case id");
        if (!admitted_shape(c)) throw std::runtime_error("Shape outside the fixed WMMA boundary cases");
        if (c.warmups != 10 || c.samples != 10 || c.launches != 10)
            throw std::runtime_error("Probe fixes 10 warmups and 10 samples of 10 launches");
        c.a = read_operand(directory + "/" + c.id + ".a.f16");
        c.b = read_operand(directory + "/" + c.id + ".b.f16");
        validate_inputs(c);
        cases.push_back(std::move(c));
        if (cases.size() > 12) throw std::runtime_error("Maximum 12 cases");
    }
    if (!file.eof() || cases.empty()) throw std::runtime_error("Invalid or empty plan");
    return cases;
}

__global__ void wmma_tile_kernel(const __half* a, const __half* b, float* output, unsigned chunks) {
    using namespace mxmaca::wmma;
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_fragment;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_fragment;
    fragment<accumulator, 16, 16, 16, float> accumulator_fragment;
    fill_fragment(accumulator_fragment, 0.0f);
    // Every physical thread follows the same loop, including the zero-chunk case.
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        load_matrix_sync(a_fragment, a + chunk * 256, 16);
        load_matrix_sync(b_fragment, b + chunk * 256, 16);
        mma_sync(accumulator_fragment, a_fragment, b_fragment, accumulator_fragment);
    }
    store_matrix_sync(output, accumulator_fragment, 16, mem_row_major);
}

void launch(const Case& c, const __half* a, const __half* b, float* output) {
    wmma_tile_kernel<<<1, 64>>>(a, b, output, (c.k + 15) / 16);
    MC_CHECK(mcGetLastError());
}

uintptr_t observed_alignment(const void* pointer) {
    const uintptr_t value = reinterpret_cast<uintptr_t>(pointer);
    if (!value) throw std::runtime_error("Device allocation returned a null pointer");
    return value & (~value + 1);
}

void run(const std::string& input_directory, const std::string& output_directory) {
    const uint16_t endian = 1;
    if (*reinterpret_cast<const uint8_t*>(&endian) != 1)
        throw std::runtime_error("Probe requires a little-endian host");
    // Complete host file admission and packing validation precede every device API.
    const auto cases = read_plan(input_directory);
    if (!std::filesystem::is_directory(output_directory) || !std::filesystem::is_empty(output_directory))
        throw std::runtime_error("Output directory must already exist and be empty");
    std::ofstream records(output_directory + "/raw.jsonl");
    if (!records) throw std::runtime_error("Cannot create raw.jsonl");
    records << std::setprecision(12);
    int count = 0;
    MC_CHECK(mcGetDeviceCount(&count));
    if (count != 1) throw std::runtime_error("Exactly one leased MACA device must be visible");
    MC_CHECK(mcSetDevice(0));
    mcDeviceProp_t property{};
    MC_CHECK(mcGetDeviceProperties(&property, 0));
    if (std::string(property.name) != "MetaX C550" || property.waveSize != 64)
        throw std::runtime_error("Expected exact device MetaX C550 with observed wave size 64");
    if (property.maxThreadsPerBlock < 64) throw std::runtime_error("Runtime device limit below block64");
    char pci[64]{};
    MC_CHECK(mcDeviceGetPCIBusId(pci, sizeof(pci), 0));
    int runtime_version = 0, driver_version = 0;
    MC_CHECK(mcRuntimeGetVersion(&runtime_version));
    MC_CHECK(mcDriverGetVersion(&driver_version));
    records << "{\"type\":\"device\",\"logical_device\":0,\"visible_device_count\":" << count
            << ",\"name\":" << json_string(property.name) << ",\"pci_bus_id\":" << json_string(pci)
            << ",\"runtime_version_api\":" << runtime_version << ",\"driver_version_api\":" << driver_version
            << ",\"api_major\":" << property.major << ",\"api_minor\":" << property.minor
            << ",\"total_global_mem_bytes\":" << property.totalGlobalMem
            << ",\"wave_size_api\":" << property.waveSize << ",\"warp_size_compat_api\":" << property.warpSize
            << ",\"multiprocessor_count\":" << property.multiProcessorCount
            << ",\"l2_cache_bytes\":" << property.l2CacheSize
            << ",\"shared_mem_per_block_bytes\":" << property.sharedMemPerBlock
            << ",\"max_threads_per_multiprocessor\":" << property.maxThreadsPerMultiProcessor
            << ",\"max_threads_per_block\":" << property.maxThreadsPerBlock << "}\n";
    records << "{\"type\":\"protocol\",\"schema_version\":1,\"experiment\":\"native-wmma-fp16-fp32-16x16\","
               "\"operand_dtype\":\"float16\",\"accumulator_dtype\":\"float32\",\"output_elements\":256,"
               "\"guard_elements_each_side\":64,\"required_wave_size\":64,\"timer\":\"mcEventElapsedTime\","
               "\"comparison\":\"finite exact numeric equality; signed zero equivalent; no tolerance\","
               "\"header_route\":\"native __clang_maca_mma_functions.h; mxmaca::wmma; no cu-bridge\","
               "\"timed_scope\":\"default-stream events around 10 full tile launches; may include host submission gaps\","
               "\"transfer_scope\":\"CPU packing, allocation, H2D/D2H copies, reset, attribute queries and file writes excluded\","
               "\"cache_policy\":\"repeated addresses; no explicit application cache reset; runtime cache policy unknown\","
               "\"interpretation\":\"descriptive full tile kernel; no native-instruction throughput or end-to-end GEMM claim\","
               "\"pointer_alignment_scope\":\"largest power of two dividing each observed device address; no imported alignment requirement\","
               "\"MACA_LAUNCH_MODE\":" << json_environment("MACA_LAUNCH_MODE")
            << ",\"MACA_LAUNCH_BLOCKING\":" << json_environment("MACA_LAUNCH_BLOCKING")
            << ",\"MACA_DIRECT_DISPATCH\":" << json_environment("MACA_DIRECT_DISPATCH")
            << ",\"MACA_CACHE_PATH\":" << json_environment("MACA_CACHE_PATH")
            << ",\"MACA_CACHE_DISABLE\":" << json_environment("MACA_CACHE_DISABLE") << "}\n";
    records.flush();
    __half *device_a = nullptr, *device_b = nullptr;
    float* device_c = nullptr;
    MC_CHECK(mcMalloc(reinterpret_cast<void**>(&device_a), kOperandHalfwords * sizeof(__half)));
    MC_CHECK(mcMalloc(reinterpret_cast<void**>(&device_b), kOperandHalfwords * sizeof(__half)));
    MC_CHECK(mcMalloc(reinterpret_cast<void**>(&device_c), (kOutputWords + 2 * kGuardWords) * sizeof(float)));
    float* payload = device_c + kGuardWords;
    const uintptr_t a_alignment = observed_alignment(device_a), b_alignment = observed_alignment(device_b);
    const uintptr_t c_alignment = observed_alignment(payload);
    mcEvent_t start, stop;
    MC_CHECK(mcEventCreate(&start));
    MC_CHECK(mcEventCreate(&stop));
    for (const auto& c : cases) {
        MC_CHECK(mcMemcpy(device_a, c.a.data(), c.a.size() * sizeof(uint16_t), mcMemcpyHostToDevice));
        MC_CHECK(mcMemcpy(device_b, c.b.data(), c.b.size() * sizeof(uint16_t), mcMemcpyHostToDevice));
        MC_CHECK(mcMemset(device_c, 0xff, (kOutputWords + 2 * kGuardWords) * sizeof(float)));
        mcFuncAttributes attributes{};
        MC_CHECK(mcFuncGetAttributes(&attributes, reinterpret_cast<const void*>(wmma_tile_kernel)));
        records << "{\"type\":\"case\",\"id\":" << json_string(c.id)
                << ",\"m\":" << c.m << ",\"n\":" << c.n << ",\"k\":" << c.k
                << ",\"tile_m\":16,\"tile_n\":16,\"tile_k\":16,\"k_chunks\":" << (c.k + 15) / 16
                << ",\"packed_chunks\":4,\"a_layout\":\"row_major\",\"b_layout\":\"col_major\",\"c_layout\":\"row_major\","
                   "\"leading_dimension\":16,\"operand_dtype\":\"float16\",\"accumulator_dtype\":\"float32\","
                   "\"output_elements\":256,\"operand_halfwords_each\":1024,\"physical_threads\":64,"
                   "\"block_x\":64,\"block_y\":1,\"block_z\":1,\"grid_x\":1,\"grid_y\":1,\"grid_z\":1,"
                   "\"required_wave_size\":64,\"participation\":\"all 64 physical threads; uniform K-chunk loop\","
                   "\"a_file\":" << json_string(c.id + ".a.f16") << ",\"b_file\":" << json_string(c.id + ".b.f16")
                << ",\"output_file\":" << json_string(c.id + ".f32") << ",\"guard_elements_each_side\":64,"
                   "\"warmups\":10,\"samples\":10,\"launches_per_sample\":10,\"total_launches\":110,"
                   "\"pointer_alignment_observed_bytes\":{\"a\":" << a_alignment << ",\"b\":" << b_alignment << ",\"c_payload\":" << c_alignment << "},"
                   "\"function_attributes_before_timing\":{\"maxThreadsPerBlock\":" << attributes.maxThreadsPerBlock
                << ",\"numRegs\":" << attributes.numRegs << ",\"sharedSizeBytes\":" << attributes.sharedSizeBytes
                << ",\"localSizeBytes\":" << attributes.localSizeBytes << "}}\n";
        records.flush();
        for (unsigned i = 0; i < c.warmups; ++i) launch(c, device_a, device_b, payload);
        MC_CHECK(mcDeviceSynchronize());
        for (unsigned sample = 0; sample < c.samples; ++sample) {
            MC_CHECK(mcEventRecord(start, 0));
            const auto host_start = std::chrono::steady_clock::now();
            for (unsigned i = 0; i < c.launches; ++i) launch(c, device_a, device_b, payload);
            const auto host_stop = std::chrono::steady_clock::now();
            MC_CHECK(mcEventRecord(stop, 0));
            MC_CHECK(mcEventSynchronize(stop));
            float elapsed_ms = 0;
            MC_CHECK(mcEventElapsedTime(&elapsed_ms, start, stop));
            const double host_us = std::chrono::duration<double, std::micro>(host_stop - host_start).count();
            if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0 || !std::isfinite(host_us) || host_us <= 0)
                throw std::runtime_error("Nonpositive or nonfinite timing");
            records << "{\"type\":\"sample\",\"id\":" << json_string(c.id) << ",\"sample\":" << sample
                    << ",\"event_batch_ms\":" << elapsed_ms << ",\"host_enqueue_batch_us\":" << host_us << "}\n";
        }
        std::vector<float> output(kOutputWords + 2 * kGuardWords);
        MC_CHECK(mcMemcpy(output.data(), device_c, output.size() * sizeof(float), mcMemcpyDeviceToHost));
        std::ofstream file(output_directory + "/" + c.id + ".f32", std::ios::binary);
        file.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
        file.close();
        if (!file) throw std::runtime_error("Could not retain complete C output and guards");
        records.flush();
        if (!records) throw std::runtime_error("Could not retain raw records");
    }
    MC_CHECK(mcEventDestroy(stop));
    MC_CHECK(mcEventDestroy(start));
    MC_CHECK(mcFree(device_c));
    MC_CHECK(mcFree(device_b));
    MC_CHECK(mcFree(device_a));
    MC_CHECK(mcDeviceSynchronize());
    records << "{\"type\":\"complete\",\"cases\":" << cases.size() << ",\"cpu_correctness_checked\":false}\n";
    records.close();
    if (!records) throw std::runtime_error("Could not finish raw records");
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
        return 1;
    }
}
