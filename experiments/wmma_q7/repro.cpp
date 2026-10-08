// Single-case native MACA collector. CPU acceptance is a separate process.
#include <mcr/mc_runtime.h>
#include <__clang_maca_mma_functions.h>

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
constexpr unsigned kInputWords = 1024, kOutputWords = 384, kGuardWords = 64;
static_assert(sizeof(__half) == 2 && sizeof(float) == 4, "Requires FP16 inputs and FP32 outputs");

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

std::string environment_value(const char* key) {
    const char* value = std::getenv(key);
    return value ? json_string(value) : "null";
}

template <typename T>
void write_words(const std::string& directory, const std::string& name, const std::vector<T>& words) {
    std::ofstream file(directory + "/" + name, std::ios::binary);
    file.write(reinterpret_cast<const char*>(words.data()), words.size() * sizeof(T));
    file.close();
    if (!file) throw std::runtime_error("Could not retain " + name);
}

void flush_record(std::ostream& records) {
    records << '\n';
    records.flush();
    if (!records) throw std::runtime_error("Could not retain metadata");
}

// The following two device bodies are copied exactly from frozen 747c7b5.
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

__global__ void scalar_tile_kernel(const __half* a, const __half* b, float* output, unsigned chunks) {
    const unsigned index = threadIdx.x;
    const unsigned row = index / 16, col = index % 16;
    float total = 0.0f;
    for (unsigned k = 0; k < chunks * 16; ++k) {
        const unsigned chunk = k / 16, local_k = k % 16;
        const float left = __half2float(a[chunk * 256 + row * 16 + local_k]);
        const float right = __half2float(b[chunk * 256 + col * 16 + local_k]);
        total += left * right;
    }
    output[index] = total;
}

void launch_once(bool scalar, const __half* a, const __half* b, float* output) {
    if (scalar) scalar_tile_kernel<<<1, 256>>>(a, b, output, 1);
    else wmma_tile_kernel<<<1, 64>>>(a, b, output, 1);
    MC_CHECK(mcGetLastError());
    MC_CHECK(mcDeviceSynchronize());
}

uintptr_t observed_alignment(const void* pointer) {
    const uintptr_t value = reinterpret_cast<uintptr_t>(pointer);
    if (!value) throw std::runtime_error("Device allocation returned a null pointer");
    return value & (~value + 1);
}

void retain_snapshot(const std::string& directory, const std::string& order, const char* phase,
                     const char* after, const char* before, const __half* a, const __half* b,
                     std::ostream& records) {
    MC_CHECK(mcDeviceSynchronize());
    const __half* pointers[] = {a, b};
    const char* operands[] = {"a", "b"};
    for (unsigned index = 0; index < 2; ++index) {
        std::vector<uint16_t> words(kInputWords);
        MC_CHECK(mcMemcpy(words.data(), pointers[index], words.size() * sizeof(uint16_t), mcMemcpyDeviceToHost));
        write_words(directory, std::string(phase) + "." + operands[index] + ".f16", words);
    }
    records << "{\"type\":\"input_snapshot\",\"order\":" << json_string(order)
            << ",\"phase\":" << json_string(phase)
            << ",\"after_variant\":" << (after ? json_string(after) : "null")
            << ",\"before_variant\":" << (before ? json_string(before) : "null")
            << ",\"operand_halfwords_each\":1024,\"a_file\":" << json_string(std::string(phase) + ".a.f16")
            << ",\"b_file\":" << json_string(std::string(phase) + ".b.f16") << '}';
    flush_record(records);
}

void collect(const std::string& order, const std::string& directory) {
    if (order != "wmma-first" && order != "scalar-first")
        throw std::runtime_error("Order must be wmma-first or scalar-first");
    if (!std::filesystem::is_directory(directory) || !std::filesystem::is_empty(directory))
        throw std::runtime_error("Output directory must already exist and be empty");
    const uint16_t endian = 1;
    if (*reinterpret_cast<const uint8_t*>(&endian) != 1)
        throw std::runtime_error("Requires a little-endian host");

    // Fixed raw words; no arithmetic-based or observed-output-based input generation.
    std::vector<uint16_t> a(kInputWords, 0), b(kInputWords, 0);
    a[0] = 0xb700; a[1] = 0x2c00;
    b[0] = 0xac00; b[1] = 0xb800;
    write_words(directory, "prepared.a.f16", a);
    write_words(directory, "prepared.b.f16", b);
    std::ofstream records(directory + "/raw.jsonl");
    if (!records) throw std::runtime_error("Cannot create raw.jsonl");

    int count = 0;
    MC_CHECK(mcGetDeviceCount(&count));
    if (count != 1) throw std::runtime_error("Exactly one leased MACA device must be visible");
    MC_CHECK(mcSetDevice(0));
    mcDeviceProp_t properties{};
    MC_CHECK(mcGetDeviceProperties(&properties, 0));
    if (std::string(properties.name) != "MetaX C550" || properties.waveSize != 64)
        throw std::runtime_error("Expected exact device MetaX C550 with observed wave size 64");
    if (properties.maxThreadsPerBlock < 256)
        throw std::runtime_error("Device block limit is below 256");
    char pci[64]{};
    int runtime = 0, driver = 0;
    MC_CHECK(mcDeviceGetPCIBusId(pci, sizeof(pci), 0));
    MC_CHECK(mcRuntimeGetVersion(&runtime));
    MC_CHECK(mcDriverGetVersion(&driver));
    records << "{\"type\":\"device\",\"name\":" << json_string(properties.name)
            << ",\"logical_device\":0,\"visible_device_count\":" << count
            << ",\"wave_size_api\":" << properties.waveSize << ",\"pci_bus_id\":" << json_string(pci)
            << ",\"runtime_version_api\":" << runtime << ",\"driver_version_api\":" << driver
            << ",\"max_threads_per_block\":" << properties.maxThreadsPerBlock << '}';
    flush_record(records);

    const char* variants[] = {order == "wmma-first" ? "wmma" : "scalar", order == "wmma-first" ? "scalar" : "wmma"};
    records << "{\"type\":\"protocol\",\"schema\":\"metax-kernelwiki.wmma-q7-repro.v1\",\"order\":" << json_string(order)
            << ",\"variant_order\":[" << json_string(variants[0]) << ',' << json_string(variants[1]) << "],"
               "\"logical_shape\":[16,16,2],\"tile\":[16,16,16],\"packed_chunks\":4,\"k_chunks\":1,"
               "\"operand_halfwords_each\":1024,\"input_words_prefix\":{\"a\":[46848,11264],\"b\":[44032,47104]},"
               "\"input_padding_uint16\":0,\"prepared_files\":{\"a\":\"prepared.a.f16\",\"b\":\"prepared.b.f16\"},"
               "\"layouts\":{\"a\":\"row_major\",\"b\":\"col_major\",\"c\":\"row_major\"},\"leading_dimension\":16,"
               "\"operand_dtype\":\"float16\",\"accumulator_dtype\":\"float32\",\"output_dtype\":\"float32\","
               "\"output_elements\":256,\"guard_elements_each_side\":64,\"guard_uint32\":4294967295,"
               "\"initial_payload_uint32\":4294967295,\"launches_per_variant\":1,\"total_launches\":2,\"warmups\":0,"
               "\"input_rewrite_between_variants\":false,\"snapshot_phases\":[\"before\",\"between\",\"after\"],"
               "\"comparison\":\"all 256 outputs finite and numerically exact; signed zeros equivalent; no tolerance\",\"environment\":{";
    const char* environment[] = {"MACA_LAUNCH_MODE", "MACA_LAUNCH_BLOCKING", "MACA_DIRECT_DISPATCH", "MACA_CACHE_PATH", "MACA_CACHE_DISABLE"};
    for (unsigned index = 0; index < 5; ++index) {
        if (index) records << ',';
        records << json_string(environment[index]) << ':' << environment_value(environment[index]);
    }
    records << "}}";
    flush_record(records);

    __half *device_a = nullptr, *device_b = nullptr;
    float* device_c[2] = {nullptr, nullptr};  // Fixed slots: WMMA=0, scalar=1, independent of order.
    MC_CHECK(mcMalloc(reinterpret_cast<void**>(&device_a), a.size() * sizeof(uint16_t)));
    MC_CHECK(mcMalloc(reinterpret_cast<void**>(&device_b), b.size() * sizeof(uint16_t)));
    for (auto& pointer : device_c)
        MC_CHECK(mcMalloc(reinterpret_cast<void**>(&pointer), kOutputWords * sizeof(float)));
    MC_CHECK(mcMemcpy(device_a, a.data(), a.size() * sizeof(uint16_t), mcMemcpyHostToDevice));
    MC_CHECK(mcMemcpy(device_b, b.data(), b.size() * sizeof(uint16_t), mcMemcpyHostToDevice));
    retain_snapshot(directory, order, "before", nullptr, variants[0], device_a, device_b, records);

    for (unsigned stage = 0; stage < 2; ++stage) {
        const bool scalar = std::string(variants[stage]) == "scalar";
        const unsigned slot = scalar ? 1 : 0;
        float* base = device_c[slot];
        float* payload = base + kGuardWords;
        MC_CHECK(mcMemset(base, 0xff, kOutputWords * sizeof(float)));
        mcFuncAttributes attributes{};
        const void* function = scalar ? reinterpret_cast<const void*>(scalar_tile_kernel)
                                      : reinterpret_cast<const void*>(wmma_tile_kernel);
        MC_CHECK(mcFuncGetAttributes(&attributes, function));
        const uintptr_t aa = observed_alignment(device_a), ba = observed_alignment(device_b), ca = observed_alignment(payload);
        launch_once(scalar, device_a, device_b, payload);
        std::vector<uint32_t> output(kOutputWords);
        MC_CHECK(mcMemcpy(output.data(), base, output.size() * sizeof(uint32_t), mcMemcpyDeviceToHost));
        write_words(directory, std::string(variants[stage]) + ".f32", output);
        records << "{\"type\":\"variant\",\"order\":" << json_string(order)
                << ",\"variant\":" << json_string(variants[stage]) << ",\"kernel\":" << json_string(std::string(variants[stage]) + "_tile_kernel")
                << ",\"block\":[" << (scalar ? 256 : 64) << ",1,1],\"grid\":[1,1,1],\"k_chunks\":1,\"launch_count\":1,\"c_allocation_slot\":" << slot
                << ",\"output_file\":" << json_string(std::string(variants[stage]) + ".f32")
                << ",\"output_words\":384,\"payload_offset_words\":64,\"payload_elements\":256,\"guard_elements_each_side\":64,"
                   "\"function_attributes_before_launch\":{\"maxThreadsPerBlock\":" << attributes.maxThreadsPerBlock
                << ",\"numRegs\":" << attributes.numRegs << ",\"sharedSizeBytes\":" << attributes.sharedSizeBytes << ",\"localSizeBytes\":" << attributes.localSizeBytes << "},"
                   "\"pointer_alignment_observed_bytes\":{\"a\":" << aa << ",\"b\":" << ba << ",\"c_payload\":" << ca << "}}";
        flush_record(records);
        retain_snapshot(directory, order, stage == 0 ? "between" : "after", variants[stage],
                        stage == 0 ? variants[1] : nullptr, device_a, device_b, records);
    }
    for (auto pointer : device_c) MC_CHECK(mcFree(pointer));
    MC_CHECK(mcFree(device_b));
    MC_CHECK(mcFree(device_a));
    MC_CHECK(mcDeviceSynchronize());
    records << "{\"type\":\"complete\",\"order\":" << json_string(order)
            << ",\"variant_count\":2,\"total_launches\":2,\"outputs_retained\":true,\"device_buffers_freed\":true,\"cpu_correctness_checked\":false}";
    flush_record(records);
    records.close();
    if (!records) throw std::runtime_error("Could not finish metadata");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 4 || std::string(argv[1]) != "--run") {
        std::cerr << "Usage: repro --run {wmma-first|scalar-first} FRESH_EMPTY_OUTPUT_DIRECTORY\n";
        return 2;
    }
    try {
        collect(argv[2], argv[3]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "{\"type\":\"error\",\"message\":" << json_string(error.what()) << "}\n";
        return 1;
    }
}
